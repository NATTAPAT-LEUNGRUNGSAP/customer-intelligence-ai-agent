"""Leakage-aware predictive models for customer and revenue questions.

Classification labels are built from historical rolling snapshots. Features
only use transactions at or before each snapshot; labels only use transactions
after it. Revenue forecasting uses chronological holdout validation and
recursive weekly predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.features import build_customer_features


CUSTOMER_FEATURES = [
    "recency", "frequency", "monetary", "avg_order_value",
    "unique_products", "units", "tenure", "purchase_rate",
]
FORECAST_LAGS = (1, 2, 4, 13)


@dataclass
class ClassificationBundle:
    model: Pipeline
    target: str
    horizon_days: int
    metrics: dict[str, Any]
    drivers: list[dict[str, Any]]


@dataclass
class ForecastBundle:
    model: TransformedTargetRegressor
    history: pd.Series
    metrics: dict[str, Any]
    residual_band: float
    drivers: list[dict[str, Any]]
    selected_method: str


def _validate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    required = {"CustomerID", "InvoiceNo", "InvoiceDate", "Revenue", "StockCode", "Quantity"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Predictive models require columns: {sorted(missing)}")
    tx = transactions.copy()
    tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"], errors="coerce")
    for column in ("Revenue", "Quantity"):
        tx[column] = pd.to_numeric(tx[column], errors="coerce")
    tx[["Revenue", "Quantity"]] = tx[["Revenue", "Quantity"]].replace(
        [np.inf, -np.inf], np.nan
    )
    tx = tx.dropna(subset=["CustomerID", "InvoiceDate", "Revenue"])
    tx = tx[tx.Revenue >= 0].sort_values("InvoiceDate")
    if tx.InvoiceDate.nunique() < 30:
        raise ValueError("Predictive modeling requires transactions across at least 30 distinct dates.")
    return tx


def build_temporal_customer_dataset(
    transactions: pd.DataFrame,
    horizon_days: int,
    target: Literal["repeat_purchase", "churn"],
    minimum_history_days: int = 90,
    snapshot_step_days: int = 30,
) -> pd.DataFrame:
    """Build customer-snapshot rows without using future data in features."""
    tx = _validate_transactions(transactions)
    first = tx.InvoiceDate.min().normalize()
    last = tx.InvoiceDate.max().normalize()
    first_snapshot = first + pd.Timedelta(days=minimum_history_days)
    last_snapshot = last - pd.Timedelta(days=horizon_days)
    if first_snapshot > last_snapshot:
        raise ValueError(
            f"Not enough history for a {horizon_days}-day outcome window. "
            f"Need more than {minimum_history_days + horizon_days} days of data."
        )
    snapshots = pd.date_range(first_snapshot, last_snapshot, freq=f"{snapshot_step_days}D")
    rows = []
    for snapshot in snapshots:
        history = tx[tx.InvoiceDate <= snapshot]
        future = tx[(tx.InvoiceDate > snapshot)
                    & (tx.InvoiceDate <= snapshot + pd.Timedelta(days=horizon_days))]
        features = build_customer_features(history, snapshot_date=snapshot)
        future_buyers = set(future.CustomerID.astype(str))
        purchased = features.CustomerID.astype(str).isin(future_buyers).astype(int)
        features["label"] = purchased if target == "repeat_purchase" else 1 - purchased
        features["snapshot_date"] = snapshot
        rows.append(features[["CustomerID", "snapshot_date", *CUSTOMER_FEATURES, "label"]])
    dataset = pd.concat(rows, ignore_index=True)
    dataset[CUSTOMER_FEATURES] = dataset[CUSTOMER_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    if dataset.label.nunique() < 2:
        raise ValueError(f"Historical snapshots do not contain both classes for {target}.")
    return dataset


def _classification_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=42)),
    ])


def train_customer_classifier(
    transactions: pd.DataFrame,
    target: Literal["repeat_purchase", "churn"],
    horizon_days: int,
) -> ClassificationBundle:
    dataset = build_temporal_customer_dataset(transactions, horizon_days, target)
    snapshots = sorted(dataset.snapshot_date.unique())
    validation_count = max(1, min(2, len(snapshots) // 3))
    validation_start = snapshots[-validation_count]
    train = dataset[dataset.snapshot_date < validation_start]
    valid = dataset[dataset.snapshot_date >= validation_start]
    if train.empty or train.label.nunique() < 2 or valid.label.nunique() < 2:
        raise ValueError("Temporal split needs both outcome classes in training and validation data.")

    validation_model = _classification_pipeline().fit(train[CUSTOMER_FEATURES], train.label)
    probability = validation_model.predict_proba(valid[CUSTOMER_FEATURES])[:, 1]
    metrics = {
        "roc_auc": round(float(roc_auc_score(valid.label, probability)), 4),
        "average_precision": round(float(average_precision_score(valid.label, probability)), 4),
        "brier_score": round(float(brier_score_loss(valid.label, probability)), 4),
        "validation_rows": int(len(valid)),
        "validation_positive_rate": round(float(valid.label.mean()), 4),
        "temporal_split": f"train before {pd.Timestamp(validation_start).date()}; validate on later snapshots",
        "training_snapshots": int(len(snapshots) - validation_count),
        "validation_snapshots": int(validation_count),
    }
    final_model = _classification_pipeline().fit(dataset[CUSTOMER_FEATURES], dataset.label)
    coefficients = final_model.named_steps["model"].coef_[0]
    drivers = sorted([
        {
            "feature": feature,
            "direction": "increases probability" if coefficient > 0 else "decreases probability",
            "importance": round(abs(float(coefficient)), 4),
        }
        for feature, coefficient in zip(CUSTOMER_FEATURES, coefficients)
    ], key=lambda item: item["importance"], reverse=True)
    return ClassificationBundle(final_model, target, horizon_days, metrics, drivers)


def score_customer_classifier(bundle: ClassificationBundle, transactions: pd.DataFrame,
                              current_customers: pd.DataFrame) -> pd.DataFrame:
    tx = _validate_transactions(transactions)
    as_of = tx.InvoiceDate.max().normalize() + pd.Timedelta(days=1)
    features = build_customer_features(tx, snapshot_date=as_of)
    features[CUSTOMER_FEATURES] = features[CUSTOMER_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    features["probability"] = bundle.model.predict_proba(features[CUSTOMER_FEATURES])[:, 1]
    segment_columns = [column for column in ("CustomerID", "cluster_persona", "rule_segment")
                       if column in current_customers.columns]
    return features.merge(current_customers[segment_columns], on="CustomerID", how="left")


def aggregate_customer_predictions(scored: pd.DataFrame, threshold: float,
                                   probability_name: str) -> dict:
    flagged = scored.probability >= threshold
    segment_scores = scored.groupby("cluster_persona", dropna=False).agg(
        customers=("CustomerID", "nunique"),
        mean_probability=("probability", "mean"),
        flagged_customers=("probability", lambda values: int((values >= threshold).sum())),
    ).reset_index().rename(columns={"cluster_persona": "segment"})
    records = [{
        "segment": str(row.segment),
        "customers": int(row.customers),
        "mean_probability": round(float(row.mean_probability), 4),
        "flagged_customers": int(row.flagged_customers),
    } for row in segment_scores.sort_values("mean_probability", ascending=False).itertuples()]
    return {
        "scored_customers": int(scored.CustomerID.nunique()),
        "flagged_customers": int(flagged.sum()),
        "flagged_share": round(float(flagged.mean()), 4),
        "mean_probability": round(float(scored.probability.mean()), 4),
        "threshold": round(float(threshold), 4),
        "probability_name": probability_name,
        "segment_scores": records,
    }


def _weekly_revenue(transactions: pd.DataFrame) -> tuple[pd.Series, bool]:
    tx = _validate_transactions(transactions)
    weekly = (
        tx.set_index("InvoiceDate").Revenue.resample("W-SUN").sum().astype(float)
    )
    weekly = weekly.asfreq("W-SUN", fill_value=0.0)
    incomplete_week_dropped = bool(tx.InvoiceDate.max().normalize() < weekly.index[-1].normalize())
    if incomplete_week_dropped and len(weekly) > 1:
        weekly = weekly.iloc[:-1]
    return weekly, incomplete_week_dropped


def _forecast_row(history: list[float], period: pd.Timestamp, trend: int) -> dict:
    if len(history) < max(FORECAST_LAGS):
        raise ValueError("Revenue forecast requires at least 13 weeks of history.")
    week = int(period.isocalendar().week)
    row = {f"lag_{lag}": history[-lag] for lag in FORECAST_LAGS}
    row.update({
        "rolling_mean_4": float(np.mean(history[-4:])),
        "rolling_mean_13": float(np.mean(history[-13:])),
        "trend": trend,
        "week_sin": float(np.sin(2 * np.pi * week / 52.18)),
        "week_cos": float(np.cos(2 * np.pi * week / 52.18)),
    })
    return row


def _forecast_supervised(weekly: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    rows, targets, dates = [], [], []
    values = weekly.tolist()
    for index in range(max(FORECAST_LAGS), len(weekly)):
        rows.append(_forecast_row(values[:index], weekly.index[index], index))
        targets.append(values[index])
        dates.append(weekly.index[index])
    return pd.DataFrame(rows, index=dates), pd.Series(targets, index=dates, name="revenue")


def _bounded_expm1(values):
    return np.expm1(np.clip(values, -50, 30))


def _forecast_estimator() -> TransformedTargetRegressor:
    regressor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ])
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        # A bounded inverse prevents numerical overflow before operational
        # history-based clipping is applied to recursive predictions.
        inverse_func=_bounded_expm1,
        check_inverse=False,
    )


def _safe_forecast_values(values, history: list[float]) -> np.ndarray:
    """Replace non-finite forecasts and cap explosive recursive extrapolation."""
    recent = np.asarray(history[-13:], dtype=float)
    recent = recent[np.isfinite(recent) & (recent >= 0)]
    fallback = float(recent[-1]) if len(recent) else 0.0
    observed_max = float(np.max(recent)) if len(recent) else fallback
    operational_cap = max(observed_max * 2, fallback * 2, 1.0)
    safe = np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=fallback,
        posinf=operational_cap,
        neginf=0.0,
    )
    return np.clip(safe, 0.0, operational_cap)


def train_revenue_forecaster(transactions: pd.DataFrame) -> ForecastBundle:
    weekly, incomplete_week_dropped = _weekly_revenue(transactions)
    features, target = _forecast_supervised(weekly)
    if len(features) < 20:
        raise ValueError("Revenue forecasting requires at least 33 weeks of transaction history.")
    validation_weeks = max(4, min(8, len(features) // 4))
    train_x, valid_x = features.iloc[:-validation_weeks], features.iloc[-validation_weeks:]
    train_y, valid_y = target.iloc[:-validation_weeks], target.iloc[-validation_weeks:]
    validation_model = _forecast_estimator().fit(train_x, train_y)
    ridge_prediction = _safe_forecast_values(
        validation_model.predict(valid_x), train_y.tolist()
    )
    naive_prediction = valid_x["lag_1"].to_numpy(dtype=float)
    rolling_prediction = valid_x["rolling_mean_4"].to_numpy(dtype=float)
    candidate_predictions = {
        "ridge": ridge_prediction,
        "naive_last": naive_prediction,
        "rolling_mean_4": rolling_prediction,
    }
    candidate_mae = {
        name: float(mean_absolute_error(valid_y, values))
        for name, values in candidate_predictions.items()
    }
    selected_method = min(candidate_mae, key=candidate_mae.get)
    prediction = candidate_predictions[selected_method]
    mae = candidate_mae[selected_method]
    baseline_mae = candidate_mae["naive_last"]
    validation_mean = float(valid_y.mean())
    normalized_mae = mae / validation_mean if validation_mean > 0 else float("inf")
    reliability = "high" if normalized_mae <= .20 else "medium" if normalized_mae <= .40 else "low"
    nonzero = valid_y.to_numpy() != 0
    mape = float(np.mean(np.abs((valid_y.to_numpy()[nonzero] - prediction[nonzero])
                                / valid_y.to_numpy()[nonzero]))) if nonzero.any() else None
    residuals = np.abs(valid_y.to_numpy() - prediction)
    metrics = {
        "mae": round(mae, 2),
        "rmse": round(float(mean_squared_error(valid_y, prediction) ** .5), 2),
        "mape": round(mape, 4) if mape is not None else None,
        "baseline_mae": round(baseline_mae, 2),
        "ridge_mae": round(candidate_mae["ridge"], 2),
        "rolling_mean_4_mae": round(candidate_mae["rolling_mean_4"], 2),
        "selected_method": selected_method,
        "normalized_mae": round(float(normalized_mae), 4),
        "reliability": reliability,
        "model_beats_baseline": bool(mae < baseline_mae),
        "validation_weeks": int(validation_weeks),
        "temporal_split": f"last {validation_weeks} observed weeks held out",
        "incomplete_week_dropped": incomplete_week_dropped,
    }
    final_model = _forecast_estimator().fit(features, target)
    ridge = final_model.regressor_.named_steps["model"]
    drivers = sorted([
        {
            "feature": feature,
            "direction": "positive relationship" if coefficient > 0 else "negative relationship",
            "importance": round(abs(float(coefficient)), 4),
        }
        for feature, coefficient in zip(features.columns, ridge.coef_)
    ], key=lambda item: item["importance"], reverse=True)
    return ForecastBundle(
        model=final_model,
        history=weekly,
        metrics=metrics,
        residual_band=round(float(np.quantile(residuals, .90)), 2),
        drivers=drivers,
        selected_method=selected_method,
    )


def forecast_revenue(bundle: ForecastBundle, weeks: int) -> list[dict]:
    history = bundle.history.tolist()
    safety_reference = history.copy()
    current_date = bundle.history.index[-1]
    output = []
    for step in range(1, weeks + 1):
        period = current_date + pd.Timedelta(weeks=step)
        if bundle.selected_method == "naive_last":
            predicted = float(history[-1])
        elif bundle.selected_method == "rolling_mean_4":
            predicted = float(np.mean(history[-4:]))
        else:
            row = pd.DataFrame([_forecast_row(history, period, len(history))])
            predicted = float(_safe_forecast_values(
                bundle.model.predict(row), safety_reference
            )[0])
        history.append(predicted)
        output.append({
            "week_ending": str(period.date()),
            "predicted_revenue_gbp": round(predicted, 2),
            "lower_bound_gbp": round(max(0, predicted - bundle.residual_band), 2),
            "upper_bound_gbp": round(predicted + bundle.residual_band, 2),
        })
    return output


class PredictiveEngine:
    """Caches fitted models for one immutable transaction snapshot."""

    def __init__(self, transactions: pd.DataFrame):
        self.transactions = transactions
        self._classifiers: dict[str, ClassificationBundle] = {}
        self._forecast: ForecastBundle | None = None

    def classifier(self, model_name: Literal["churn", "repeat_purchase"]) -> ClassificationBundle:
        if model_name not in self._classifiers:
            target, horizon = ("churn", 90) if model_name == "churn" else ("repeat_purchase", 30)
            self._classifiers[model_name] = train_customer_classifier(
                self.transactions, target=target, horizon_days=horizon
            )
        return self._classifiers[model_name]

    def revenue_forecaster(self) -> ForecastBundle:
        if self._forecast is None:
            self._forecast = train_revenue_forecaster(self.transactions)
        return self._forecast
