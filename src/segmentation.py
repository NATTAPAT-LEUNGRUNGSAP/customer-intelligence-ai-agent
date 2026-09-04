from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler

MODEL_FEATURES = ["recency", "frequency", "monetary", "avg_order_value", "unique_products"]


@dataclass
class SegmentationResult:
    customers: pd.DataFrame
    score_by_k: dict[int, float]
    selected_k: int
    automatic_k: int
    diagnostics: pd.DataFrame
    caps: dict[str, float]


def _rule_label(row, q):
    if row.frequency == 1 and row.recency <= q["recent"]:
        return "New Customers"
    if row.recency <= q["recent"] and row.frequency >= q["frequent"] and row.monetary >= q["high_value"]:
        return "Champions"
    if row.recency >= q["stale"] and row.monetary >= q["high_value"]:
        return "At Risk High Value"
    if row.recency >= q["stale"]:
        return "Hibernating"
    return "Potential Loyalists"


def segment_customers(features: pd.DataFrame, k_min=3, k_max=6, random_state=42,
                      selected_k: int | None = None) -> SegmentationResult:
    if len(features) < 4:
        raise ValueError("At least 4 customers are required for clustering")
    out = features.copy()
    q = {
        "recent": out.recency.quantile(.35), "stale": out.recency.quantile(.70),
        "frequent": out.frequency.quantile(.70), "high_value": out.monetary.quantile(.70),
    }
    out["rule_segment"] = out.apply(_rule_label, axis=1, q=q)
    # Cap only the model input, not the customer facts shown in reports.
    # This prevents a handful of wholesale-sized buyers from defining every cluster.
    caps = out[MODEL_FEATURES].quantile(.99).to_dict()
    model_input = out[MODEL_FEATURES].clip(lower=0).astype(float)
    for col, cap in caps.items():
        model_input[col] = model_input[col].clip(upper=cap)
    scaled = RobustScaler().fit_transform(np.log1p(model_input))
    scores, models, diagnostics = {}, {}, []
    upper = min(k_max, len(out) - 1)
    for k in range(k_min, upper + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit(scaled)
        if len(set(model.labels_)) > 1:
            silhouette = float(silhouette_score(scaled, model.labels_))
            db = float(davies_bouldin_score(scaled, model.labels_))
            second = KMeans(n_clusters=k, n_init=20, random_state=random_state + 17).fit(scaled)
            stability = float(adjusted_rand_score(model.labels_, second.labels_))
            counts = np.bincount(model.labels_)
            min_share = float(counts.min() / len(out))
            # Reward separation and repeatability; penalize unusably tiny audiences.
            utility = silhouette + .08 * stability - max(0, .03 - min_share) * 3
            scores[k] = utility
            models[k] = model
            diagnostics.append({"k": k, "silhouette": silhouette, "davies_bouldin": db,
                                "stability": stability, "smallest_cluster_share": min_share,
                                "selection_score": utility})
    automatic = max(scores, key=scores.get)
    if selected_k is not None and selected_k not in models:
        raise ValueError(f"selected_k must be one of {sorted(models)}")
    selected = selected_k if selected_k is not None else automatic
    out["cluster"] = models[selected].labels_
    profiles = out.groupby("cluster")[MODEL_FEATURES].mean()
    ranks = profiles.rank(pct=True)
    names = {}
    remaining = set(profiles.index)

    def assign(cluster, label):
        if cluster in remaining:
            names[cluster] = f"{label} (C{cluster})"
            remaining.remove(cluster)

    assign((ranks.monetary + ranks.frequency - ranks.recency).idxmax(), "Loyal High Value")
    if remaining:
        premium_score = ranks.loc[list(remaining), "avg_order_value"] - .25 * ranks.loc[list(remaining), "frequency"]
        assign(premium_score.idxmax(), "Occasional Premium")
    if remaining:
        dormant_score = ranks.loc[list(remaining), "recency"] - ranks.loc[list(remaining), "monetary"]
        assign(dormant_score.idxmax(), "Hibernating Low Value")
    if remaining:
        assign(profiles.loc[list(remaining), "recency"].idxmin(), "Recent / New Buyers")
    if remaining:
        lapsed_score = ranks.loc[list(remaining), "recency"] + ranks.loc[list(remaining), "monetary"]
        assign(lapsed_score.idxmax(), "Lapsed Valuable")
    for cluster in sorted(remaining):
        assign(cluster, "Value Regulars")
    out["cluster_persona"] = out["cluster"].map(names)
    return SegmentationResult(out, scores, selected, automatic, pd.DataFrame(diagnostics), caps)
