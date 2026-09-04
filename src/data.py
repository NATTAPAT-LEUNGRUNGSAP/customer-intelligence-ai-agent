from dataclasses import dataclass, asdict, fields
from pathlib import Path
import pandas as pd

REQUIRED = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}


@dataclass
class DataQualityReport:
    raw_rows: int
    duplicate_rows_removed: int
    missing_customer_id_removed: int
    invalid_invoice_date_removed: int
    missing_invoice_or_product_removed: int
    invalid_numeric_removed: int
    invalid_required_fields_removed: int
    cancelled_rows_removed: int
    nonpositive_rows_removed: int
    clean_rows: int
    customers: int
    orders: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "DataQualityReport":
        """Restore an audit record while tolerating fields added in later versions."""
        return cls(**{field.name: int(values.get(field.name, 0)) for field in fields(cls)})


def load_transactions(path: str | Path) -> pd.DataFrame:
    return load_transactions_with_report(path)[0]


def load_transactions_with_report(path: str | Path) -> tuple[pd.DataFrame, DataQualityReport]:
    return clean_transactions_dataframe(pd.read_csv(path, encoding_errors="ignore"))


def clean_transactions_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """Validate and clean a raw transaction dataframe from any source."""
    df = df.copy()
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    raw_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    # Kaggle mirrors of Online Retail contain multiple valid date layouts.
    # `format="mixed"` prevents pandas from locking onto only the first layout.
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed", errors="coerce")
    for col in ["Quantity", "UnitPrice", "CustomerID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    missing_customer_mask = df["CustomerID"].isna()
    missing_customer = int(missing_customer_mask.sum())
    df = df.loc[~missing_customer_mask].copy()
    invalid_date_mask = df["InvoiceDate"].isna()
    invalid_date = int(invalid_date_mask.sum())
    df = df.loc[~invalid_date_mask].copy()
    missing_key_mask = df[["InvoiceNo", "StockCode"]].isna().any(axis=1)
    missing_key = int(missing_key_mask.sum())
    df = df.loc[~missing_key_mask].copy()
    invalid_numeric_mask = df[["Quantity", "UnitPrice"]].isna().any(axis=1)
    invalid_numeric = int(invalid_numeric_mask.sum())
    df = df.loc[~invalid_numeric_mask].copy()
    invalid_required = missing_customer + invalid_date + missing_key + invalid_numeric
    cancelled_mask = df["InvoiceNo"].astype(str).str.upper().str.startswith("C")
    cancelled = int(cancelled_mask.sum())
    df = df.loc[~cancelled_mask].copy()
    nonpositive_mask = (df["Quantity"] <= 0) | (df["UnitPrice"] <= 0)
    nonpositive = int(nonpositive_mask.sum())
    df = df.loc[~nonpositive_mask].copy()
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    report = DataQualityReport(
        raw_rows=raw_rows,
        duplicate_rows_removed=duplicate_rows,
        missing_customer_id_removed=missing_customer,
        invalid_invoice_date_removed=invalid_date,
        missing_invoice_or_product_removed=missing_key,
        invalid_numeric_removed=invalid_numeric,
        invalid_required_fields_removed=invalid_required,
        cancelled_rows_removed=cancelled,
        nonpositive_rows_removed=nonpositive,
        clean_rows=len(df),
        customers=df["CustomerID"].nunique(),
        orders=df["InvoiceNo"].nunique(),
    )
    return df, report


def resolve_default_data(project_root: Path) -> Path:
    real = project_root / "data" / "online_retail.csv"
    sample = project_root / "data" / "sample_transactions.csv"
    return real if real.exists() else sample
