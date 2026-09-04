"""Optional PostgreSQL adapter for the transaction pipeline."""
from __future__ import annotations

import json
import pandas as pd

from src.data import DataQualityReport, clean_transactions_dataframe

TRANSACTION_QUERY = """
SELECT
    invoice_no AS "InvoiceNo",
    stock_code AS "StockCode",
    description AS "Description",
    quantity AS "Quantity",
    invoice_date AS "InvoiceDate",
    unit_price AS "UnitPrice",
    customer_id AS "CustomerID",
    country AS "Country"
FROM transactions
"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS transactions (
    id BIGSERIAL PRIMARY KEY,
    invoice_no TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    description TEXT,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    invoice_date TIMESTAMP NOT NULL,
    unit_price NUMERIC(14,4) NOT NULL CHECK (unit_price > 0),
    customer_id TEXT NOT NULL,
    country TEXT
);
CREATE INDEX IF NOT EXISTS idx_transactions_customer_date
    ON transactions(customer_id, invoice_date);
CREATE INDEX IF NOT EXISTS idx_transactions_product
    ON transactions(stock_code);
CREATE TABLE IF NOT EXISTS data_import_audit (
    id BIGSERIAL PRIMARY KEY,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_name TEXT NOT NULL,
    report JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_data_import_audit_imported_at
    ON data_import_audit(imported_at DESC, id DESC);
"""

DATABASE_INSERT_CHUNK_SIZE = 1_000


def _sqlalchemy():
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to enable PostgreSQL support.") from exc
    return create_engine, text


def load_transactions_from_database(database_url: str) -> tuple[pd.DataFrame, DataQualityReport]:
    """Read clean transactions and restore the original import audit when available."""
    if not database_url.strip():
        raise ValueError("DATABASE_URL is required for PostgreSQL mode.")
    create_engine, text = _sqlalchemy()
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            raw = pd.read_sql(text(TRANSACTION_QUERY), connection)
            audit_exists = bool(connection.execute(
                text("SELECT to_regclass('public.data_import_audit') IS NOT NULL")
            ).scalar_one())
            stored_audit = None
            if audit_exists:
                stored_audit = connection.execute(text(
                    "SELECT report FROM data_import_audit ORDER BY imported_at DESC, id DESC LIMIT 1"
                )).scalar_one_or_none()
    finally:
        engine.dispose()
    if raw.empty:
        raise ValueError("The transactions table is empty. Load a CSV before using PostgreSQL mode.")
    clean, database_report = clean_transactions_dataframe(raw)
    if stored_audit is None:
        return clean, database_report
    if isinstance(stored_audit, str):
        stored_audit = json.loads(stored_audit)
    imported_report = DataQualityReport.from_dict(stored_audit)
    # Never show stale audit metadata if the transaction table was modified later.
    return (clean, imported_report) if imported_report.clean_rows == len(clean) else (clean, database_report)


def write_transactions_to_database(clean: pd.DataFrame, database_url: str,
                                   replace: bool = False,
                                   report: DataQualityReport | None = None,
                                   source_name: str = "unknown") -> int:
    """Load validated transactions, refusing accidental duplicates by default."""
    create_engine, text = _sqlalchemy()
    engine = create_engine(database_url, pool_pre_ping=True)
    output = clean.rename(columns={
        "InvoiceNo": "invoice_no", "StockCode": "stock_code",
        "Description": "description", "Quantity": "quantity",
        "InvoiceDate": "invoice_date", "UnitPrice": "unit_price",
        "CustomerID": "customer_id", "Country": "country",
    })
    columns = ["invoice_no", "stock_code", "description", "quantity",
               "invoice_date", "unit_price", "customer_id", "country"]
    for column in columns:
        if column not in output:
            output[column] = None

    try:
        with engine.begin() as connection:
            for statement in SCHEMA_SQL.split(";"):
                if statement.strip():
                    connection.execute(text(statement))
            existing = int(connection.execute(text("SELECT COUNT(*) FROM transactions")).scalar_one())
            if existing and not replace:
                raise ValueError(
                    f"transactions already contains {existing:,} rows. Use --replace to reload it intentionally."
                )
            if replace:
                connection.execute(text("TRUNCATE TABLE transactions RESTART IDENTITY"))
            output[columns].to_sql(
                "transactions",
                connection,
                if_exists="append",
                index=False,
                chunksize=DATABASE_INSERT_CHUNK_SIZE,
            )
            if report is not None:
                connection.execute(text(
                    "INSERT INTO data_import_audit (source_name, report) "
                    "VALUES (:source_name, CAST(:report AS JSONB))"
                ), {
                    "source_name": source_name,
                    "report": json.dumps(report.to_dict()),
                })
        return len(output)
    finally:
        engine.dispose()
