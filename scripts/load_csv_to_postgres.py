"""Validate a retail CSV and load its clean rows into PostgreSQL."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from src.data import load_transactions_with_report
from src.database import write_transactions_to_database


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Transaction CSV path")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--replace", action="store_true",
                        help="Truncate existing transactions before loading")
    args = parser.parse_args()
    if not args.database_url:
        parser.error("Set DATABASE_URL or pass --database-url")

    clean, report = load_transactions_with_report(args.csv)
    rows = write_transactions_to_database(
        clean,
        args.database_url,
        replace=args.replace,
        report=report,
        source_name=args.csv.name,
    )
    print(f"Loaded {rows:,} clean rows from {report.raw_rows:,} raw rows.")
    print(f"Customers: {report.customers:,}; orders: {report.orders:,}")


if __name__ == "__main__":
    main()
