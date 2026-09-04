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
CREATE INDEX IF NOT EXISTS idx_transactions_customer_date ON transactions(customer_id, invoice_date);
CREATE INDEX IF NOT EXISTS idx_transactions_product ON transactions(stock_code);

CREATE TABLE IF NOT EXISTS data_import_audit (
    id BIGSERIAL PRIMARY KEY,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_name TEXT NOT NULL,
    report JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_data_import_audit_imported_at
    ON data_import_audit(imported_at DESC, id DESC);
