CREATE TABLE transactions (
    invoice_no TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    description TEXT,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    invoice_date TIMESTAMP NOT NULL,
    unit_price NUMERIC(12,2) NOT NULL CHECK (unit_price > 0),
    customer_id TEXT NOT NULL,
    country TEXT,
    PRIMARY KEY (invoice_no, stock_code)
);
CREATE INDEX idx_transactions_customer_date ON transactions(customer_id, invoice_date);

