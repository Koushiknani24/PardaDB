-- Playground database (writable, for your own test tables). Created on first start.
-- The bank databases come from the problem pack and are only ever opened read-only.
CREATE TABLE IF NOT EXISTS test_customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT, mobile TEXT, email TEXT, card_no TEXT,
    aadhaar TEXT, pan TEXT, balance REAL, notes TEXT);

CREATE TABLE IF NOT EXISTS test_tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, customer_name TEXT, channel TEXT, ticket_text TEXT);
