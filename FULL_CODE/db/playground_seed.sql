-- Synthetic sample rows for the Playground (no real people; test card numbers).
INSERT INTO test_customers (full_name, mobile, email, card_no, aadhaar, pan, balance, notes) VALUES
 ('Kiran Rao', '+91-9000011111', 'kiran.rao@mail.example', '4111 1111 1111 1111', '2345 6789 0123', 'ABCPK1234F', 125000.75, 'Prefers calls after 6 pm'),
 ('Fatima Shaikh', '9999922222', 'fatima.s@inbox.example', '5555-5555-5555-4444', 'XXXX XXXX 4321', 'BCDPF2345G', 9800.00, 'Changed mobile from 9000033333 last month'),
 ('Joseph Dsouza', '+91 99999 44444', NULL, '4012888888881881', '3456 7890 1234', 'CDEPJ3456H', 2450000.00, NULL);
INSERT INTO test_tickets (customer_name, channel, ticket_text) VALUES
 ('Kiran Rao', 'App Chat', 'Mera card 4111 1111 1111 1111 kho gaya, turant block karo. Mera number 9000011111 hai.'),
 ('Fatima Shaikh', 'Email', 'Please update my PAN BCDPF2345G on account 12345678901234.\nRegards,\nFatima Shaikh'),
 ('Joseph Dsouza', 'Phone', 'Sent Rs 5000 to joseph.d@saffronupi but it failed. Call me on +91 99999 44444.');
