ALTER TABLE runs ADD COLUMN source TEXT;
-- prod already has rid; applied files never re-run, so this only affects fresh local databases
ALTER TABLE runs ADD COLUMN rid TEXT;
