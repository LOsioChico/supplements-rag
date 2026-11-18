-- chunks keyed by chunk_id (the postings + fusion ids resolve through this),
-- one term-indexed postings row per vocabulary term (BM25 at query time reads
-- only the question's terms, never the whole index), corpus-wide stats in
-- bm25_meta, and the visitor-ask trace log.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    body TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bm25_terms (
    term TEXT PRIMARY KEY,
    df INTEGER NOT NULL,
    postings TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bm25_meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    q TEXT NOT NULL,
    events TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS runs_ts ON runs(ts);
