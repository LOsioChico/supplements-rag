-- terms with huge postings (stopwords live in ~17k docs) exceed D1's
-- statement size — postings shard per term instead of one row per term;
-- df duplicates per shard row so a single query still answers everything
DROP TABLE IF EXISTS bm25_terms;

CREATE TABLE IF NOT EXISTS bm25_postings (
    term TEXT NOT NULL,
    shard INTEGER NOT NULL,
    df INTEGER NOT NULL,
    postings TEXT NOT NULL,
    PRIMARY KEY (term, shard)
);
