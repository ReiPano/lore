# Reciprocal rank fusion

Reciprocal rank fusion combines multiple ranked lists without needing their
scores to be calibrated. For each document `d` and each list `i`, it adds
`weight_i / (k + rank_i(d))` to the fused score. The constant `k` is
typically 60. A document that ranks highly in both lists wins; a document
that only appears in one still surfaces, just with a lower fused score. RRF
is a good default when BM25 and cosine similarity produce scores on
incomparable scales.
