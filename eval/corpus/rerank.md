# Cross-encoder reranking

A cross-encoder reranker scores a query-document pair jointly, rather than
independently like a bi-encoder. Models such as `bge-reranker-base` or
`ms-marco-MiniLM-L-6-v2` take 50-200 milliseconds per pair on CPU, so
rerankers are best used on the top 20-50 candidates after an initial recall
stage. On corpora with many near-duplicates or tricky lexical overlap the
quality gain can be transformative; on others the cost barely pays for
itself. Eval scripts should make this explicit.
