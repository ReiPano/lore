# Hybrid Search Eval

- corpus: `/Users/reipano/Personal/better-mem/eval/corpus`
- queries: `/Users/reipano/Personal/better-mem/eval/queries.jsonl`
- embedder: `fake-hash-v1`
- k: 10

| config | hit@k | mrr | ndcg@k | total ms |
|---|---|---|---|---|
| bm25 | 0.200 | 0.200 | 0.200 | 0.7 |
| vector | 1.000 | 0.292 | 0.594 | 2.7 |
| hybrid | 1.000 | 0.438 | 0.705 | 5.2 |

