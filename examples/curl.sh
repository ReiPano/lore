#!/usr/bin/env bash
# Example requests against the local FastAPI server.
# Start it first: make up && .venv/bin/hybrid-search serve-api

set -euo pipefail
BASE="http://127.0.0.1:8765"

echo '--- /health'
curl -sS "$BASE/health" | jq .

echo '--- POST /index'
curl -sS -X POST "$BASE/index" \
  -H 'Content-Type: application/json' \
  -d '{"paths":["./eval/corpus"]}' | jq .

echo '--- POST /search'
curl -sS -X POST "$BASE/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"reciprocal rank fusion","k":5,"rerank":false}' | jq .
