# Qdrant

Qdrant is an open-source vector database with native hybrid search support,
payload filters, and a Python client. It can run as a Docker container on a
laptop, or entirely in-memory for tests via `QdrantClient(":memory:")`. Point
IDs are either 64-bit unsigned integers or UUIDs; storing a compact integer
derived from a content hash keeps the collection cheap to inspect. Collection
metadata like the embedding model name should be recorded so a collection
built with one encoder is never queried with another.
