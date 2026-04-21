# Chunking strategy

Chunking splits a document into retrieval-sized units before indexing. Good
chunkers respect structure: Markdown chunkers split on headings and keep
fenced code blocks atomic, while code chunkers break on function and class
boundaries so a single chunk remains a meaningful unit of meaning. Chunk IDs
should be deterministic hashes of the source path, start offset, and text so
reindexing a modified file can diff cleanly against the existing index and
only reembed what actually changed.
