# Dense vector retrieval

Dense vector retrieval encodes a query and each document into a shared
embedding space, then ranks candidates by cosine similarity. Small sentence
transformers like `bge-small-en-v1.5` produce 384-dimensional vectors and
are cheap enough to run on a laptop without a GPU. Vectors shine for
paraphrased queries where the surface terms do not overlap but the meaning
matches. They are weaker at identifier lookup and struggle on rare terms
that were never well represented in the training corpus.
