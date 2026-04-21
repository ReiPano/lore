"""LangChain retriever backed by Lore's HTTP API.

Drop-in :class:`BaseRetriever` that calls ``POST /search`` on a running
``lore serve-api``. Each hit becomes a :class:`Document` whose ``page_content``
is the chunk text and whose ``metadata`` carries ``chunk_id``, ``source_path``,
``score``, and the per-component score breakdown.

Install the optional dep: ``pip install 'lore-memory[langchain]'``.

Example
-------

.. code-block:: python

    from hybrid_search.integrations.langchain import LoreRetriever

    retriever = LoreRetriever(base_url="http://127.0.0.1:8765", k=5)
    docs = retriever.invoke("auth middleware")
"""

from __future__ import annotations

from typing import Any

try:
    import httpx
except ImportError as exc:  # pragma: no cover - httpx is a core dep already
    raise ImportError("httpx is required by LoreRetriever") from exc

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as exc:  # pragma: no cover - guarded by optional extra
    raise ImportError(
        "LoreRetriever needs the 'langchain' optional dependency. "
        "Install with: pip install 'lore-memory[langchain]'"
    ) from exc


DEFAULT_BASE_URL = "http://127.0.0.1:8765"


class LoreRetriever(BaseRetriever):
    """LangChain retriever that hits Lore's ``POST /search`` endpoint."""

    base_url: str = DEFAULT_BASE_URL
    k: int = 10
    rerank: bool | None = None
    project: str | None = None
    auth_token: str | None = None
    timeout: float = 30.0

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def _payload(self, query: str) -> dict[str, Any]:
        body: dict[str, Any] = {"query": query, "k": self.k}
        if self.rerank is not None:
            body["rerank"] = self.rerank
        if self.project:
            body["project"] = self.project
        return body

    def _to_document(self, hit: dict[str, Any]) -> Document:
        return Document(
            page_content=hit.get("text", ""),
            metadata={
                "chunk_id": hit.get("chunk_id"),
                "source_path": hit.get("source_path"),
                "score": hit.get("score"),
                "scores_breakdown": hit.get("scores_breakdown") or {},
            },
        )

    def _get_relevant_documents(  # type: ignore[override]
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url.rstrip('/')}/search",
                json=self._payload(query),
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
        return [self._to_document(r) for r in data.get("results", [])]

    async def _aget_relevant_documents(  # type: ignore[override]
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url.rstrip('/')}/search",
                json=self._payload(query),
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
        return [self._to_document(r) for r in data.get("results", [])]
