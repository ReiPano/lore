"""FastAPI HTTP endpoints for hybrid search.

Exposes the same functionality as the MCP server so non-MCP agents (plain
curl, LangChain retrievers, etc.) can drive the index over HTTP. The
``create_app`` factory takes prebuilt ``HybridSearch`` and ``Indexer``
instances, so wiring and lifecycle are the caller's concern — nothing global
is created at import time.

Index jobs run in the background via ``asyncio.create_task`` +
``asyncio.to_thread`` so callers get a ``job_id`` back immediately and poll
``GET /jobs/{id}`` for progress.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .indexer import IndexStats, Indexer
from .search import HybridSearch

log = logging.getLogger(__name__)


# ---- request/response models ------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=500)
    rerank: bool | None = None


class SearchResultModel(BaseModel):
    chunk_id: str
    text: str
    source_path: str
    score: float
    scores_breakdown: dict[str, float] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResultModel]


class IndexRequest(BaseModel):
    paths: list[str] = Field(min_length=1)


class IndexResponse(BaseModel):
    job_id: str
    status: str


class JobResponse(BaseModel):
    id: str
    status: str
    paths: list[str]
    stats: dict[str, Any] | None = None
    error: str | None = None
    created_at: float
    finished_at: float | None = None


class DeleteRequest(BaseModel):
    paths: list[str] | None = None
    chunk_ids: list[str] | None = None


class DeleteResponse(BaseModel):
    chunks_removed: int
    files_removed: int


class HealthResponse(BaseModel):
    chunk_count: int
    sources_count: int
    embedding_model: str
    collection_name: str


# ---- job registry -----------------------------------------------------


@dataclass
class JobState:
    id: str
    paths: list[str]
    status: str
    created_at: float
    finished_at: float | None = None
    stats: IndexStats | None = None
    error: str | None = None


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = asyncio.Lock()

    async def create(self, paths: list[str]) -> JobState:
        async with self._lock:
            job = JobState(
                id=uuid.uuid4().hex,
                paths=paths,
                status="pending",
                created_at=time.time(),
            )
            self._jobs[job.id] = job
            return job

    async def get(self, job_id: str) -> JobState | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(self, job_id: str, **fields: Any) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for name, value in fields.items():
                setattr(job, name, value)


# ---- factory ----------------------------------------------------------


def create_app(
    *,
    search: HybridSearch,
    indexer: Indexer,
    auth_token: str | None = None,
) -> FastAPI:
    app = FastAPI(title="Hybrid Search", version="0.1.0")
    registry = JobRegistry()

    async def require_auth(authorization: str | None = Header(default=None)) -> None:
        if auth_token is None:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        provided = authorization.removeprefix("Bearer ").strip()
        if provided != auth_token:
            raise HTTPException(status_code=401, detail="invalid bearer token")

    guard = [Depends(require_auth)]

    @app.post("/search", response_model=SearchResponse, dependencies=guard)
    async def search_endpoint(req: SearchRequest) -> SearchResponse:
        results = await search.aquery(req.query, req.k, rerank=req.rerank)
        return SearchResponse(
            results=[
                SearchResultModel(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    source_path=r.source_path,
                    score=r.score,
                    scores_breakdown=dict(r.scores_breakdown),
                )
                for r in results
            ]
        )

    @app.post("/index", response_model=IndexResponse, dependencies=guard)
    async def index_endpoint(req: IndexRequest) -> IndexResponse:
        job = await registry.create(list(req.paths))
        asyncio.create_task(_run_index_job(registry, indexer, job))
        return IndexResponse(job_id=job.id, status=job.status)

    @app.get("/jobs/{job_id}", response_model=JobResponse, dependencies=guard)
    async def job_endpoint(job_id: str) -> JobResponse:
        job = await registry.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_to_dto(job)

    @app.delete("/documents", response_model=DeleteResponse, dependencies=guard)
    async def delete_endpoint(req: DeleteRequest) -> DeleteResponse:
        if not req.paths and not req.chunk_ids:
            raise HTTPException(status_code=400, detail="paths or chunk_ids required")
        chunks_removed = 0
        files_removed = 0
        if req.chunk_ids:
            await asyncio.to_thread(indexer.vector.delete, req.chunk_ids)
            chunks_removed += await asyncio.to_thread(indexer.lexical.delete, req.chunk_ids)
        if req.paths:
            for p in req.paths:
                stats = await asyncio.to_thread(indexer.remove_path, p)
                chunks_removed += stats.chunks_removed
                files_removed += stats.files_removed
        return DeleteResponse(chunks_removed=chunks_removed, files_removed=files_removed)

    @app.get("/health", response_model=HealthResponse, dependencies=guard)
    async def health_endpoint() -> HealthResponse:
        count = await asyncio.to_thread(indexer.lexical.count)
        sources = await asyncio.to_thread(indexer.lexical.sources)
        return HealthResponse(
            chunk_count=count,
            sources_count=len(sources),
            embedding_model=indexer.vector.embedder.model_name,
            collection_name=indexer.vector.collection_name,
        )

    return app


async def _run_index_job(
    registry: JobRegistry,
    indexer: Indexer,
    job: JobState,
) -> None:
    await registry.update(job.id, status="running")
    combined = IndexStats()
    try:
        for p in job.paths:
            stats = await asyncio.to_thread(indexer.index_path, p)
            combined.merge(stats)
    except Exception as exc:  # noqa: BLE001
        await registry.update(
            job.id,
            status="error",
            error=str(exc),
            stats=combined,
            finished_at=time.time(),
        )
        log.exception("index job %s failed", job.id)
        return
    await registry.update(
        job.id,
        status="done",
        stats=combined,
        finished_at=time.time(),
    )


def _job_to_dto(job: JobState) -> JobResponse:
    stats_dict = None
    if job.stats is not None:
        stats_dict = asdict(job.stats)
    return JobResponse(
        id=job.id,
        status=job.status,
        paths=list(job.paths),
        stats=stats_dict,
        error=job.error,
        created_at=job.created_at,
        finished_at=job.finished_at,
    )
