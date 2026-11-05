from __future__ import annotations

import time
from typing import Any

from geoai_rag.interfaces import Embedder, Reranker, VectorStore
from geoai_rag.models.retrieval_result import (
    ChunkStageScore,
    RetrievalResult,
    RetrievalTrace,
    StageTrace,
)
from geoai_rag.models.scored_chunk import ScoredChunk


def _elapsed_ms(start: float) -> int:
    """Return whole milliseconds elapsed since a ``time.perf_counter()`` reading."""
    return int((time.perf_counter() - start) * 1000)


def _stage_trace(
    stage: str,
    config: dict[str, Any],
    started: float,
    output: list[ScoredChunk],
    input_count: int = 0,
) -> StageTrace:
    """Record one finished retrieval stage from the chunks it produced.

    Call this right after the stage's work completes so ``latency_ms`` covers
    only that work.

    Args:
        stage: Stable stage name: "dense", "sparse", "fusion", or "reranking".
        config: Settings that describe how the stage ran (model name, RRF k).
        started: ``time.perf_counter()`` reading taken before the stage began.
        output: Chunks the stage produced, in rank order.
        input_count: Candidates entering the stage. Zero for the dense and
            sparse searches, which query the whole index rather than a
            candidate list.

    Returns:
        The stage trace, with one score entry per output chunk.
    """
    latency_ms = _elapsed_ms(started)
    return StageTrace(
        stage=stage,
        config=config,
        latency_ms=latency_ms,
        input_count=input_count,
        output_count=len(output),
        chunk_scores=[ChunkStageScore(chunk_id=sc.chunk.id, score=sc.score) for sc in output],
    )


class HybridStrategy:
    """Retrieval via dense + sparse search fused with Reciprocal Rank Fusion."""

    _OVERSAMPLING = 3

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker: Reranker,
        rrf_k: int = 60,
        *,
        embedder_model_name: str = "",
        sparse_model_name: str = "",
        reranker_model_name: str = "",
    ) -> None:
        """Wire the retrieval components and RRF tuning constant.

        The model names only label stages in the retrieval trace. They do not
        select or configure any model.
        """
        self._embedder = embedder
        self._store = vector_store
        self._reranker = reranker
        self._rrf_k = rrf_k
        self._embedder_model_name = embedder_model_name
        self._sparse_model_name = sparse_model_name
        self._reranker_model_name = reranker_model_name

    def retrieve(
        self,
        question: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Fetch from dense and sparse indexes, fuse via RRF, then rerank.

        When sparse returns empty (old collection, no terms produced),
        skips RRF and passes dense results directly to the reranker.

        Returns:
            The reranked chunks and a trace with one entry per stage that ran
            (dense, sparse, fusion, reranking). The sparse and fusion entries
            are absent when sparse returned nothing.
        """
        started = time.perf_counter()
        rerank_pool = min(top_k * self._OVERSAMPLING, 50)
        stages: list[StageTrace] = []

        # Query embedding is timed with the search because it has no stage of its own.
        stage_start = time.perf_counter()
        vector = self._embedder.embed_query(question)
        dense_results = self._store.search_dense(vector, rerank_pool, filters)
        stages.append(
            _stage_trace(
                "dense",
                {"model": self._embedder_model_name},
                stage_start,
                dense_results,
            )
        )

        stage_start = time.perf_counter()
        sparse_results = self._store.search_sparse(question, rerank_pool, filters)

        if sparse_results:
            stages.append(
                _stage_trace(
                    "sparse",
                    {"model": self._sparse_model_name},
                    stage_start,
                    sparse_results,
                )
            )

            stage_start = time.perf_counter()
            candidates = self._rrf_fuse([dense_results, sparse_results], rerank_pool)
            stages.append(
                _stage_trace(
                    "fusion",
                    {"method": "rrf", "k": self._rrf_k},
                    stage_start,
                    candidates,
                    input_count=len(dense_results) + len(sparse_results),
                )
            )
        else:
            candidates = dense_results[:rerank_pool]

        stage_start = time.perf_counter()
        reranked = self._reranker.rerank(question, candidates, top_k)
        stages.append(
            _stage_trace(
                "reranking",
                # The strategy only sees the Reranker protocol, so the reranker's
                # score threshold is not known here.
                {"model": self._reranker_model_name, "threshold": None},
                stage_start,
                reranked,
                input_count=len(candidates),
            )
        )

        return RetrievalResult(
            chunks=reranked,
            trace=RetrievalTrace(
                stages=stages,
                retrieval_query=question,
                total_latency_ms=_elapsed_ms(started),
            ),
        )

    def _rrf_fuse(self, results_lists: list[list[ScoredChunk]], top_k: int) -> list[ScoredChunk]:
        """Reciprocal Rank Fusion: score = sum of 1/(k + rank) per list.

        Ranks are 1-indexed. Chunks appearing in multiple lists get their
        scores summed, which boosts items found by both dense and sparse.
        """
        scores: dict[str, float] = {}
        chunks: dict[str, Any] = {}

        for results in results_lists:
            for rank, sc in enumerate(results, start=1):
                cid = sc.chunk.id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self._rrf_k + rank)
                chunks[cid] = sc.chunk

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        return [
            ScoredChunk(chunk=chunks[cid], score=scores[cid], ranking_method="hybrid_rrf")
            for cid in sorted_ids[:top_k]
        ]


class DenseStrategy:
    """Retrieval via dense vector similarity search."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker: Reranker,
        *,
        embedder_model_name: str = "",
        reranker_model_name: str = "",
    ) -> None:
        """Wire embedder, store, and reranker for the dense retrieval path.

        The model names only label stages in the retrieval trace. They do not
        select or configure any model.
        """
        self._embedder = embedder
        self._store = vector_store
        self._reranker = reranker
        self._embedder_model_name = embedder_model_name
        self._reranker_model_name = reranker_model_name

    def retrieve(
        self,
        question: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Embed query, search dense vectors, and rerank results.

        Args:
            question: Natural language query.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters (e.g. {"chapter": "03"}).

        Returns:
            The reranked chunks and a trace with a dense and a reranking stage.
        """
        started = time.perf_counter()
        fetch_k = min(top_k * 3, 50)
        stages: list[StageTrace] = []

        # Query embedding is timed with the search because it has no stage of its own.
        stage_start = time.perf_counter()
        vector = self._embedder.embed_query(question)
        candidates = self._store.search_dense(vector, fetch_k, filters)
        stages.append(
            _stage_trace(
                "dense",
                {"model": self._embedder_model_name},
                stage_start,
                candidates,
            )
        )

        stage_start = time.perf_counter()
        reranked = self._reranker.rerank(question, candidates, top_k)
        stages.append(
            _stage_trace(
                "reranking",
                # The strategy only sees the Reranker protocol, so the reranker's
                # score threshold is not known here.
                {"model": self._reranker_model_name, "threshold": None},
                stage_start,
                reranked,
                input_count=len(candidates),
            )
        )

        return RetrievalResult(
            chunks=reranked,
            trace=RetrievalTrace(
                stages=stages,
                retrieval_query=question,
                total_latency_ms=_elapsed_ms(started),
            ),
        )
