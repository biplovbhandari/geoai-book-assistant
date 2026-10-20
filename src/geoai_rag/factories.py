"""Factory functions for building pipeline components from config."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from geoai_rag.config import Settings

if TYPE_CHECKING:
    from geoai_rag.interfaces import Embedder, VectorStore
    from geoai_rag.query import QueryPipeline

logger = logging.getLogger(__name__)

VIDEO_CHAPTER_MAP_PATH = Path("data/video_chapter_map.yml")


@dataclass
class Pipelines:
    """Container for wired pipeline objects, set on app.state at startup."""

    indexing: object | None = field(default=None)
    query: QueryPipeline | None = field(default=None)
    vector_store: VectorStore | None = field(default=None)
    embedder: Embedder | None = field(default=None)


def _create_embedder(config: Settings):
    """Build the dense embedder from config (provider pattern)."""
    from geoai_rag.indexing.embedder import LocalEmbeddingModel

    if config.embedding_provider == "local":
        return LocalEmbeddingModel(config.embedding_model_name, config.hf_home)
    raise ValueError(f"Unknown embedding_provider: {config.embedding_provider}")


def _create_sparse_embedder(config: Settings):
    """Build the sparse embedder for hybrid search.

    cache_dir is derived from hf_home so fastembed persists models alongside
    sentence-transformers weights -- /models/fastembed in Docker, writable
    local path during uv run.
    """
    from geoai_rag.indexing.sparse_embedder import LocalSparseEmbeddingModel

    return LocalSparseEmbeddingModel(
        model_name=config.sparse_model_name,
        cache_dir=str(Path(config.hf_home) / "fastembed"),
    )


def _create_vector_store(
    config: Settings, dense_dim: int, create_if_missing: bool, sparse_embedder=None
):
    """Build the vector store from config (provider pattern).

    sparse_embedder is passed through to QdrantStore so it can compute and
    store sparse vectors during upsert and encode queries for sparse search.
    """
    from geoai_rag.indexing.qdrant_store import QdrantStore

    if config.vector_store_provider == "qdrant":
        return QdrantStore(
            config.qdrant_url,
            config.qdrant_collection,
            dense_dim,
            create_if_missing=create_if_missing,
            sparse_embedder=sparse_embedder,
        )
    raise ValueError(f"Unknown vector_store_provider: {config.vector_store_provider}")


def _load_video_chapter_map() -> dict[str, dict]:
    """Load the video-to-chapter mapping from data/video_chapter_map.yml.

    A broken or missing mapping file must never crash the factory -- all
    error paths return an empty dict so VideoChunker falls back gracefully.
    yaml is imported lazily because pyyaml lives in the indexer/dev groups
    only and is not available in the API image.
    """
    import yaml

    map_path = VIDEO_CHAPTER_MAP_PATH
    if not map_path.exists():
        return {}

    try:
        with open(map_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.warning("Failed to parse data/video_chapter_map.yml: %s", e)
        return {}

    if not isinstance(data, dict):
        return {}

    videos = data.get("videos")
    if not isinstance(videos, dict):
        return {}

    chapter_map: dict[str, dict] = {}
    for video_id, entry in videos.items():
        if not isinstance(entry, dict):
            logger.warning("Skipping video entry %r in chapter map: not a dict", video_id)
            continue
        chapter_map[video_id] = entry

    return chapter_map


def _create_parsers() -> dict:
    """Build the file-extension-to-parser registry (registry pattern)."""
    from geoai_rag.indexing.parsers import (
        BibParser,
        MarkdownParser,
        NotebookParser,
        PdfParser,
        TranscriptParser,
    )

    return {
        ".md": MarkdownParser(),
        ".qmd": MarkdownParser(),
        ".ipynb": NotebookParser(),
        ".bib": BibParser(),
        ".pdf": PdfParser(),
        ".json": TranscriptParser(),
    }


def _create_chunkers() -> dict:
    """Build the file-extension-to-chunker registry (registry pattern)."""
    from geoai_rag.indexing.chunkers import (
        BibChunker,
        NotebookChunker,
        PdfChunker,
        SectionChunker,
        VideoChunker,
    )

    return {
        ".md": SectionChunker(),
        ".qmd": SectionChunker(),
        ".ipynb": NotebookChunker(),
        ".bib": BibChunker(),
        ".pdf": PdfChunker(),
        ".json": VideoChunker(chapter_map=_load_video_chapter_map()),
    }


def _create_reranker(config: Settings):
    """Build the reranker from config (provider pattern).

    Validates model name only when local_cross_encoder is selected,
    so noop users are not affected by an empty RERANKER_MODEL_NAME.
    """
    from geoai_rag.retrieval.rerankers import NoOpReranker

    if config.reranker_provider == "noop":
        return NoOpReranker()
    if config.reranker_provider == "local_cross_encoder":
        from geoai_rag.retrieval.rerankers import LocalCrossEncoderReranker

        model_name = config.reranker_model_name.strip()
        if not model_name:
            raise ValueError(
                "RERANKER_MODEL_NAME must not be empty when RERANKER_PROVIDER=local_cross_encoder"
            )
        return LocalCrossEncoderReranker(model_name, config.hf_home)
    raise ValueError(f"Unknown reranker_provider: {config.reranker_provider}")


def _create_retrieval_strategy(config: Settings, embedder, store, reranker):
    """Build the retrieval strategy from config (provider pattern).

    Hybrid passes rrf_k from config so RRF fusion can be tuned without
    code changes.
    """
    from geoai_rag.retrieval.strategies import DenseStrategy, HybridStrategy

    if config.retrieval_strategy == "dense":
        return DenseStrategy(embedder, store, reranker)
    if config.retrieval_strategy == "hybrid":
        return HybridStrategy(embedder, store, reranker, rrf_k=config.rrf_k)
    raise ValueError(f"Unknown retrieval_strategy: {config.retrieval_strategy}")


def _create_llm_client(config: Settings):
    """Build the LLM client from config (provider pattern)."""
    from geoai_rag.generation.llm_client import OpenAICompatibleClient

    if config.llm_provider == "openai_compatible":
        return OpenAICompatibleClient(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key.get_secret_value(),
            model=config.llm_model,
            timeout=config.llm_timeout_seconds,
            reasoning_effort=config.llm_reasoning_effort,
            extra_create_params=config.llm_extra_create_params,
        )
    raise ValueError(f"Unknown llm_provider: {config.llm_provider}")


def _create_context_builder():
    """Build the context builder (direct pattern, single adapter)."""
    from geoai_rag.generation.context_builder import DefaultContextBuilder

    return DefaultContextBuilder()


def _create_citation_builder():
    """Build the citation builder (direct pattern, single adapter)."""
    from geoai_rag.citations.citation_builder import DefaultCitationBuilder

    return DefaultCitationBuilder()


def create_indexing_pipeline(config: Settings):
    """Build the indexing pipeline for CLI use (create_if_missing=True).

    Always creates a sparse embedder regardless of RETRIEVAL_STRATEGY so the
    index contains both dense and sparse vectors. Switching strategies later
    is then a config-only change with no re-index.
    """
    from geoai_rag.indexing.pipeline import IndexingPipeline

    embedder = _create_embedder(config)
    sparse_embedder = _create_sparse_embedder(config)
    store = _create_vector_store(
        config, embedder.get_dimension(), create_if_missing=True, sparse_embedder=sparse_embedder
    )

    return IndexingPipeline(
        parsers=_create_parsers(),
        chunkers=_create_chunkers(),
        embedder=embedder,
        vector_store=store,
        commit_sha=config.book_commit_sha,
    )


def create_pipelines(config: Settings) -> Pipelines:
    """Build the query-side pipelines for FastAPI startup (create_if_missing=False).

    Sparse embedder is only created when RETRIEVAL_STRATEGY=hybrid to avoid
    loading the ~500 MB SPLADE model in dense-only mode. A sparse failure here
    crashes startup (not degraded like LLM) because hybrid was explicitly requested.
    """
    from geoai_rag.query import QueryPipeline

    embedder = _create_embedder(config)
    sparse_embedder = (
        _create_sparse_embedder(config) if config.retrieval_strategy == "hybrid" else None
    )
    store = _create_vector_store(
        config, embedder.get_dimension(), create_if_missing=False, sparse_embedder=sparse_embedder
    )
    reranker = _create_reranker(config)
    strategy = _create_retrieval_strategy(config, embedder, store, reranker)

    context_builder = None
    llm_client = None
    citation_builder = None
    try:
        context_builder = _create_context_builder()
        llm_client = _create_llm_client(config)
        citation_builder = _create_citation_builder()
    except Exception:
        logger.warning(
            "LLM client creation failed; /ask will return 503 but /search is unaffected",
            exc_info=True,
        )

    return Pipelines(
        query=QueryPipeline(
            strategy=strategy,
            context_builder=context_builder,
            llm_client=llm_client,
            citation_builder=citation_builder,
            top_k=config.retrieval_top_k,
        ),
        vector_store=store,
        embedder=embedder,
    )
