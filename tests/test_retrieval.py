import pytest
from conftest import make_scored_chunk

from geoai_rag.retrieval import DenseStrategy, HybridStrategy, NoOpReranker


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.1] * 10 for _ in texts]

    def embed_query(self, text):
        return [0.5] * 10

    def get_dimension(self):
        return 10


class FakeStore:
    def __init__(self, results, sparse_results=None):
        self._results = results
        self._sparse_results = sparse_results or []

    def upsert(self, chunks, vectors):
        pass

    def search_dense(self, vector, top_k=10, filters=None):
        return self._results[:top_k]

    def search_sparse(self, text, top_k=10, filters=None):
        return self._sparse_results[:top_k]

    def get_by_ids(self, ids):
        return []

    def delete_by_source(self, source_path):
        pass

    def count(self):
        return 100


# --- DenseStrategy tests ---


def test_dense_strategy_retrieves_and_reranks():
    canned = [
        make_scored_chunk("U-Net architecture", 0.95),
        make_scored_chunk("CNN basics", 0.80),
    ]
    store = FakeStore(canned)
    strategy = DenseStrategy(FakeEmbedder(), store, NoOpReranker())

    results = strategy.retrieve("What is U-Net?", top_k=2).chunks

    assert len(results) == 2
    assert results[0].chunk.content == "U-Net architecture"
    assert results[0].score == 0.95
    assert results[1].chunk.content == "CNN basics"


# --- HybridStrategy tests ---


def test_hybrid_strategy_fuses_dense_and_sparse():
    shared_chunk = make_scored_chunk("U-Net architecture", 0.95)
    dense_only = make_scored_chunk("CNN basics", 0.80)
    sparse_only = make_scored_chunk("NDVI index", 0.70, method="sparse")
    sparse_shared = make_scored_chunk("U-Net architecture", 0.60, method="sparse")
    sparse_shared.chunk = shared_chunk.chunk

    store = FakeStore(
        results=[shared_chunk, dense_only],
        sparse_results=[sparse_shared, sparse_only],
    )
    strategy = HybridStrategy(FakeEmbedder(), store, NoOpReranker(), rrf_k=60)

    results = strategy.retrieve("What is U-Net?", top_k=10).chunks

    assert results[0].chunk.content == "U-Net architecture"
    assert results[0].ranking_method == "hybrid_rrf"
    assert results[0].score == pytest.approx(1 / 61 + 1 / 61)
    assert results[0].score > results[1].score

    contents = {r.chunk.content for r in results}
    assert "CNN basics" in contents
    assert "NDVI index" in contents


def test_hybrid_strategy_empty_sparse_skips_rrf():
    dense = [
        make_scored_chunk("Dense result", 0.95),
    ]
    store = FakeStore(results=dense, sparse_results=[])
    strategy = HybridStrategy(FakeEmbedder(), store, NoOpReranker(), rrf_k=60)

    results = strategy.retrieve("query", top_k=5).chunks

    assert len(results) == 1
    assert results[0].score == 0.95
    assert results[0].ranking_method == "dense"


def test_hybrid_strategy_both_empty():
    store = FakeStore(results=[], sparse_results=[])
    strategy = HybridStrategy(FakeEmbedder(), store, NoOpReranker(), rrf_k=60)

    results = strategy.retrieve("query", top_k=5).chunks

    assert results == []


def test_hybrid_strategy_respects_top_k():
    dense = [make_scored_chunk(f"Dense {i}", 0.9 - i * 0.1) for i in range(5)]
    sparse = [make_scored_chunk(f"Sparse {i}", 0.8 - i * 0.1, method="sparse") for i in range(5)]

    store = FakeStore(results=dense, sparse_results=sparse)
    strategy = HybridStrategy(FakeEmbedder(), store, NoOpReranker(), rrf_k=60)

    results = strategy.retrieve("query", top_k=3).chunks

    assert len(results) == 3


# --- CrossEncoder reranker tests ---
class FakeReranker:
    """Records candidates received and reverses their order."""

    def __init__(self):
        self.last_candidates = []
        self.last_top_k = 0

    def rerank(self, query, candidates, top_k):
        self.last_candidates = list(candidates)
        self.last_top_k = top_k
        return list(reversed(candidates))[:top_k]


def test_cross_encoder_reranker_reorders():
    from unittest.mock import MagicMock, patch

    with patch("sentence_transformers.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        mock_model.num_labels = 1
        mock_model.predict.return_value = [0.2, 0.9, 0.5]
        MockCE.return_value = mock_model

        from geoai_rag.retrieval.rerankers import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker("fake-model", "/tmp")

        candidates = [
            make_scored_chunk("Low", 0.1),
            make_scored_chunk("High", 0.1),
            make_scored_chunk("Mid", 0.1),
        ]

        results = reranker.rerank("query", candidates, top_k=3)

        assert results[0].chunk.content == "High"
        assert results[0].score == pytest.approx(0.9)
        assert results[1].chunk.content == "Mid"
        assert results[2].chunk.content == "Low"
        assert all(r.ranking_method == "reranked" for r in results)


def test_cross_encoder_reranker_empty_candidates():
    from unittest.mock import MagicMock, patch

    with patch("sentence_transformers.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        mock_model.num_labels = 1
        MockCE.return_value = mock_model

        from geoai_rag.retrieval.rerankers import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker("fake-model", "/tmp")
        results = reranker.rerank("query", [], top_k=5)
        assert results == []


def test_cross_encoder_reranker_replaces_nonfinite_scores():
    import math
    from unittest.mock import MagicMock, patch

    import numpy as np

    with patch("sentence_transformers.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        mock_model.num_labels = 1
        mock_model.predict.return_value = np.array([float("nan"), float("inf"), float("-inf"), 0.5])
        MockCE.return_value = mock_model

        from geoai_rag.retrieval.rerankers import (
            _NON_FINITE_SENTINEL,
            LocalCrossEncoderReranker,
        )

        reranker = LocalCrossEncoderReranker("fake-model", "/tmp")

        candidates = [
            make_scored_chunk("NaN", 0.1),
            make_scored_chunk("Inf", 0.1),
            make_scored_chunk("NegInf", 0.1),
            make_scored_chunk("Valid", 0.1),
        ]

        results = reranker.rerank("query", candidates, top_k=4)

        assert results[0].chunk.content == "Valid"
        assert results[0].score == pytest.approx(0.5)
        assert all(r.score == _NON_FINITE_SENTINEL for r in results[1:])
        assert all(math.isfinite(r.score) for r in results)


# --- Over-fetch tests ---
class TrackingStore(FakeStore):
    """FakeStore that records the top_k passed to search methods."""

    def __init__(self, results, sparse_results=None):
        super().__init__(results, sparse_results)
        self.dense_top_k = None
        self.sparse_top_k = None

    def search_dense(self, vector, top_k=10, filters=None):
        self.dense_top_k = top_k
        return self._results[:top_k]

    def search_sparse(self, text, top_k=10, filters=None):
        self.sparse_top_k = top_k
        return self._sparse_results[:top_k]


def test_dense_strategy_overfetches_for_reranker():
    canned = [make_scored_chunk(f"Result {i}", 0.9 - i * 0.05) for i in range(10)]
    store = TrackingStore(canned)
    fake_reranker = FakeReranker()
    strategy = DenseStrategy(FakeEmbedder(), store, fake_reranker)

    strategy.retrieve("query", top_k=3)

    assert store.dense_top_k == 9
    assert len(fake_reranker.last_candidates) == 9
    assert fake_reranker.last_top_k == 3


def test_hybrid_strategy_overfetches_for_reranker():
    dense = [make_scored_chunk(f"Dense {i}", 0.9 - i * 0.05) for i in range(10)]
    sparse = [make_scored_chunk(f"Sparse {i}", 0.8 - i * 0.05, method="sparse") for i in range(10)]
    store = TrackingStore(dense, sparse)
    fake_reranker = FakeReranker()
    strategy = HybridStrategy(FakeEmbedder(), store, fake_reranker, rrf_k=60)

    strategy.retrieve("query", top_k=3)

    assert store.dense_top_k == 9
    assert store.sparse_top_k == 9
    assert len(fake_reranker.last_candidates) == 9
    assert fake_reranker.last_top_k == 3
    assert all(sc.ranking_method == "hybrid_rrf" for sc in fake_reranker.last_candidates)
    contents = {sc.chunk.content for sc in fake_reranker.last_candidates}
    assert any(c.startswith("Dense") for c in contents)
    assert any(c.startswith("Sparse") for c in contents)


def test_hybrid_sparse_empty_overfetches_for_reranker():
    dense = [make_scored_chunk(f"Dense {i}", 0.9 - i * 0.05) for i in range(10)]
    store = TrackingStore(dense, sparse_results=[])
    fake_reranker = FakeReranker()
    strategy = HybridStrategy(FakeEmbedder(), store, fake_reranker, rrf_k=60)

    strategy.retrieve("query", top_k=3)

    assert store.dense_top_k == 9
    assert len(fake_reranker.last_candidates) == 9
    assert fake_reranker.last_top_k == 3


def test_cross_encoder_reranker_rejects_bad_score_shape():
    from unittest.mock import MagicMock, patch

    import numpy as np

    with patch("sentence_transformers.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        mock_model.num_labels = 1
        mock_model.predict.return_value = np.array([0.1, 0.2, 0.3])  # 3 scores, 2 candidates
        MockCE.return_value = mock_model

        from geoai_rag.retrieval.rerankers import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker("fake-model", "/tmp")
        candidates = [make_scored_chunk("A", 0.1), make_scored_chunk("B", 0.1)]
        with pytest.raises(ValueError, match="returned shape"):
            reranker.rerank("query", candidates, top_k=2)


def test_cross_encoder_reranker_truncates_to_top_k():
    from unittest.mock import MagicMock, patch

    with patch("sentence_transformers.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        mock_model.num_labels = 1
        mock_model.predict.return_value = [0.1, 0.9, 0.5, 0.7]
        MockCE.return_value = mock_model

        from geoai_rag.retrieval.rerankers import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker("fake-model", "/tmp")
        candidates = [make_scored_chunk(c, 0.1) for c in ("A", "B", "C", "D")]

        results = reranker.rerank("query", candidates, top_k=2)

        assert len(results) == 2
        assert results[0].chunk.content == "B"  # 0.9, highest
        assert results[1].chunk.content == "D"  # 0.7, second
