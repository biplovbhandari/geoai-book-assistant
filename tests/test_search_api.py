from unittest.mock import MagicMock

from conftest import create_test_client, make_retrieval_result, make_scored_chunk


def _make_fake_pipelines(search_results=None, count=100):
    from api.dependencies import Pipelines

    query = MagicMock()
    query.search.return_value = make_retrieval_result(search_results or [make_scored_chunk()])

    store = MagicMock()
    store.count.return_value = count

    return Pipelines(query=query, vector_store=store)  # type: ignore[arg-type]


class TestSearch:
    def test_valid_query_returns_chunks(self, monkeypatch):
        client = create_test_client(monkeypatch, _make_fake_pipelines())
        with client:
            resp = client.post("/search", json={"question": "What is U-Net?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "chunks" in body
        assert len(body["chunks"]) == 1
        assert body["chunks"][0]["chunk"]["content"] == "U-Net architecture"

    def test_dotted_filter_key_rejected(self, monkeypatch):
        client = create_test_client(monkeypatch, _make_fake_pipelines())
        with client:
            resp = client.post(
                "/search",
                json={"question": "test", "filters": {"metadata.chapter": "03"}},
            )
        assert resp.status_code == 422

    def test_transient_qdrant_failure_returns_503(self, monkeypatch):
        pipelines = _make_fake_pipelines()
        pipelines.query.search.side_effect = Exception("connection refused")  # type: ignore[union-attr]
        client = create_test_client(monkeypatch, pipelines)
        with client:
            resp = client.post("/search", json={"question": "test"})
        assert resp.status_code == 503

    def test_top_k_forwarded_to_pipeline(self, monkeypatch):
        # Guards the `body.top_k if not None` branch: a user-supplied top_k must
        # thread through to the retrieval call. Hold the query mock locally so its
        # call is assertable (pipelines.query is typed as a real QueryPipeline).
        from api.dependencies import Pipelines

        query = MagicMock()
        query.search.return_value = make_retrieval_result([make_scored_chunk()])
        store = MagicMock()
        store.count.return_value = 100
        pipelines = Pipelines(query=query, vector_store=store)  # type: ignore[arg-type]
        client = create_test_client(monkeypatch, pipelines)
        with client:
            resp = client.post("/search", json={"question": "q", "top_k": 7})
        assert resp.status_code == 200
        query.search.assert_called_once()
        assert query.search.call_args.args[1] == 7
