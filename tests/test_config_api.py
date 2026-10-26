"""Tests for GET /config, the admin-only view of the active pipeline settings."""

from __future__ import annotations

import pytest
from conftest import create_test_client

from api.dependencies import Pipelines
from geoai_rag.config import get_settings

ADMIN_TOKEN = "correct-token"

# One distinct value per setting, so a response field wired to the wrong setting shows up.
_ENV = {
    "ADMIN_TOKEN": ADMIN_TOKEN,
    "RETRIEVAL_STRATEGY": "dense",
    "RETRIEVAL_TOP_K": "5",
    "RERANKER_PROVIDER": "local_cross_encoder",
    "RERANKER_MODEL_NAME": "test/reranker",
    "EMBEDDING_MODEL_NAME": "test/embedder",
    "RRF_K": "42",
    "LLM_PROVIDER": "test-provider",
    "LLM_MODEL": "qwen3:8b",
}


@pytest.fixture
def client(monkeypatch):
    """TestClient with the real admin auth and pipeline settings taken from _ENV."""
    for name, value in _ENV.items():
        monkeypatch.setenv(name, value)
    with create_test_client(monkeypatch, Pipelines()) as test_client:
        yield test_client
    # create_test_client cached settings built from _ENV; drop them so later tests re-read the
    # real environment.
    get_settings.cache_clear()


def test_config_returns_the_active_pipeline_settings(client):
    """Every response field reads its own setting, because the eval runner stores this body."""
    resp = client.get("/config", headers={"authorization": f"Bearer {ADMIN_TOKEN}"})

    assert resp.status_code == 200
    assert resp.json() == {
        "retrieval_strategy": "dense",
        "retrieval_top_k": 5,
        "reranker_provider": "local_cross_encoder",
        "reranker_model_name": "test/reranker",
        "embedding_model_name": "test/embedder",
        "rrf_k": 42,
        "llm_provider": "test-provider",
        "llm_model": "qwen3:8b",
        "eval_stale_threshold_minutes": 10,
    }


@pytest.mark.parametrize(
    ("headers", "detail"),
    [
        pytest.param({}, "Missing Bearer token", id="no-header"),
        pytest.param({"authorization": "Bearer wrong"}, "Invalid token", id="wrong-token"),
    ],
)
def test_config_requires_the_admin_token(client, headers, detail):
    """The route lives outside the /analytics router, so it must carry its own admin check."""
    resp = client.get("/config", headers=headers)

    assert resp.status_code == 401
    assert resp.json() == {"detail": detail}
