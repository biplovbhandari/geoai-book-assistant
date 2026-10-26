from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

TOP_K_MIN = 1
TOP_K_MAX = 50


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedding_provider: str = "local"
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    embedding_dimension: int = Field(default=1024, ge=1, le=4096)
    hf_home: str = ".cache/models"

    vector_store_provider: str = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "geoai_book"

    retrieval_strategy: str = "hybrid"
    retrieval_top_k: int = Field(default=8, ge=TOP_K_MIN, le=TOP_K_MAX)
    reranker_provider: str = "noop"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    sparse_model_name: str = "prithivida/Splade_PP_en_v1"
    rrf_k: int = Field(default=60, ge=1, le=200)

    llm_provider: str = "openai_compatible"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0)
    llm_reasoning_effort: str = Field(default="none")
    llm_extra_create_params: dict = Field(default_factory=dict)
    eval_judge_reasoning_effort: str = Field(default="none")
    eval_data_dir: str = "data/eval"
    eval_stale_threshold_minutes: int = Field(default=10, ge=1, le=1440)

    book_html_dir: str = "_book"
    book_source_dir: str = "book"
    book_commit_sha: str = ""

    database_url: SecretStr = SecretStr("")

    admin_token: SecretStr = SecretStr("")

    rate_limit_per_minute: int = Field(default=30, ge=1, le=1000)
    max_concurrent_llm: int = Field(default=1, ge=1, le=8)

    app_port: int = Field(default=8000, ge=1, le=65535)
    app_env: str = "development"

    @property
    def local_api_url(self) -> str:
        """Base URL for loopback API calls (eval runner, scripts)."""
        return f"http://localhost:{self.app_port}"


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings; reads .env once, reuses on subsequent calls."""
    return Settings()
