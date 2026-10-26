import asyncio
import logging
import sys
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.responses import Response as StarletteResponse

from api.dependencies import check_generation_ready, check_retrieval_ready, require_admin_token
from api.middleware import AccessLogMiddleware, RateLimitMiddleware, RequestIDMiddleware
from api.routes.analytics import router as analytics_router
from api.routes.ask import router as ask_router
from api.routes.chat import router as chat_router
from api.routes.eval import router as eval_router
from api.routes.feedback import router as feedback_router
from api.routes.search import router as search_router
from geoai_rag import __version__
from geoai_rag.config import Settings, get_settings
from geoai_rag.db.bootstrap import ensure_active_deployment
from geoai_rag.db.engine import check_db_schema_status, create_db_engine, create_session_factory
from geoai_rag.db.models import EvalRun
from geoai_rag.factories import create_pipelines
from geoai_rag.logging_config import configure_logging

logger = logging.getLogger(__name__)


async def _sweep_orphaned_api_runs(
    session_factory: async_sessionmaker[AsyncSession],
    stale_threshold_minutes: int = 10,
) -> None:
    """Mark orphaned and stale eval runs as failed at startup.

    API runs only exist in the server process.
    Any still active at startup are orphans from a crash or restart.
    CLI runs are left alone because the CLI process may still be alive.
    A second pass marks any run (API or CLI) as failed when its heartbeat
    has not advanced within EVAL_STALE_THRESHOLD_MINUTES, indicating a stuck or crashed runner.
    """
    async with session_factory() as session:
        cursor = await session.execute(
            update(EvalRun)
            .where(EvalRun.source == "api", EvalRun.status.in_(["pending", "running"]))
            .values(status="failed")
        )
        count = cursor.rowcount  # type: ignore[union-attr]
        if count:
            logger.info("Marked %d orphaned API eval run(s) as failed", count)
        await session.commit()

        # Sweep stale runs (any source) with no heartbeat progress in stale_threshold_minutes minutes
        stale_threshold = datetime.now(UTC) - timedelta(minutes=stale_threshold_minutes)
        stale_cursor = await session.execute(
            update(EvalRun)
            .where(
                EvalRun.status.in_(["running"]),
                EvalRun.last_heartbeat_at.isnot(None),
                EvalRun.last_heartbeat_at < stale_threshold,
            )
            .values(status="failed")
        )
        stale_count = stale_cursor.rowcount  # type: ignore[union-attr]
        if stale_count:
            logger.info(
                "Marked %d stale eval run(s) as failed (no heartbeat in %d min)",
                stale_count,
                stale_threshold_minutes,
            )
        await session.commit()


async def _auto_load_eval_sets(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    """Load eval sets from YAML files into the database at startup."""
    try:
        from geoai_rag.eval.runner import load_eval_set, sync_eval_set_to_db

        eval_dir = Path(settings.eval_data_dir)
        if eval_dir.is_dir():
            for yml_path in sorted(eval_dir.glob("*.yml")):
                try:
                    eval_set_data = load_eval_set(yml_path)
                    await sync_eval_set_to_db(session_factory, eval_set_data)
                    logger.info(
                        "Loaded eval set '%s' from %s", eval_set_data["name"], yml_path.name
                    )
                except Exception:
                    logger.warning("Failed to load eval set from %s", yml_path.name, exc_info=True)
    except ImportError:
        logger.debug("pyyaml not installed; skipping eval set auto-load")


async def _periodic_stale_sweep(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    """Background task: sweep stale eval runs at the configured interval."""
    interval = settings.eval_stale_threshold_minutes * 60
    while True:
        await asyncio.sleep(interval)
        try:
            await _sweep_orphaned_api_runs(
                session_factory,
                stale_threshold_minutes=settings.eval_stale_threshold_minutes,
            )
        except Exception:
            logger.warning("Periodic stale sweep failed", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = get_settings()

    is_dev = "--reload" in sys.argv
    configure_logging(json=not is_dev, level="INFO")

    try:
        app.state.pipelines = create_pipelines(app.state.settings)
    except Exception:
        logger.exception("Failed to build pipelines; /search, /ask, and /chat will return 503")
        app.state.pipelines = None

    app.state.db_engine = None
    app.state.db_session_factory = None
    app.state.db_configured = False

    status = None
    url = app.state.settings.database_url.get_secret_value()
    if url:
        app.state.db_configured = True
        try:
            engine = create_db_engine(url)
            app.state.db_engine = engine
            app.state.db_session_factory = create_session_factory(engine)
            status = await check_db_schema_status(engine)
            if status == "ready":
                logger.info("Database connected and schema is current")
            else:
                logger.warning("Database status at startup: %s", status)
        except Exception:
            logger.warning(
                "Database configuration error; recording and admin features disabled",
                exc_info=True,
            )

    app.state.llm_semaphore = threading.Semaphore(app.state.settings.max_concurrent_llm)

    app.state.active_deployment_id = None
    app.state.active_index_run_id = None
    sweep_task = None

    db_ready = app.state.db_session_factory is not None and status == "ready"
    if db_ready:
        assert app.state.db_session_factory is not None
        session_factory = app.state.db_session_factory
        embedder = getattr(app.state.pipelines, "embedder", None) if app.state.pipelines else None
        expected_dim = app.state.settings.embedding_dimension
        if embedder is None or embedder.get_dimension() != expected_dim:
            if embedder is not None:
                logger.warning(
                    "Embedding dimension %d != %d; recording disabled",
                    embedder.get_dimension(),
                    expected_dim,
                )
        else:
            async with session_factory() as session:
                result = await ensure_active_deployment(session, app.state.settings)
                if result is not None:
                    app.state.active_deployment_id, app.state.active_index_run_id = result

        await _sweep_orphaned_api_runs(
            session_factory,
            stale_threshold_minutes=app.state.settings.eval_stale_threshold_minutes,
        )

        await _auto_load_eval_sets(session_factory, app.state.settings)

        sweep_task = asyncio.create_task(_periodic_stale_sweep(session_factory, app.state.settings))

    try:
        yield
    finally:
        if sweep_task is not None:
            sweep_task.cancel()
        if app.state.db_engine is not None:
            await app.state.db_engine.dispose()


app = FastAPI(
    title="Geo-AI Book Assistant",
    version=__version__,
    lifespan=lifespan,
)

Instrumentator().instrument(app)


@app.get("/metrics", include_in_schema=False, dependencies=[Depends(require_admin_token)])
async def metrics():
    """Prometheus metrics endpoint, admin-gated."""
    return StarletteResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    """Return application readiness for retrieval, generation, streaming chat, and database."""
    pipelines = getattr(app.state, "pipelines", None)
    ret_ready, _ = check_retrieval_ready(pipelines)
    gen_ready, _ = check_generation_ready(pipelines)
    stream_ok = False
    if gen_ready and pipelines is not None and pipelines.query is not None:
        stream_ok = callable(getattr(pipelines.query._llm_client, "chat_stream", None))
    chat_ok = ret_ready and gen_ready and stream_ok

    db_engine = getattr(app.state, "db_engine", None)
    if db_engine is not None:
        db_status = await check_db_schema_status(db_engine)
    elif getattr(app.state, "db_configured", False):
        db_status = "misconfigured"
    else:
        db_status = "disabled"

    return {
        "status": "ok",
        "version": __version__,
        "retrieval": "ready" if ret_ready else "unavailable",
        "generation": "ready" if gen_ready else "unavailable",
        "chat": "ready" if chat_ok else "unavailable",
        "database": db_status,
    }


class PipelineConfig(BaseModel):
    """Active retrieval and generation settings, read from the environment."""

    retrieval_strategy: str
    retrieval_top_k: int
    reranker_provider: str
    reranker_model_name: str
    embedding_model_name: str
    rrf_k: int
    llm_provider: str
    llm_model: str
    eval_stale_threshold_minutes: int


@app.get("/config", response_model=PipelineConfig, dependencies=[Depends(require_admin_token)])
async def pipeline_config():
    """Return the active retrieval and generation configuration to an admin."""
    s = get_settings()
    return PipelineConfig(
        retrieval_strategy=s.retrieval_strategy,
        retrieval_top_k=s.retrieval_top_k,
        reranker_provider=s.reranker_provider,
        reranker_model_name=s.reranker_model_name,
        embedding_model_name=s.embedding_model_name,
        rrf_k=s.rrf_k,
        llm_provider=s.llm_provider,
        llm_model=s.llm_model,
        eval_stale_threshold_minutes=s.eval_stale_threshold_minutes,
    )


app.add_middleware(
    RateLimitMiddleware,
    max_requests=get_settings().rate_limit_per_minute,
)
app.add_middleware(AccessLogMiddleware)
app.add_middleware(RequestIDMiddleware)

app.include_router(ask_router)
app.include_router(chat_router)
app.include_router(search_router)
app.include_router(feedback_router)
app.include_router(analytics_router)
app.include_router(eval_router, prefix="/eval")

# --- Static file mounts (catch-all book mount must be last) ---

_dashboard_dir = Path(__file__).resolve().parent / "dashboard"
_dashboard_dir.mkdir(exist_ok=True)


@app.get("/dashboard", include_in_schema=False)
async def _dashboard_redirect():
    return RedirectResponse("/dashboard/", status_code=301)


app.mount("/dashboard", StaticFiles(directory=_dashboard_dir, html=True), name="dashboard")

_book_html_dir = Path(get_settings().book_html_dir)
_book_html_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=_book_html_dir, html=True), name="book")
