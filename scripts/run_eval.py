"""Run the evaluation pipeline against the book assistant API.

Each golden question is sent to POST /ask, scored for retrieval, and rated by a judge LLM for
answer similarity and faithfulness.
The results are stored in the eval tables so the dashboard can show them.

Needs the API server running (just serve or just dev), DATABASE_URL, ADMIN_TOKEN, and LLM_API_KEY
set in .env or the environment, and an OpenAI-compatible LLM for the judge.
The judge calls go through the same OpenAICompatibleClient the server uses.
They use LLM_BASE_URL (a local Ollama when unset) and the model the server reports.

Usage:
    uv run python scripts/run_eval.py
    uv run python scripts/run_eval.py --eval-set data/eval/golden_questions.yml
    uv run python scripts/run_eval.py --tags chapter-3
    uv run python scripts/run_eval.py --question q001
    uv run python scripts/run_eval.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from geoai_rag.config import Settings, get_settings
from geoai_rag.db.engine import create_db_engine, create_session_factory
from geoai_rag.eval.runner import (
    JUDGE_TIMEOUT_SECONDS,
    EvalRunner,
    EvalRunnerError,
    NoMatchingQuestionsError,
    fetch_pipeline_config,
    format_scores,
    preview_eval_set,
)
from geoai_rag.generation.llm_client import OpenAICompatibleClient

logger = logging.getLogger(__name__)

EVAL_DIR = Path(get_settings().eval_data_dir)


def _build_parser() -> argparse.ArgumentParser:
    """Define the command line options."""
    parser = argparse.ArgumentParser(
        description="Run the evaluation pipeline against the book assistant API."
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        help="Eval set YAML file to run (default: every .yml file in data/eval/).",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        metavar="TAG",
        help="Run only questions that have at least one of these tags.",
    )
    parser.add_argument(
        "--question",
        nargs="+",
        metavar="ID",
        help="Run only these questions, for example --question q001 q003.",
    )
    parser.add_argument(
        "--base-url",
        help="API base URL (default: http://localhost:<APP_PORT from settings>).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which questions would run, without calling the API, the judge, or the database.",
    )
    parser.add_argument(
        "--resume",
        type=int,
        metavar="RUN_ID",
        help="Resume a failed or cancelled eval run by its ID.",
    )
    return parser


def _eval_files(eval_set: Path | None) -> list[Path]:
    """Return the eval set files to run: the one given, or every YAML file in data/eval/."""
    if eval_set is not None:
        return [eval_set]
    files = sorted([*EVAL_DIR.glob("*.yml"), *EVAL_DIR.glob("*.yaml")])
    if not files:
        raise FileNotFoundError(f"No eval set files (*.yml or *.yaml) found in {EVAL_DIR}")
    return files


def _judge_base_url(settings: Settings) -> str:
    """Return the judge LLM's base URL from settings.

    Settings reads LLM_BASE_URL from the environment and .env, so the judge
    uses the same endpoint as the server.
    """
    return settings.llm_base_url


def _build_judge_client(settings: Settings, llm_model: str) -> OpenAICompatibleClient:
    """Build the client for the judge LLM, the same kind the server uses for its own answers.

    Going through OpenAICompatibleClient keeps the judge provider-agnostic: LLM_BASE_URL can
    point at a local Ollama, the NASA proxy, or any other OpenAI-compatible endpoint.
    """
    base_url = _judge_base_url(settings)
    logger.info("Judge LLM: %s at %s", llm_model, base_url)
    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=llm_model,
        timeout=JUDGE_TIMEOUT_SECONDS,
        reasoning_effort=settings.eval_judge_reasoning_effort,
    )


def _require_settings(settings: Settings) -> None:
    """Fail early, naming every setting a real run needs but does not have.

    LLM_API_KEY is needed even for a local Ollama, which ignores its value, because the OpenAI
    client library refuses to start without a key.
    """
    missing = []
    if not settings.database_url.get_secret_value():
        missing.append("DATABASE_URL")
    if not settings.admin_token.get_secret_value():
        missing.append("ADMIN_TOKEN")
    if not settings.llm_api_key.get_secret_value():
        missing.append("LLM_API_KEY (any value works for a local Ollama)")
    if missing:
        raise ValueError(f"set {', '.join(missing)} in .env or the environment")


def _server_llm_model(base_url: str, admin_token: str) -> str:
    """Return the LLM model the server generates with, which the judge reuses."""
    model = str(fetch_pipeline_config(base_url, admin_token).get("llm_model") or "").strip()
    if not model:
        raise EvalRunnerError("The server reports no llm_model; set LLM_MODEL where the API runs.")
    return model


def _dry_run(files: list[Path], args: argparse.Namespace) -> int:
    """Preview every eval set file, failing only when no file has a matching question."""
    previewed = 0
    for path in files:
        try:
            preview_eval_set(path, args.tags, args.question)
        except NoMatchingQuestionsError as exc:
            print(f"Skipped {path.name}: {exc}")
        else:
            previewed += 1
    if not previewed:
        print("Error: no questions matched in any eval set.", file=sys.stderr)
        return 1
    return 0


def _print_overall(summaries: list[dict[str, Any]]) -> None:
    """Print one line per finished run and the number of questions evaluated in total."""
    total = sum(summary["question_count"] for summary in summaries)
    print(f"\nOverall: {len(summaries)} eval run(s), {total} question(s) evaluated")
    for summary in summaries:
        print(
            f"  run {summary['eval_run_id']}  {summary['eval_set']}  "
            f"{summary['question_count']} questions  {format_scores(summary['averages'])}"
        )


async def _run_all(
    files: list[Path],
    args: argparse.Namespace,
    settings: Settings,
    judge_client: OpenAICompatibleClient,
) -> int:
    """Run every eval set file against the database, then print the overall summary."""
    engine = create_db_engine(settings.database_url.get_secret_value())
    try:
        runner = EvalRunner(
            base_url=args.base_url,
            llm_client=judge_client,
            session_factory=create_session_factory(engine),
            admin_token=settings.admin_token.get_secret_value(),
        )
        summaries = []
        for path in files:
            try:
                summaries.append(await runner.run(path, tags=args.tags, question_ids=args.question))
            except NoMatchingQuestionsError as exc:
                print(f"Skipped {path.name}: {exc}")
    finally:
        await engine.dispose()

    if not summaries:
        print("Error: no questions matched in any eval set.", file=sys.stderr)
        return 1
    _print_overall(summaries)
    return 0


async def _resume_run(
    args: argparse.Namespace,
    settings: Settings,
    judge_client: OpenAICompatibleClient,
) -> int:
    """Resume a failed or cancelled eval run by its ID."""
    engine = create_db_engine(settings.database_url.get_secret_value())
    try:
        runner = EvalRunner(
            base_url=args.base_url,
            llm_client=judge_client,
            session_factory=create_session_factory(engine),
            admin_token=settings.admin_token.get_secret_value(),
        )
        started = await runner.resume_run(args.resume)
        if not started.questions:
            print(f"Run {started.run_id}: all questions already scored, marked completed")
        else:
            print(
                f"Resuming run {started.run_id}: {started.eval_set_name}, "
                f"{len(started.questions)} of {started.set_size} question(s) remaining"
            )
            result = await runner.execute_run(started)
            print(f"Run {started.run_id} {result['status']}")
    finally:
        await engine.dispose()
    return 0


def main() -> int:
    """Run the requested eval sets and return the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # The judge client sends its requests through httpx, which logs one INFO line per request and
    # would bury the per-question progress lines.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = _build_parser()
    args = parser.parse_args()
    if args.resume is not None and any([args.eval_set, args.tags, args.question, args.dry_run]):
        parser.error("--resume cannot be used with --eval-set, --tags, --question, or --dry-run")
    try:
        if args.resume is not None:
            settings = get_settings()
            _require_settings(settings)
            if not args.base_url:
                args.base_url = settings.local_api_url
            llm_model = _server_llm_model(args.base_url, settings.admin_token.get_secret_value())
            judge_client = _build_judge_client(settings, llm_model)
            return asyncio.run(_resume_run(args, settings, judge_client))
        files = _eval_files(args.eval_set)
        if args.dry_run:
            return _dry_run(files, args)
        settings = get_settings()
        _require_settings(settings)
        if not args.base_url:
            args.base_url = settings.local_api_url
        llm_model = _server_llm_model(args.base_url, settings.admin_token.get_secret_value())
        judge_client = _build_judge_client(settings, llm_model)
        return asyncio.run(_run_all(files, args, settings, judge_client))
    except (OSError, ValueError, EvalRunnerError, SQLAlchemyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Interrupted. An unfinished eval run is marked cancelled.\n"
            "Resume with: uv run python scripts/run_eval.py --resume <run_id>",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    sys.exit(main())
