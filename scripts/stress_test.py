"""Stress test the /chat SSE endpoint with concurrent requests.

Reports per-request latency, time to first token (TTFT), throughput, and error
rates for each concurrency level. This is a plain HTTP client and does not
import any geoai_rag modules.

Usage:
    uv run python scripts/stress_test.py
    uv run python scripts/stress_test.py --concurrency 1,2,5 --rounds 2
    uv run python scripts/stress_test.py --url http://macmini.local:8000 --timeout 180
    uv run python scripts/stress_test.py "What is U-Net?" "How is NDVI computed?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

DEFAULT_QUESTIONS_PATH = "data/eval/golden_questions.yml"

FALLBACK_QUESTIONS = [
    "What is semantic segmentation in the context of remote sensing?",
    "How does the book use deep learning for land cover classification?",
    "What is U-Net and why is it used for segmentation?",
    "How do I prepare satellite imagery for model training?",
    "What is transfer learning and when should I use it?",
]

QUESTION_COLUMN_WIDTH = 34


@dataclass
class RequestResult:
    """Outcome of a single /chat request."""

    question: str
    ttft: float | None = None
    total: float = 0.0
    tokens: int = 0
    status: str = "ok"

    @property
    def ok(self) -> bool:
        """Return True when the stream finished with a ``done`` event."""
        return self.status == "ok"


@dataclass
class LevelStats:
    """Aggregated results for one concurrency level across all rounds."""

    concurrency: int
    results: list[RequestResult]
    wall_times: list[float]


def load_golden_questions(path: str = DEFAULT_QUESTIONS_PATH) -> list[str]:
    """Load questions from the golden YAML, falling back to a built-in list.

    The fallback is used when PyYAML is missing, the file is absent or
    malformed, or it contains no questions.
    """
    try:
        import yaml
    except ImportError:
        print("Warning: pyyaml not available, using built-in questions", file=sys.stderr)
        return list(FALLBACK_QUESTIONS)

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        questions = [q["question"] for q in data["questions"]]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"Warning: could not load {path} ({exc}), using built-in questions", file=sys.stderr)
        return list(FALLBACK_QUESTIONS)

    if not questions:
        print(f"Warning: no questions in {path}, using built-in questions", file=sys.stderr)
        return list(FALLBACK_QUESTIONS)
    return questions


def percentile(values: list[float], pct: float) -> float:
    """Return the ``pct`` percentile (nearest-rank) of ``values``; 0.0 if empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(pct / 100 * len(ordered) + 0.5)) - 1))
    return ordered[index]


def _fmt_secs(value: float | None) -> str:
    """Format seconds for display, or ``-`` when unavailable."""
    return "-" if value is None else f"{value:.1f}s"


def _truncate(text: str, width: int) -> str:
    """Truncate ``text`` to ``width`` characters, adding an ellipsis if cut."""
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


async def send_chat(
    client: httpx.AsyncClient, url: str, question: str, timeout: float
) -> RequestResult:
    """POST one question to /chat and consume the SSE stream.

    Never raises for expected failures; the status field records the outcome:
    ``ok``, ``timeout``, ``connection error``, ``error`` (server error event),
    ``incomplete`` (stream ended without ``done``), or ``http <code>``.
    """
    result = RequestResult(question=question)
    body = {"question": question, "history": []}
    start = time.perf_counter()
    got_done = False

    try:
        async with client.stream("POST", f"{url}/chat", json=body, timeout=timeout) as response:
            if response.status_code != 200:
                result.status = f"http {response.status_code}"
                return result
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                event_type = event.get("type")
                if event_type == "token":
                    if result.ttft is None:
                        result.ttft = time.perf_counter() - start
                    result.tokens += len(event.get("content", ""))
                elif event_type == "error":
                    result.status = "error"
                    return result
                elif event_type == "done":
                    got_done = True
                    break
    except httpx.TimeoutException:
        result.status = "timeout"
    except httpx.ConnectError:
        result.status = "connection error"
    except httpx.HTTPError as exc:
        result.status = f"error ({type(exc).__name__})"
    finally:
        result.total = time.perf_counter() - start

    if result.status == "ok" and not got_done:
        result.status = "incomplete"
    return result


def pick_questions(questions: list[str], count: int, offset: int) -> list[str]:
    """Pick ``count`` questions starting at ``offset``, cycling when needed."""
    return [questions[(offset + i) % len(questions)] for i in range(count)]


def print_round(results: list[RequestResult], wall: float) -> None:
    """Print the per-request table and summary statistics for one round."""
    print(f"  {'#':>2}  {'Question':<{QUESTION_COLUMN_WIDTH}}  {'TTFT':>6}  {'Total':>7}  {'Tokens':>6}  Status")
    for i, r in enumerate(results, 1):
        print(
            f"  {i:>2}  {_truncate(r.question, QUESTION_COLUMN_WIDTH):<{QUESTION_COLUMN_WIDTH}}  "
            f"{_fmt_secs(r.ttft):>6}  {r.total:>6.1f}s  {r.tokens:>6}  {r.status}"
        )

    totals = [r.total for r in results if r.ok]
    ttfts = [r.ttft for r in results if r.ok and r.ttft is not None]
    errors = sum(1 for r in results if not r.ok)

    print()
    print(f"  Wall time:   {wall:.1f}s")
    print(f"  Throughput:  {len(totals) / wall if wall > 0 else 0.0:.2f} req/s")
    if totals:
        print(
            f"  Latency:     min={min(totals):.1f}s  avg={statistics.mean(totals):.1f}s  "
            f"median={statistics.median(totals):.1f}s  max={max(totals):.1f}s"
            + (f"  stdev={statistics.stdev(totals):.1f}s" if len(totals) > 1 else "")
        )
    if ttfts:
        print(f"  TTFT:        min={min(ttfts):.1f}s  avg={statistics.mean(ttfts):.1f}s  max={max(ttfts):.1f}s")
    print(f"  Errors:      {errors}/{len(results)}")
    print()


def print_overall(levels: list[LevelStats]) -> None:
    """Print the overall summary table across all concurrency levels."""
    print("=== Overall Summary ===\n")
    print(
        f"  {'Concurrency':>11}  {'Requests':>8}  {'Errors':>6}  {'Avg Latency':>11}  "
        f"{'P95 Latency':>11}  {'Avg TTFT':>8}  {'Throughput':>10}"
    )
    for level in levels:
        totals = [r.total for r in level.results if r.ok]
        ttfts = [r.ttft for r in level.results if r.ok and r.ttft is not None]
        errors = sum(1 for r in level.results if not r.ok)
        wall = sum(level.wall_times)
        throughput = len(totals) / wall if wall > 0 else 0.0
        avg_latency = statistics.mean(totals) if totals else None
        avg_ttft = statistics.mean(ttfts) if ttfts else None
        p95 = percentile(totals, 95) if totals else None
        print(
            f"  {level.concurrency:>11}  {len(level.results):>8}  {errors:>6}  "
            f"{_fmt_secs(avg_latency):>11}  {_fmt_secs(p95):>11}  {_fmt_secs(avg_ttft):>8}  "
            f"{throughput:>8.2f}/s"
        )
    print()


async def run_round(
    client: httpx.AsyncClient, url: str, questions: list[str], timeout: float
) -> tuple[list[RequestResult], float]:
    """Send all questions simultaneously and return results with wall time."""
    start = time.perf_counter()
    results = await asyncio.gather(*(send_chat(client, url, q, timeout) for q in questions))
    return list(results), time.perf_counter() - start


async def run_stress_test(
    url: str, levels: list[int], rounds: int, questions: list[str], timeout: float
) -> list[LevelStats]:
    """Run every concurrency level for the requested rounds.

    Stops early if the server cannot be reached, since further rounds would
    only repeat the same failure.
    """
    print(f"=== Stress Test: {url} ===\n")
    all_stats: list[LevelStats] = []
    offset = 0

    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"},
    ) as client:
        for concurrency in levels:
            stats = LevelStats(concurrency=concurrency, results=[], wall_times=[])
            all_stats.append(stats)
            for round_no in range(1, rounds + 1):
                print(f"--- Concurrency: {concurrency} | Round {round_no}/{rounds} ---\n")
                batch = pick_questions(questions, concurrency, offset)
                offset += concurrency
                results, wall = await run_round(client, url, batch, timeout)
                stats.results.extend(results)
                stats.wall_times.append(wall)
                print_round(results, wall)

                if any(r.status == "connection error" for r in results):
                    print(f"Error: cannot connect to {url}, aborting", file=sys.stderr)
                    return all_stats
    return all_stats


def _parse_levels(raw: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""
    try:
        levels = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"concurrency must be comma-separated integers, got '{raw}'")
    if not levels or any(n < 1 for n in levels):
        raise argparse.ArgumentTypeError(f"concurrency levels must be >= 1, got '{raw}'")
    return levels


def main() -> int:
    """Parse arguments, run the stress test, and return the exit code."""
    parser = argparse.ArgumentParser(description="Stress test the /chat SSE endpoint")
    parser.add_argument("questions", nargs="*", help="Custom questions (default: golden questions)")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL to test")
    parser.add_argument(
        "--concurrency", type=_parse_levels, default=[1, 2, 5], help="Comma-separated concurrency levels"
    )
    parser.add_argument("--rounds", type=int, default=1, help="Repeat each concurrency level N times")
    parser.add_argument("--timeout", type=float, default=120, help="Per-request timeout in seconds")
    args = parser.parse_args()

    if args.rounds < 1:
        print("Error: --rounds must be >= 1", file=sys.stderr)
        return 1
    if args.timeout <= 0:
        print("Error: --timeout must be > 0", file=sys.stderr)
        return 1

    questions = args.questions or load_golden_questions(
        str(Path(__file__).resolve().parent.parent / DEFAULT_QUESTIONS_PATH)
    )
    url = args.url.rstrip("/")

    stats = asyncio.run(run_stress_test(url, args.concurrency, args.rounds, questions, args.timeout))
    print_overall(stats)

    failed = sum(1 for level in stats for r in level.results if not r.ok)
    aborted = len(stats) < len(args.concurrency) or (
        stats and len(stats[-1].wall_times) < args.rounds
    )
    if failed or aborted:
        print(f"{failed} request(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
