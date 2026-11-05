"""Claim-level groundedness checking for RAG answers.

Nothing here performs I/O: no LLM calls, database access, or HTTP requests.
The caller sends the prompt to the judge model and passes the judge's raw reply
to parse_groundedness_response(), so both functions can be tested with literals.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

GROUNDEDNESS_SYSTEM_PROMPT = (
    "You are an impartial fact-checker. Your job is to analyze an AI assistant's "
    "answer and determine which claims are supported by the provided context and "
    "which are not. Be precise and thorough."
)


def build_groundedness_prompt(
    response_text: str, chunks: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Build the messages for the groundedness judge.

    Args:
        response_text: The assistant's answer to check.
        chunks: Retrieved context chunks, each with "source_path" and "content" keys.

    Returns:
        Messages list ready for an LLMClient.chat() call.
    """
    numbered_chunks = "\n\n".join(
        f"[{i}] {c['source_path']}\n{c['content']}" for i, c in enumerate(chunks, 1)
    )

    user_prompt = f"""Analyze the following answer for factual grounding against the provided context chunks.

Context chunks:
{numbered_chunks}

Answer to check:
{response_text}

Instructions:
1. Decompose the answer into individual factual claims.
2. For each claim, determine if it is "supported" or "unsupported" by the context chunks.
3. For supported claims, quote the relevant evidence and name the source chunk number.
4. For unsupported claims, explain what evidence is missing.

Respond with ONLY valid JSON in this exact format:
{{
  "claims": [
    {{
      "claim": "the specific factual claim",
      "verdict": "supported",
      "evidence": "quoted text from the context that supports this claim",
      "chunk_index": 1
    }},
    {{
      "claim": "another claim",
      "verdict": "unsupported",
      "explanation": "what evidence is missing or why this is not grounded"
    }}
  ]
}}"""

    return [
        {"role": "system", "content": GROUNDEDNESS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


@dataclass
class Claim:
    """One factual claim extracted from a response.

    ``chunk_index`` is 1-based, matching the numbering in the judge prompt.
    ``source_path`` is resolved from it when it points at one of the supplied chunks.
    """

    claim: str
    verdict: str  # "supported" or "unsupported"
    evidence: str | None = None
    explanation: str | None = None
    chunk_index: int | None = None
    source_path: str | None = None


@dataclass
class GroundednessResult:
    """Structured result of a groundedness check.

    ``score`` is the fraction of claims judged supported, or 0.0 when there are no claims.
    When the judge reply cannot be parsed, ``parse_error`` says why and ``claims`` is empty.
    """

    claims: list[Claim] = field(default_factory=list)
    total_claims: int = 0
    supported: int = 0
    unsupported: int = 0
    score: float = 0.0
    raw_response: str = ""
    parse_error: str | None = None


# Match the outermost { ... } in the response, allowing surrounding text.
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _record_parse_failure(result: GroundednessResult, message: str) -> GroundednessResult:
    """Mark a result as unparseable, log the reason, and return it.

    The judge's reply is deliberately not logged: it can be long, and the caller
    already receives it as ``result.raw_response``.

    Args:
        result: The result being built for the current judge reply.
        message: Why the reply could not be parsed.

    Returns:
        The same result object, with ``parse_error`` set.
    """
    result.parse_error = message
    logger.warning("Groundedness judge response could not be parsed: %s", message)
    return result


def parse_groundedness_response(
    response_text: str, chunks: list[dict[str, str]]
) -> GroundednessResult:
    """Parse the judge's JSON response into a structured GroundednessResult.

    Handles common LLM output issues: markdown code fences, leading/trailing text,
    and malformed JSON. Falls back gracefully when parsing fails: this function never
    raises on bad judge output, and reports the problem in ``parse_error`` instead.

    Args:
        response_text: Raw text from the judge LLM.
        chunks: The original context chunks, used to resolve source_path from chunk_index.

    Returns:
        Parsed result with per-claim verdicts and summary statistics.
    """
    result = GroundednessResult(raw_response=response_text)

    # Strip markdown code fences if present
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)

    match = _JSON_BLOCK.search(cleaned)
    if not match:
        return _record_parse_failure(result, "No JSON object found in judge response")

    try:
        data = json.loads(match.group())
    except (json.JSONDecodeError, RecursionError) as exc:
        # json.loads raises RecursionError, not JSONDecodeError, on pathologically nested input.
        return _record_parse_failure(result, f"JSON parse error: {exc}")

    if not isinstance(data, dict) or "claims" not in data:
        return _record_parse_failure(result, "JSON missing 'claims' key")

    raw_claims = data["claims"]
    if not isinstance(raw_claims, list):
        return _record_parse_failure(result, "JSON 'claims' value is not a list")

    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        claim_text = item.get("claim", "")
        raw_verdict = item.get("verdict")
        verdict = raw_verdict.lower() if isinstance(raw_verdict, str) else "unsupported"
        if verdict not in ("supported", "unsupported"):
            verdict = "unsupported"

        chunk_idx = item.get("chunk_index")
        # bool is a subclass of int, so exclude it explicitly: True must not resolve to chunk 1.
        if isinstance(chunk_idx, bool) or not isinstance(chunk_idx, int):
            chunk_idx = None
        source_path = None
        # chunk_index is 1-based in the prompt
        if chunk_idx is not None and 1 <= chunk_idx <= len(chunks):
            source_path = chunks[chunk_idx - 1].get("source_path")

        result.claims.append(
            Claim(
                claim=claim_text,
                verdict=verdict,
                evidence=item.get("evidence"),
                explanation=item.get("explanation"),
                chunk_index=chunk_idx,
                source_path=source_path,
            )
        )

    result.total_claims = len(result.claims)
    result.supported = sum(1 for c in result.claims if c.verdict == "supported")
    result.unsupported = result.total_claims - result.supported
    result.score = result.supported / result.total_claims if result.total_claims > 0 else 0.0

    return result
