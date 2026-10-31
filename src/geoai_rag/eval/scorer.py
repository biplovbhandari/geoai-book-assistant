"""Pure scoring helpers for the RAG evaluation pipeline.

Nothing here performs I/O: no HTTP calls, database access, or file reads.
The eval runner owns all I/O and passes plain values in, so every function
can be tested with literals.
"""

from __future__ import annotations

import re
from math import log2

_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator of question answering systems. "
    "Apply the rating scale you are given exactly, and reply with only the number."
)

_SIMILARITY_PROMPT = """\
You are evaluating an answer from a RAG system about an earth science textbook.

Question: {question}
Expected answer: {expected_answer}
Actual answer: {actual_answer}

Rate the actual answer on a scale of 0.0 to 1.0:
- 1.0: Fully addresses the question with accurate, complete information matching the expected answer
- 0.8: Addresses the question well with accurate information but misses minor details
- 0.5: Partially addresses the question or has notable gaps compared to the expected answer
- 0.2: Touches on the topic but largely misses the key points of the expected answer
- 0.0: Does not address the question or is mostly incorrect

Respond with only a number between 0.0 and 1.0."""

_FAITHFULNESS_PROMPT = """\
You are evaluating whether a RAG system's answer is faithful to its retrieved context.
The answer should only make claims that are supported by the context chunks.

Context chunks:
{chunks_text}

Answer:
{actual_answer}

Rate faithfulness on a scale of 0.0 to 1.0:
- 1.0: Every claim in the answer is supported by the context
- 0.8: Nearly all claims are supported, with only minor unsupported details
- 0.5: A mix of supported and unsupported claims
- 0.2: Mostly unsupported claims with some grounded information
- 0.0: The answer mostly contains information not found in the context

Respond with only a number between 0.0 and 1.0."""

# An optionally negative number such as "0.8", "1", "1.", ".5" or "-0.3".
# The minus sign is part of the match so that a negative reply clamps to 0.0
# instead of being read as its positive magnitude.
_SCORE_PATTERN = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)")


def retrieval_precision(retrieved_paths: list[str], expected_prefixes: list[str]) -> float:
    """Fraction of retrieved chunks whose source path starts with an expected source.

    Each retrieved chunk counts once, so several chunks from one file all count.
    Matching is a plain string prefix: "book/03" also matches "book/03_Segmentation/...".
    End a prefix with "/" to stop it at a directory boundary.

    Args:
        retrieved_paths: One source_path per retrieved chunk. Repeated paths are kept.
        expected_prefixes: Path prefixes of the sources a good answer should draw on.

    Returns:
        A value in [0.0, 1.0], or 0.0 when nothing was retrieved.
    """
    if not retrieved_paths:
        return 0.0
    prefixes = tuple(expected_prefixes)
    matching = sum(1 for path in retrieved_paths if path.startswith(prefixes))
    return matching / len(retrieved_paths)


def retrieval_recall(retrieved_paths: list[str], expected_prefixes: list[str]) -> float:
    """Fraction of expected sources that at least one retrieved chunk comes from.

    An expected source is found when any retrieved path starts with its prefix.
    Each expected source counts once, however many chunks match it.

    Args:
        retrieved_paths: One source_path per retrieved chunk.
        expected_prefixes: Path prefixes of the sources a good answer should draw on.

    Returns:
        A value in [0.0, 1.0], or 0.0 when no sources are expected.
    """
    if not expected_prefixes:
        return 0.0
    found = sum(
        any(path.startswith(prefix) for path in retrieved_paths) for prefix in expected_prefixes
    )
    return found / len(expected_prefixes)


def retrieval_ndcg(retrieved_paths: list[str], expected_prefixes: list[str]) -> float:
    """Rank-aware retrieval quality: relevant chunks near the top score higher.

    A chunk is relevant (gain 1) when its path starts with any expected prefix, else 0.
    DCG discounts each gain by log2(rank + 1), and NDCG divides by the DCG of the ideal
    ordering of the same chunks, where every relevant chunk comes first.

    Args:
        retrieved_paths: One source_path per retrieved chunk, in rank order.
        expected_prefixes: Path prefixes of the sources a good answer should draw on.

    Returns:
        A value in [0.0, 1.0]. Returns 1.0 when there is nothing to rank or nothing
        expected, since no ordering can be wrong. Returns 0.0 when sources are expected
        and chunks were retrieved, but none are relevant.
    """
    if not retrieved_paths or not expected_prefixes:
        return 1.0
    prefixes = tuple(expected_prefixes)
    gains = [1 if path.startswith(prefixes) else 0 for path in retrieved_paths]
    relevant_count = sum(gains)
    if relevant_count == 0:
        return 0.0
    dcg = sum(gain / log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, relevant_count + 1))
    return dcg / ideal_dcg


def _judge_messages(user_prompt: str) -> list[dict[str, str]]:
    """Pair a rendered judge prompt with the shared system message."""
    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_similarity_prompt(
    question: str,
    expected_answer: str,
    actual_answer: str,
) -> list[dict[str, str]]:
    """Build the judge messages that rate an answer against a reference answer.

    The inputs are inserted into the prompt verbatim.
    The judge is asked for a single 0.0 to 1.0 number, which parse_judge_score reads.

    Args:
        question: The question that was asked.
        expected_answer: Reference text describing what a good answer covers.
        actual_answer: The answer the RAG system produced.

    Returns:
        A system and a user message, in the shape LLMClient.chat() accepts.
    """
    user_prompt = _SIMILARITY_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer,
    )
    return _judge_messages(user_prompt)


def build_faithfulness_prompt(actual_answer: str, chunks_text: str) -> list[dict[str, str]]:
    """Build the judge messages that check an answer only claims what the chunks support.

    The inputs are inserted into the prompt verbatim.
    The judge is asked for a single 0.0 to 1.0 number, which parse_judge_score reads.

    Args:
        actual_answer: The answer the RAG system produced.
        chunks_text: The retrieved chunk text the answer was generated from.

    Returns:
        A system and a user message, in the shape LLMClient.chat() accepts.
    """
    user_prompt = _FAITHFULNESS_PROMPT.format(
        actual_answer=actual_answer,
        chunks_text=chunks_text,
    )
    return _judge_messages(user_prompt)


def parse_judge_score(response_text: str) -> float:
    """Extract the judge's score from its reply, clamped to [0.0, 1.0].

    The judge is told to answer with only a number, but models often add words around it,
    so the first number anywhere in the reply is used.
    A reply with no number scores 0.0, which keeps one bad judge reply from aborting a run.
    That 0.0 looks the same as a genuine lowest rating, so callers should keep the raw reply.

    Args:
        response_text: Raw text returned by the judge model.

    Returns:
        The score as a float between 0.0 and 1.0 inclusive.
    """
    match = _SCORE_PATTERN.search(response_text)
    if match is None:
        return 0.0
    return max(0.0, min(1.0, float(match.group())))
