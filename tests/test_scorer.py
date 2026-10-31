import pytest

from geoai_rag.eval.scorer import (
    build_faithfulness_prompt,
    build_similarity_prompt,
    retrieval_ndcg,
)

_RELEVANT = "book/03_Semantic_Segmentation/"
_HIT = f"{_RELEVANT}index.qmd"
_MISS = "book/11_Ethics/index.qmd"


def test_ndcg_is_one_when_relevant_chunks_rank_first():
    """Ideal ordering scores 1.0 regardless of how many irrelevant chunks follow."""
    assert retrieval_ndcg([_HIT, _HIT, _MISS, _MISS], [_RELEVANT]) == pytest.approx(1.0)


def test_ndcg_penalises_relevant_chunks_ranked_last():
    """A single relevant chunk at rank 4 scores 1/log2(5), far below a rank-1 hit."""
    score = retrieval_ndcg([_MISS, _MISS, _MISS, _HIT], [_RELEVANT])

    assert score == pytest.approx(0.4307, abs=1e-3)


def test_ndcg_of_mixed_ranking_lies_between_best_and_worst():
    """Moving the relevant chunk from last to second improves the score but not to 1.0."""
    worst = retrieval_ndcg([_MISS, _MISS, _MISS, _HIT], [_RELEVANT])
    mixed = retrieval_ndcg([_MISS, _HIT, _MISS, _MISS], [_RELEVANT])
    best = retrieval_ndcg([_HIT, _MISS, _MISS, _MISS], [_RELEVANT])

    assert worst < mixed < best, "rank position must change the score"


@pytest.mark.parametrize(
    ("retrieved", "expected"),
    [
        pytest.param([], [_RELEVANT], id="nothing-retrieved"),
        pytest.param([_HIT, _MISS], [], id="nothing-expected"),
    ],
)
def test_ndcg_is_one_when_there_is_nothing_to_rank(retrieved, expected):
    """With no retrieved chunks or no expected sources, no ordering can be wrong."""
    assert retrieval_ndcg(retrieved, expected) == 1.0


def test_ndcg_is_zero_when_no_retrieved_chunk_is_relevant():
    """Expected sources exist and chunks came back, but none matched."""
    assert retrieval_ndcg([_MISS, _MISS], [_RELEVANT]) == 0.0


@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param(build_similarity_prompt("q", "expected", "actual"), id="similarity"),
        pytest.param(build_faithfulness_prompt("answer", "chunks"), id="faithfulness"),
    ],
)
def test_judge_prompts_use_five_point_scale(prompt):
    """Both judge prompts offer the intermediate 0.8 and 0.2 ratings."""
    user_content = prompt[1]["content"]

    assert "0.8:" in user_content
    assert "0.2:" in user_content
