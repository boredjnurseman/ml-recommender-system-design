from recommender_demo.models import Candidate


def score_candidate(
    candidate: Candidate,
    weights: dict[str, float],
) -> float:
    """Calculate the transparent policy score used for reranking.

    Args:
        candidate: Candidate and its normalised serving-time attributes.
        weights: Policy weights for relevance, novelty, long-tail exposure and
            repetition.

    Returns:
        The weighted candidate score; higher values rank first.
    """
    # Keep the proxies deliberately simple so policy effects remain inspectable.
    novelty = 0.0 if candidate.recently_seen else 1.0
    long_tail = 1.0 - candidate.popularity

    # Consumption is a stronger repetition signal than a prior recommendation.
    repeat = max(
        1.0 if candidate.recently_seen else 0.0,
        0.5 if candidate.recently_recommended else 0.0,
    )

    relevance_weight = weights["relevance"]
    novelty_weight = weights["novelty"]
    long_tail_weight = weights["long_tail"]
    repeat_penalty_weight = weights["repeat_penalty"]

    return (
        relevance_weight * candidate.relevance
        + novelty_weight * novelty
        + long_tail_weight * long_tail
        - repeat_penalty_weight * repeat
    )


def rerank_candidates(
    candidates: list[Candidate],
    weights: dict[str, float],
) -> list[Candidate]:
    """Return candidates ordered by decreasing policy score.

    Args:
        candidates: Region-eligible candidates to rerank.
        weights: Candidate score weights from the selected policy.

    Returns:
        A new list ordered by decreasing weighted score.
    """
    return sorted(
        candidates,
        key=lambda candidate: score_candidate(candidate, weights),
        reverse=True,
    )
