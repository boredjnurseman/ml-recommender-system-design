from recommender_demo.constraints import filter_candidates
from recommender_demo.models import Candidate, CandidateSnapshot
from recommender_demo.reranker import rerank_candidates
from recommender_demo.rollout import assign_policy


def serve_baseline(
    snapshot: CandidateSnapshot,
    region: str,
) -> list[Candidate]:
    """Return the base top-five list after applying regional constraints.

    Args:
        snapshot: Cached top-25 output from the offline recommender.
        region: Region code for the current request.

    Returns:
        Five eligible candidates in the base relevance order.

    Raises:
        ValueError: If fewer than five candidates remain after filtering.
    """
    top_25_candidates = snapshot.top_25_candidates

    filtered_candidates = filter_candidates(top_25_candidates, region)
    if len(filtered_candidates) < 5:
        raise ValueError(
            "Fewer than five eligible candidates available after filtering"
        )

    return filtered_candidates[:5]


def serve_recommendations(
    snapshot: CandidateSnapshot,
    region: str,
    baseline_policy: dict,
    experimental_policy: dict,
    experimental_fraction: float,
    reranker_enabled: bool = True,
) -> list[Candidate]:
    """Filter, policy-rerank and return five candidates for one request.

    Args:
        snapshot: Cached top-25 output from the offline recommender.
        region: Region code for the current request.
        baseline_policy: Versioned weights for the control cohort.
        experimental_policy: Versioned weights for the experimental cohort.
        experimental_fraction: Fraction of users assigned to the experiment.
        reranker_enabled: Whether to apply policy scoring after constraints.

    Returns:
        Five eligible candidates in policy-score or baseline order.

    Raises:
        ValueError: If fewer than five candidates remain after filtering or the
            experimental fraction is invalid.
    """
    top_25_candidates = snapshot.top_25_candidates

    filtered_candidates = filter_candidates(top_25_candidates, region)
    if len(filtered_candidates) < 5:
        raise ValueError(
            "Fewer than five eligible candidates available after filtering"
        )

    if not reranker_enabled:
        # Preserve a cheap fallback that does not depend on policy scoring.
        return filtered_candidates[:5]

    user_policy = assign_policy(
        user_id=snapshot.user_id,
        baseline_version=baseline_policy["version"],
        experimental_version=experimental_policy["version"],
        experimental_fraction=experimental_fraction,
    )

    if user_policy == experimental_policy["version"]:
        reranked_candidates = rerank_candidates(
            filtered_candidates,
            experimental_policy["weights"],
        )
    else:
        reranked_candidates = rerank_candidates(
            filtered_candidates,
            baseline_policy["weights"],
        )

    return reranked_candidates[:5]
