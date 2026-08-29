from recommender_demo.candidates import REGIONS
from recommender_demo.models import Candidate


def filter_candidates(
    candidates: list[Candidate],
    region: str,
) -> list[Candidate]:
    """Keep candidates that may legally be served in the requested region.

    Args:
        candidates: Ranked candidates produced by the offline recommender.
        region: Region code for the current request.

    Returns:
        Eligible candidates in their original order.

    Raises:
        ValueError: If the region is not supported by the demonstrator.
    """
    if region not in REGIONS:
        raise ValueError(f"region {region} is not a valid region")

    return [
        candidate
        for candidate in candidates
        if region in candidate.regional_availability
    ]
