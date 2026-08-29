from recommender_demo.models import Candidate
from recommender_demo.candidates import REGIONS

def filter_candidates(
    candidates: list[Candidate],
    region: str,
) -> list[Candidate]:
    if region not in REGIONS:
        raise ValueError(f"region {region} is not a valid region")

    filtered_candidates = [
        candidate
        for candidate in candidates
        if region in candidate.regional_availability
    ]
    return filtered_candidates



