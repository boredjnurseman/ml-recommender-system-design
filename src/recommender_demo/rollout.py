import hashlib
from recommender_demo.models import CandidateSnapshot
from pathlib import Path

ROOT = Path(__file__).parent.parent.absolute()
conservative_policy = "policy-v1.yaml"
experimental_policy = "policy-v2.yaml"


def assign_policy(
    user_id: str,
    baseline_version: str,
    experimental_version: str,
    experimental_fraction: float,
) -> str:
    if not 0 <= experimental_fraction <= 1:
        raise ValueError("experimental_fraction must be between 0 and 1")

    digest = hashlib.sha256(user_id.encode()).hexdigest()
    integer = int(digest[:8], 16)
    bucket = integer / int("ffffffff", 16)

    if bucket < experimental_fraction:
        return experimental_version

    return baseline_version

