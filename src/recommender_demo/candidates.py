import hashlib
import random

from recommender_demo.models import Candidate, CandidateSnapshot

GENRES = [
    "drama",
    "comedy",
    "sci-fi",
    "thriller",
    "documentary",
]

REGIONS = ["UK", "US", "FR", "DE", "PH"]


def make_seed(
    user_id: str,
    model_version: str,
    candidate_refresh_id: str,
) -> int:
    """Derive a repeatable random seed for one versioned candidate snapshot.

    Args:
        user_id: Stable pseudonymous user identifier.
        model_version: Version of the base recommender.
        candidate_refresh_id: Identifier for the offline candidate refresh.

    Returns:
        A 32-bit integer derived from the snapshot identity.
    """
    key = f"{user_id}|{model_version}|{candidate_refresh_id}"
    digest = hashlib.sha256(key.encode()).hexdigest()

    return int(digest[:8], 16)


def generate_region_subset(
    regions: list[str],
    rng: random.Random,
) -> list[str]:
    """Sample the regions in which one synthetic candidate is available.

    Args:
        regions: Region codes available to the generator.
        rng: Seeded random number generator for repeatable sampling.

    Returns:
        A possibly empty subset of the supplied region codes.

    Raises:
        ValueError: If no region codes are supplied.
    """
    if not regions:
        raise ValueError("No regions specified")

    subset_size = rng.randint(0, len(regions))
    return rng.sample(regions, subset_size)


def generate_candidate(
    item_id: int,
    rng: random.Random,
) -> Candidate:
    """Generate one synthetic candidate for exercising the serving path.

    Args:
        item_id: Stable identifier to assign to the candidate.
        rng: Seeded random number generator for repeatable attributes.

    Returns:
        A candidate with synthetic relevance, history and availability values.
    """
    return Candidate(
        item_id=item_id,
        relevance=rng.random(),
        genre=rng.choice(GENRES),
        popularity=rng.random(),
        recently_seen=rng.choice([True, False]),
        recently_recommended=rng.choice([True, False]),
        regional_availability=generate_region_subset(REGIONS, rng),
    )


def get_candidate_snapshot(
    user_id: str,
    model_version: str,
    candidate_refresh_id: str,
) -> CandidateSnapshot:
    """Build the deterministic top-25 stand-in for an offline model output.

    Args:
        user_id: Stable pseudonymous user identifier.
        model_version: Version of the base recommender being represented.
        candidate_refresh_id: Identifier for the offline candidate refresh.

    Returns:
        A versioned snapshot containing 25 candidates in relevance order.
    """
    seed = make_seed(user_id, model_version, candidate_refresh_id)
    rng = random.Random(seed)

    candidates = [generate_candidate(item_id=i, rng=rng) for i in range(1, 26)]
    candidates.sort(key=lambda candidate: candidate.relevance, reverse=True)

    return CandidateSnapshot(
        user_id=user_id,
        model_version=model_version,
        candidate_refresh_id=candidate_refresh_id,
        top_25_candidates=candidates,
    )
