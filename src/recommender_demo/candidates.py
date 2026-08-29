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
	key = f"{user_id}|{model_version}|{candidate_refresh_id}"
	digest = hashlib.sha256(key.encode()).hexdigest()

	return int(digest[:8], 16)


def generate_region_subset(
	regions: list[str],
	rng: random.Random,
) -> list[str]:
	if not regions:
		raise ValueError("No regions specified")

	k = rng.randint(0, len(regions))
	return rng.sample(regions, k)

def generate_candidate(
    item_id: int,
    rng: random.Random,
) -> Candidate:
	relevance = rng.random()
	genre = rng.choice(GENRES)
	popularity = rng.random()
	recently_seen = rng.choice([True, False])
	recently_recommended = rng.choice([True, False])
	regional_availability = generate_region_subset(REGIONS, rng)

	candidate = Candidate(
		item_id=item_id,
		relevance=relevance,
		genre=genre,
		popularity=popularity,
		recently_seen=recently_seen,
		recently_recommended=recently_recommended,
		regional_availability=regional_availability,
	)

	return candidate

def get_candidate_snapshot(
	user_id: str,
	model_version: str,
	candidate_refresh_id: str,
) -> CandidateSnapshot:
	seed = make_seed(user_id, model_version, candidate_refresh_id)
	rng = random.Random(seed)

	candidates = [generate_candidate(item_id=i, rng=rng) for i in range(1, 26)]
	candidates.sort(key=lambda c: c.relevance, reverse=True)

	candidate_snapshot = CandidateSnapshot(
		user_id=user_id,
		model_version=model_version,
		candidate_refresh_id=candidate_refresh_id,
		top_25_candidates=candidates
	)

	return candidate_snapshot