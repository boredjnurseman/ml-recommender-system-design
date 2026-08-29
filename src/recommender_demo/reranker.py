from recommender_demo.models import Candidate


def score_candidate(
    candidate: Candidate,
    weights: dict[str, float],
) -> float:
	# calculated candidate-level metrics
	novelty = 0.0 if candidate.recently_seen else 1.0

	long_tail = 1.0 - candidate.popularity

	repeat = max(
		1.0 if candidate.recently_seen else 0.0,
		0.5 if candidate.recently_recommended else 0.0,
	)

	# weights from yaml policy
	relevance_weight = weights['relevance']
	novelty_weight = weights['novelty']
	long_tail_weight = weights['long_tail']
	repeat_penalty_weight = weights['repeat_penalty']

	return (
		relevance_weight * candidate.relevance + novelty_weight * novelty
		+ long_tail_weight * long_tail - repeat_penalty_weight * repeat
	)

def rerank_candidates(
		candidates: list[Candidate],
		weights: dict[str, float],
) -> list[Candidate]:
	return sorted(
		candidates,
		key=lambda candidate: score_candidate(candidate, weights),
		reverse=True,
	)