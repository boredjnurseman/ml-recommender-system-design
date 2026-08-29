from recommender_demo.reranker import score_candidate, rerank_candidates
from recommender_demo.config import load_policy
from recommender_demo.models import Candidate
from pathlib import Path

candidate1 = Candidate(
				item_id=1,
				genre="Drama",
				relevance=0.75,
				popularity=0.89,
				recently_seen=True,
				recently_recommended=False
			)

candidate2 = Candidate(
				item_id=2,
				genre="Comedy",
				relevance=0.6,
				popularity=0.48,
				recently_seen=False,
				recently_recommended=False
			)

# relevant and popular candidate goes first
candidate_pair = [candidate1, candidate2]

ROOT = Path(__file__).parent.parent

conservative_policy = load_policy(f'{ROOT}/config/policy-v1.yaml')
discovery_policy = load_policy(f'{ROOT}/config/policy-v2.yaml')

def test_candidate_reranker():
	conservative_ranking = rerank_candidates(
		candidate_pair,
		conservative_policy["weights"],
	)

	assert conservative_ranking == candidate_pair

	discovery_ranking = rerank_candidates(
		candidate_pair,
		discovery_policy["weights"],
	)

	assert discovery_ranking != candidate_pair