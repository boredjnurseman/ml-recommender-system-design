from recommender_demo.candidates import get_candidate_snapshot
from recommender_demo.serving import serve_baseline, serve_recommendations
from recommender_demo.config import load_policy
from recommender_demo.constraints import filter_candidates
from recommender_demo.reranker import rerank_candidates
from pathlib import Path
from dataclasses import replace
import pytest

snapshot1 = get_candidate_snapshot("2026A",
                                   "0.10",
                                   "0319-91")
snapshot2 = get_candidate_snapshot("2026A",
                                   "0.10",
                                   "0319-91")

def test_exactly_5_items_served():
    first_top_5 = serve_baseline(snapshot1, "UK")
    second_top_5 = serve_baseline(snapshot2, "UK")

    assert len(first_top_5) == 5
    assert len(second_top_5) == 5

def test_all_items_served_available_in_region():
    first_top_5 = serve_baseline(snapshot1, "UK")
    second_top_5 = serve_baseline(snapshot2, "PH")

    assert all("UK" in candidate.regional_availability for candidate in first_top_5)
    assert all("PH" in candidate.regional_availability for candidate in second_top_5)

def test_fewer_than_5_eligible_candidates_raises_error():
    snapshot_copy = replace(snapshot1)
    snapshot_copy.top_25_candidates = snapshot1.top_25_candidates[:4]

    with pytest.raises(ValueError):
        serve_baseline(snapshot_copy, "UK")

ROOT = Path(__file__).parent.parent
baseline_policy = load_policy(f"{ROOT}/config/policy-v1.yaml")
experimental_policy = load_policy(f"{ROOT}/config/policy-v2.yaml")

def test_reranker_disabled_falls_back_to_baseline():
    baseline = serve_baseline(snapshot1,
                              "UK")
    reranked = serve_recommendations(snapshot1,
                                     "UK",
                                     baseline_policy,
                                     experimental_policy,
                                     0.1,
                                     False)

    assert baseline == reranked

def test_zero_experimental_fraction_uses_baseline_policy():
    eligible = filter_candidates(
        snapshot1.top_25_candidates,
        "UK",
    )

    expected = rerank_candidates(
        eligible,
        baseline_policy["weights"],
    )[:5]

    actual = serve_recommendations(
        snapshot1,
        "UK",
        baseline_policy,
        experimental_policy,
        experimental_fraction=0.0,
    )

    assert actual == expected


def test_full_experimental_fraction_uses_experimental_policy():
    eligible = filter_candidates(
        snapshot1.top_25_candidates,
        "UK",
    )

    expected = rerank_candidates(
        eligible,
        experimental_policy["weights"],
    )[:5]

    actual = serve_recommendations(
        snapshot1,
        "UK",
        baseline_policy,
        experimental_policy,
        experimental_fraction=1.0,
    )

    assert actual == expected


def test_all_serving_paths_return_exactly_five_items():
    fallback = serve_recommendations(
        snapshot1,
        "UK",
        baseline_policy,
        experimental_policy,
        experimental_fraction=0.1,
        reranker_enabled=False,
    )

    baseline_reranked = serve_recommendations(
        snapshot1,
        "UK",
        baseline_policy,
        experimental_policy,
        experimental_fraction=0.0,
        reranker_enabled=True,
    )

    experimental_reranked = serve_recommendations(
        snapshot1,
        "UK",
        baseline_policy,
        experimental_policy,
        experimental_fraction=1.0,
        reranker_enabled=True,
    )

    assert len(fallback) == 5
    assert len(baseline_reranked) == 5
    assert len(experimental_reranked) == 5

def test_policy_configs_have_required_structure():
    for policy in (baseline_policy, experimental_policy):
        assert "version" in policy
        assert "weights" in policy

        assert set(policy["weights"]) >= {
            "relevance",
            "novelty",
            "long_tail",
            "repeat_penalty",
        }