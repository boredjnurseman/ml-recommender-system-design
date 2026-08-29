from recommender_demo.candidates import get_candidate_snapshot


def test_candidate_snapshot_is_deterministic():
    snapshot1 = get_candidate_snapshot("2026A", "0.10", "0319-91")
    snapshot2 = get_candidate_snapshot("2026A", "0.10", "0319-91")

    assert snapshot1 == snapshot2


def test_candidate_snapshot_changes_with_refresh_id():
    snapshot1 = get_candidate_snapshot("2026A", "0.10", "0319-91")
    snapshot2 = get_candidate_snapshot("2026A", "0.10", "0319-92")

    assert snapshot1 != snapshot2
