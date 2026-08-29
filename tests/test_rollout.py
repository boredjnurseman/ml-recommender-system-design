from unittest.mock import patch

import pytest

from recommender_demo.rollout import assign_policy


def test_same_user_id_gets_same_policy():
    policy1 = assign_policy("67", "conservative", "discovery", 0.1)
    policy2 = assign_policy("67", "conservative", "discovery", 0.1)

    assert policy1 == policy2


def test_invalid_experimental_fraction_raises_exception():
    with pytest.raises(ValueError):
        assign_policy("67", "conservative", "discovery", 1.1)

    with pytest.raises(ValueError):
        assign_policy("67", "conservative", "discovery", -0.1)


def test_fraction_boundaries_assign_expected_policy():
    policy1 = assign_policy("67", "conservative", "discovery", 0.0)
    policy2 = assign_policy("68", "conservative", "discovery", 1.0)

    assert policy1 == "conservative"
    assert policy2 == "discovery"


def test_full_fraction_includes_maximum_hash_bucket():
    maximum_digest = "f" * 64

    with patch("recommender_demo.rollout.hashlib.sha256") as sha256:
        sha256.return_value.hexdigest.return_value = maximum_digest
        policy = assign_policy(
            "boundary-user",
            "conservative",
            "discovery",
            1.0,
        )

    assert policy == "discovery"
