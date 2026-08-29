import hashlib


def assign_policy(
    user_id: str,
    baseline_version: str,
    experimental_version: str,
    experimental_fraction: float,
) -> str:
    """Assign a user to a baseline or experimental policy by stable hash bucket.

    Args:
        user_id: Stable pseudonymous identifier used for deterministic bucketing.
        baseline_version: Policy version for the control cohort.
        experimental_version: Policy version for the experimental cohort.
        experimental_fraction: Fraction of hash space assigned to the experiment.

    Returns:
        The baseline or experimental policy version.

    Raises:
        ValueError: If the experimental fraction lies outside [0, 1].
    """
    if not 0 <= experimental_fraction <= 1:
        raise ValueError("experimental_fraction must be between 0 and 1")

    # Keep each user's bucket stable without storing per-user state.
    digest = hashlib.sha256(user_id.encode()).hexdigest()
    integer = int(digest[:8], 16)
    bucket = integer / 2**32

    if bucket < experimental_fraction:
        return experimental_version

    return baseline_version
