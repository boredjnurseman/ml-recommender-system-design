from pathlib import Path

import yaml


def load_policy(path: str | Path) -> dict[str, object]:
    """Load and validate a versioned reranking policy from YAML.

    Args:
        path: Path to a YAML file containing a version and score weights.

    Returns:
        The parsed policy mapping.

    Raises:
        FileNotFoundError: If the policy file does not exist.
        ValueError: If the file is empty or omits required policy fields.
    """
    yaml_path = Path(path).expanduser().resolve()

    if not yaml_path.is_file():
        raise FileNotFoundError(f"Policy file does not exist at: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as policy_file:
        policy = yaml.safe_load(policy_file)

    if not isinstance(policy, dict):
        raise ValueError(f"File at {yaml_path} is empty or not a valid dictionary.")

    if "version" not in policy:
        raise ValueError(f"File at {yaml_path} does not contain 'version' key.")

    if "weights" not in policy:
        raise ValueError(f"File at {yaml_path} does not contain 'weights' key.")

    weights = policy["weights"]

    if not isinstance(weights, dict):
        raise ValueError(f"'weights' in {yaml_path} must be a dictionary.")

    required_weights = {
        "relevance",
        "novelty",
        "long_tail",
        "repeat_penalty",
    }
    missing_weights = required_weights - weights.keys()

    if missing_weights:
        raise ValueError(
            f"Invalid policy config. Missing required weights: {missing_weights}"
        )

    return policy
