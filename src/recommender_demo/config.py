import yaml
from pathlib import Path


def load_policy(path: str | Path) -> dict:
	yaml_path = Path(path).expanduser().resolve()

	if not yaml_path.is_file():
		raise FileNotFoundError(f"Policy file does not exist at: {yaml_path}")

	with yaml_path.open("r", encoding="utf-8") as f:
		policy = yaml.safe_load(f)

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
		raise ValueError(f"Invalid policy config. Missing required weights: {missing_weights}")

	return policy

