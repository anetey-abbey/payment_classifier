import os

import yaml


def load_config():
    config = {}

    env = os.getenv("ENVIRONMENT", "dev")
    valid_envs = ["dev", "staging", "prod"]

    if env not in valid_envs:
        raise ValueError(
            f"Invalid environment '{env}'. Must be one of: {', '.join(valid_envs)}"
        )

    env_config_path = os.path.join(
        os.path.dirname(__file__), f"../../config/{env}.yaml"
    )

    # Load base config first
    base_config_path = os.path.join(os.path.dirname(__file__), "../../config/base.yaml")
    if os.path.exists(base_config_path):
        with open(base_config_path, "r") as f:
            base_config = yaml.safe_load(f)
            config.update(base_config)

    # Override with environment-specific config if it exists
    if os.path.exists(env_config_path):
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f)
            config.update(env_config)

    return config
