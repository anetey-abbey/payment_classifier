import argparse

import yaml


def load_config(config_path=None):

    config = {}

    if config_path:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)

    # Placeholder config
    config["place_holder_config"] = "Place holder config test success."

    return config


def get_args():
    parser = argparse.ArgumentParser(description="Payment Classification Service")

    parser.add_argument("--config", type=str, help="Path to config file")

    return parser.parse_args()
