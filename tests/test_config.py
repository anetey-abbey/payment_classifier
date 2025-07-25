from app.utils.config import load_config


def test_load_config_basic():
    config = load_config()
    assert "place_holder_config" in config
    assert config["place_holder_config"] == "Place holder config test success."
