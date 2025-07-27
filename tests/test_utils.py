from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from app.utils.config import load_config
from app.utils.prompt_loader import PromptLoader


class TestPromptLoader:

    def test_init_default_config_dir(self):
        loader = PromptLoader()
        assert loader.config_dir == Path("config")

    def test_init_custom_config_dir(self):
        loader = PromptLoader("custom_config")
        assert loader.config_dir == Path("custom_config")

    @patch("pathlib.Path.glob")
    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_load_prompts_single_file(self, mock_yaml_load, mock_file_open, mock_glob):
        mock_glob.return_value = [Path("config/test.yaml")]
        mock_yaml_load.return_value = {"test_prompt": "Hello world"}

        loader = PromptLoader()

        assert "test_prompt" in loader._prompts
        assert loader._prompts["test_prompt"] == "Hello world"

    @patch("pathlib.Path.glob")
    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_load_prompts_multiple_files(
        self, mock_yaml_load, mock_file_open, mock_glob
    ):
        mock_glob.return_value = [Path("config/file1.yaml"), Path("config/file2.yaml")]
        mock_yaml_load.side_effect = [
            {"prompt1": "First prompt"},
            {"prompt2": "Second prompt"},
        ]

        loader = PromptLoader()

        assert "prompt1" in loader._prompts
        assert "prompt2" in loader._prompts
        assert loader._prompts["prompt1"] == "First prompt"
        assert loader._prompts["prompt2"] == "Second prompt"

    @patch("pathlib.Path.glob")
    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_load_prompts_empty_file(self, mock_yaml_load, mock_file_open, mock_glob):
        mock_glob.return_value = [Path("config/empty.yaml")]
        mock_yaml_load.return_value = None

        loader = PromptLoader()

        assert loader._prompts == {}

    @patch("pathlib.Path.glob")
    @patch("builtins.open")
    def test_load_prompts_file_not_found_error(self, mock_file_open, mock_glob):
        mock_glob.return_value = [Path("config/missing.yaml")]
        mock_file_open.side_effect = FileNotFoundError("File not found")

        with pytest.raises(
            RuntimeError,
            match="Failed to load prompts from.*missing.yaml.*File not found",
        ):
            PromptLoader()

    @patch("pathlib.Path.glob")
    @patch("builtins.open")
    def test_load_prompts_permission_error(self, mock_file_open, mock_glob):
        mock_glob.return_value = [Path("config/restricted.yaml")]
        mock_file_open.side_effect = PermissionError("Permission denied")

        with pytest.raises(
            RuntimeError,
            match="Failed to load prompts from.*restricted.yaml.*Permission denied",
        ):
            PromptLoader()

    @patch("pathlib.Path.glob")
    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_load_prompts_yaml_error(self, mock_yaml_load, mock_file_open, mock_glob):
        mock_glob.return_value = [Path("config/invalid.yaml")]
        mock_yaml_load.side_effect = yaml.YAMLError("Invalid YAML")

        with pytest.raises(
            RuntimeError,
            match="Failed to load prompts from.*invalid.yaml.*Invalid YAML",
        ):
            PromptLoader()

    def test_get_prompt_existing_key(self):
        loader = PromptLoader()
        loader._prompts = {"test_key": "test_value"}

        result = loader.get_prompt("test_key")
        assert result == "test_value"

    def test_get_prompt_missing_key(self):
        loader = PromptLoader()
        loader._prompts = {}

        result = loader.get_prompt("missing_key")
        assert result == ""

    def test_get_formatted_prompt_with_kwargs(self):
        loader = PromptLoader()
        loader._prompts = {"greeting": "Hello {name}, welcome to {place}!"}

        result = loader.get_formatted_prompt("greeting", name="Alice", place="Python")
        assert result == "Hello Alice, welcome to Python!"

    def test_get_formatted_prompt_missing_key(self):
        loader = PromptLoader()
        loader._prompts = {}

        result = loader.get_formatted_prompt("missing", name="Alice")
        assert result == ""

    def test_get_formatted_prompt_no_placeholders(self):
        loader = PromptLoader()
        loader._prompts = {"simple": "Just a simple prompt"}

        result = loader.get_formatted_prompt("simple")
        assert result == "Just a simple prompt"

    def test_get_formatted_prompt_missing_placeholder(self):
        loader = PromptLoader()
        loader._prompts = {"template": "Hello {name}, welcome to {place}!"}

        with pytest.raises(
            ValueError, match="Failed to format prompt 'template'.*'place'"
        ):
            loader.get_formatted_prompt("template", name="Alice")

    def test_get_formatted_prompt_invalid_format(self):
        loader = PromptLoader()
        loader._prompts = {"bad_template": "Hello {name"}

        with pytest.raises(ValueError, match="Failed to format prompt 'bad_template'"):
            loader.get_formatted_prompt("bad_template", name="Alice")
