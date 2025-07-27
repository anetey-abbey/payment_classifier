from pathlib import Path
from typing import Dict

import yaml


class PromptLoader:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        prompts: Dict[str, str] = {}

        for file_path in sorted(self.config_dir.glob("*.yaml")):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    file_prompts = yaml.safe_load(file) or {}
                    prompts.update(file_prompts)
            except (FileNotFoundError, PermissionError, yaml.YAMLError) as e:
                raise RuntimeError(f"Failed to load prompts from {file_path}: {e}")

        return prompts

    def get_prompt(self, key: str) -> str:
        return self._prompts.get(key, "")

    def get_formatted_prompt(self, key: str, **kwargs) -> str:
        template = self.get_prompt(key)
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Failed to format prompt '{key}': {e}")
