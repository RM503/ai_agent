# Utility function for loading prompts
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Optional

def load_prompts(
        name: str,
        category: str="system",
        template_vars: Optional[dict]=None,
        prompts_file: Path | str="prompts.yaml"
) -> dict[str, str]:
    with open(prompts_file, "r") as f:
        prompts = yaml.safe_load(f)["prompts"]

    try:
        prompt = prompts[category][name]
    except KeyError:
        raise KeyError(f"Prompt '{name}' not found under category '{category}'.")

    if template_vars:
        prompt = prompt.format(**template_vars)

    return prompt