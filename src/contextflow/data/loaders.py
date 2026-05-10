from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contextflow.data.schema import InputExample


def load_json_examples(path: str | Path) -> list[InputExample]:
    """Load a CacheBlend-style JSON list into InputExample objects."""

    data_path = Path(path)
    with data_path.open("r", encoding="utf-8") as file:
        raw_examples = json.load(file)

    if not isinstance(raw_examples, list):
        raise ValueError(f"Expected a JSON list in {data_path}, got {type(raw_examples).__name__}.")

    return parse_input_examples(raw_examples)


def parse_input_examples(raw_examples: list[dict[str, Any]]) -> list[InputExample]:
    """Parse raw QA dictionaries with question, answers, and ctxs fields."""

    examples: list[InputExample] = []
    for index, raw_example in enumerate(raw_examples):
        if not isinstance(raw_example, dict):
            raise ValueError(
                f"Expected example {index} to be a JSON object, got {type(raw_example).__name__}."
            )
        examples.append(InputExample.from_dict(raw_example))
    return examples
