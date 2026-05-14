from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from contextflow.data import load_json_or_jsonl, normalize_dataset_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local JSONL datasets from configured sources."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a dataset YAML config, for example configs/datasets/musique.yaml.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of examples to write. Overrides sample_size in config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the configured local output path if it already exists.",
    )
    return parser.parse_args()


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "scripts/prepare_dataset.py requires PyYAML. Install dataset setup extras with "
            '`pip install -e ".[datasets]"` or install `pyyaml` manually.'
        ) from error

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML config {config_path} to contain a mapping.")
    return config


def resolve_limit(config: dict[str, Any], cli_limit: int | None) -> int | None:
    if cli_limit is not None:
        if cli_limit <= 0:
            raise ValueError("--limit must be positive when provided.")
        return cli_limit
    sample_size = config.get("sample_size")
    if sample_size is None:
        return None
    sample_size = int(sample_size)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive when provided.")
    return sample_size


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected config field {key!r} to be a mapping.")
    return value


def load_records_from_config(config: dict[str, Any], limit: int | None) -> Iterable[dict[str, Any]]:
    dataset_key = normalize_dataset_key(str(config["dataset_key"]))
    source = require_mapping(config, "source")
    source_type = str(source.get("type", "")).lower()

    if source_type == "huggingface":
        return load_huggingface_records(source, limit=limit)
    if source_type == "local":
        source_path = source.get("path")
        if source_path is None:
            raise ValueError("local source requires source.path.")
        records = load_json_or_jsonl(source_path)
        return iter_limited_records(records, limit=limit)

    raise ValueError(
        f"Unsupported source.type={source_type!r} for dataset {dataset_key!r}. "
        "Supported source types: huggingface, local."
    )


def load_huggingface_records(source: dict[str, Any], limit: int | None) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Preparing HuggingFace datasets requires the `datasets` package. Install with "
            '`pip install -e ".[datasets]"` or `pip install datasets`.'
        ) from error

    dataset_name = source.get("name")
    if dataset_name is None:
        raise ValueError("huggingface source requires source.name.")
    split = source.get("split")
    if split is None:
        raise ValueError("huggingface source requires source.split.")

    load_kwargs: dict[str, Any] = {}
    if source.get("config") is not None:
        load_kwargs["name"] = source["config"]
    if source.get("data_dir") is not None:
        load_kwargs["data_dir"] = source["data_dir"]
    if source.get("revision") is not None:
        load_kwargs["revision"] = source["revision"]
    if source.get("trust_remote_code") is not None:
        load_kwargs["trust_remote_code"] = bool(source["trust_remote_code"])

    if source.get("streaming") is not None:
        load_kwargs["streaming"] = bool(source["streaming"])

    dataset = load_dataset(str(dataset_name), split=str(split), **load_kwargs)
    return iter_limited_records(dataset, limit=limit)


def iter_limited_records(records: Iterable[Any], limit: int | None) -> Iterable[dict[str, Any]]:
    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            break
        if not isinstance(record, dict):
            try:
                record = dict(record)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Expected record {index} to be dict-like, got {type(record).__name__}."
                ) from error
        yield record


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_jsonl(records: Iterable[dict[str, Any]], output_path: str | Path, overwrite: bool) -> int:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(to_jsonable(record), ensure_ascii=False))
            file.write("\n")
            count += 1
    return count


def resolve_output_path(config: dict[str, Any]) -> str:
    output = require_mapping(config, "output")
    output_format = str(output.get("format", "jsonl")).lower()
    if output_format != "jsonl":
        raise ValueError(f"Only output.format=jsonl is currently supported, got {output_format!r}.")
    local_path = output.get("local_path")
    if local_path is None:
        raise ValueError("output.local_path is required.")
    return str(local_path)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    dataset_key = normalize_dataset_key(str(config["dataset_key"]))
    limit = resolve_limit(config, args.limit)
    output_path = resolve_output_path(config)
    records = load_records_from_config(config, limit=limit)
    count = write_jsonl(records, output_path=output_path, overwrite=args.overwrite)

    print(f"dataset_key: {dataset_key}")
    print(f"output_path: {output_path}")
    print(f"records_written: {count}")
    print(f"limit: {limit}")


if __name__ == "__main__":
    main()
