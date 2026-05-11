from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contextflow.data.schema import Context, InputExample


SUPPORTED_QA_DATASET_KEYS = {"musique", "2wiki"}


def load_json_examples(path: str | Path) -> list[InputExample]:
    """Load a CacheBlend-style JSON list into InputExample objects."""

    data_path = Path(path)
    with data_path.open("r", encoding="utf-8") as file:
        raw_examples = json.load(file)

    if not isinstance(raw_examples, list):
        raise ValueError(f"Expected a JSON list in {data_path}, got {type(raw_examples).__name__}.")

    return parse_input_examples(raw_examples)


def load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a local JSON list/object or JSONL file as raw dictionaries."""

    data_path = Path(path)
    if data_path.suffix.lower() == ".jsonl":
        raw_examples = []
        with data_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                raw_example = json.loads(stripped)
                if not isinstance(raw_example, dict):
                    raise ValueError(
                        f"Expected JSONL line {line_number} in {data_path} to be an object, "
                        f"got {type(raw_example).__name__}."
                    )
                raw_examples.append(raw_example)
        return raw_examples

    with data_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if isinstance(raw_data, list):
        raw_examples = raw_data
    elif isinstance(raw_data, dict):
        raw_examples = extract_examples_from_json_object(raw_data, data_path)
    else:
        raise ValueError(
            f"Expected a JSON list or object in {data_path}, got {type(raw_data).__name__}."
        )

    for index, raw_example in enumerate(raw_examples):
        if not isinstance(raw_example, dict):
            raise ValueError(
                f"Expected example {index} in {data_path} to be an object, "
                f"got {type(raw_example).__name__}."
            )
    return raw_examples


def extract_examples_from_json_object(raw_data: dict[str, Any], data_path: Path) -> list[Any]:
    for key in ("data", "examples", "instances", "records"):
        value = raw_data.get(key)
        if isinstance(value, list):
            return value
    if "question" in raw_data:
        return [raw_data]
    raise ValueError(
        f"Expected JSON object in {data_path} to contain one of data/examples/instances/records "
        "or a single example with a question field."
    )


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


def load_qa_dataset_examples(dataset: str, path: str | Path) -> list[InputExample]:
    """Load local JSON/JSONL MuSiQue or 2Wiki examples into InputExample objects.

    Dataset loader scope is intentionally narrow: dataset/prepared input -> InputExample.
    It does not perform retrieval, reranking, chunking, evidence filtering, or model execution.
    The dataset key "2wiki" refers to 2WikiMultiHopQA / 2WikiMQA-style multi-hop QA data.
    """

    raw_examples = load_json_or_jsonl(path)
    return parse_qa_dataset_examples(dataset=dataset, raw_examples=raw_examples)


def parse_qa_dataset_examples(dataset: str, raw_examples: list[dict[str, Any]]) -> list[InputExample]:
    dataset_key = normalize_dataset_key(dataset)
    examples: list[InputExample] = []
    for index, raw_example in enumerate(raw_examples):
        if not isinstance(raw_example, dict):
            raise ValueError(
                f"Expected example {index} to be a JSON object, got {type(raw_example).__name__}."
            )
        if dataset_key == "musique":
            examples.append(parse_musique_example(raw_example))
        elif dataset_key == "2wiki":
            examples.append(parse_2wiki_example(raw_example))
        else:
            raise ValueError(f"Unsupported dataset key: {dataset!r}.")
    return examples


def normalize_dataset_key(dataset: str) -> str:
    dataset_key = dataset.strip().lower()
    aliases = {
        "musique": "musique",
        "2wiki": "2wiki",
        "2wikimultihopqa": "2wiki",
        "2wikimqa": "2wiki",
        "wikimqa": "2wiki",
    }
    if dataset_key not in aliases:
        raise ValueError(
            f"Unsupported dataset key {dataset!r}. Supported keys: {sorted(SUPPORTED_QA_DATASET_KEYS)}."
        )
    return aliases[dataset_key]


def parse_musique_example(raw_example: dict[str, Any]) -> InputExample:
    """Normalize MuSiQue raw/prepared layouts into CacheBlend-style InputExample."""

    if is_cacheblend_prepared_example(raw_example):
        return normalize_prepared_input_example(raw_example, dataset_key="musique")
    if "question" not in raw_example and "query" in raw_example:
        raise ValueError(
            "This looks like a CoRAG-style MuSiQue row with query/context_doc_ids, not raw "
            "MuSiQue paragraphs. It does not contain passage text for InputExample.ctxs. "
            "Use configs/datasets/musique.yaml with a MuSiQue source that exposes "
            "question/answer/paragraphs, then rerun scripts/prepare_dataset.py."
        )

    question = require_string_field(raw_example, "question")
    answers = normalize_answers(raw_example, keys=("answers", "answer"))
    raw_paragraphs = raw_example.get("paragraphs")
    if raw_paragraphs is None:
        raw_paragraphs = raw_example.get("ctxs", [])
    if not isinstance(raw_paragraphs, list):
        raise ValueError("MuSiQue example paragraphs must be a list.")

    ctxs: list[Context] = []
    paragraph_metadata: list[dict[str, Any]] = []
    for paragraph_index, paragraph in enumerate(raw_paragraphs):
        ctx, metadata = normalize_musique_paragraph(paragraph, paragraph_index)
        ctxs.append(ctx)
        if metadata:
            paragraph_metadata.append(metadata)

    return InputExample(
        question=question,
        answers=answers,
        ctxs=ctxs,
        example_id=normalize_example_id(raw_example, keys=("id", "qid")),
        metadata=build_dataset_metadata(
            raw_example,
            dataset_key="musique",
            known_keys={"id", "qid", "question", "answer", "answers", "paragraphs", "ctxs"},
            context_metadata=paragraph_metadata,
        ),
    )


def parse_2wiki_example(raw_example: dict[str, Any]) -> InputExample:
    """Normalize 2WikiMultiHopQA / 2WikiMQA-style layouts into InputExample."""

    if is_cacheblend_prepared_example(raw_example):
        return normalize_prepared_input_example(raw_example, dataset_key="2wiki")

    question = require_string_field(raw_example, "question")
    answers = normalize_answers(raw_example, keys=("answers", "answer", "golden_answers"))
    raw_contexts = raw_example.get("context")
    if raw_contexts is None and isinstance(raw_example.get("metadata"), dict):
        raw_contexts = raw_example["metadata"].get("context")
    if raw_contexts is None:
        raw_contexts = raw_example.get("ctxs", [])
    ctxs = normalize_2wiki_contexts(raw_contexts)

    return InputExample(
        question=question,
        answers=answers,
        ctxs=ctxs,
        example_id=normalize_example_id(raw_example, keys=("id", "_id")),
        metadata=build_dataset_metadata(
            raw_example,
            dataset_key="2wiki",
            known_keys={
                "id",
                "_id",
                "question",
                "answer",
                "answers",
                "golden_answers",
                "context",
                "ctxs",
            },
        ),
    )


def is_cacheblend_prepared_example(raw_example: dict[str, Any]) -> bool:
    return "question" in raw_example and (
        isinstance(raw_example.get("ctxs"), list)
        or isinstance(raw_example.get("contexts"), list)
    )


def normalize_prepared_input_example(raw_example: dict[str, Any], dataset_key: str) -> InputExample:
    question = require_string_field(raw_example, "question")
    answers = normalize_answers(raw_example, keys=("answers", "answer", "golden_answers"))
    raw_contexts = raw_example.get("ctxs")
    if raw_contexts is None:
        raw_contexts = raw_example.get("contexts", [])
    if not isinstance(raw_contexts, list):
        raise ValueError("Prepared input ctxs/contexts must be a list.")

    ctxs = [normalize_prepared_context(ctx, index) for index, ctx in enumerate(raw_contexts)]
    return InputExample(
        question=question,
        answers=answers,
        ctxs=ctxs,
        example_id=normalize_example_id(raw_example, keys=("id", "example_id", "qid", "_id")),
        metadata=build_dataset_metadata(
            raw_example,
            dataset_key=dataset_key,
            known_keys={
                "id",
                "example_id",
                "qid",
                "_id",
                "question",
                "answer",
                "answers",
                "golden_answers",
                "ctxs",
                "contexts",
            },
        ),
    )


def normalize_prepared_context(raw_context: Any, context_index: int) -> Context:
    if not isinstance(raw_context, dict):
        raise ValueError(
            f"Prepared context {context_index} must be an object, got {type(raw_context).__name__}."
        )
    title = normalize_optional_string(raw_context.get("title", ""))
    text = normalize_optional_string(raw_context.get("text", raw_context.get("paragraph_text", "")))
    return Context(title=title, text=text)


def normalize_musique_paragraph(
    raw_paragraph: Any,
    paragraph_index: int,
) -> tuple[Context, dict[str, Any]]:
    if not isinstance(raw_paragraph, dict):
        raise ValueError(
            f"MuSiQue paragraph {paragraph_index} must be an object, "
            f"got {type(raw_paragraph).__name__}."
        )

    title = normalize_optional_string(
        raw_paragraph.get(
            "title",
            raw_paragraph.get("paragraph_id", raw_paragraph.get("id", "")),
        )
    )
    text = normalize_optional_string(raw_paragraph.get("paragraph_text", raw_paragraph.get("text", "")))
    metadata = {
        key: raw_paragraph[key]
        for key in ("is_supporting", "paragraph_id", "id")
        if key in raw_paragraph
    }
    return Context(title=title, text=text), metadata


def normalize_2wiki_context(raw_context: Any, context_index: int) -> Context:
    if isinstance(raw_context, dict):
        return normalize_prepared_context(raw_context, context_index)

    if isinstance(raw_context, (list, tuple)) and len(raw_context) >= 2:
        title = normalize_optional_string(raw_context[0])
        sentences = raw_context[1]
        if isinstance(sentences, list):
            text = " ".join(str(sentence) for sentence in sentences)
        else:
            text = normalize_optional_string(sentences)
        return Context(title=title, text=text)

    raise ValueError(
        f"2wiki context {context_index} must be an object or [title, sentences], "
        f"got {type(raw_context).__name__}."
    )


def normalize_2wiki_contexts(raw_contexts: Any) -> list[Context]:
    if isinstance(raw_contexts, dict):
        titles = raw_contexts.get("title")
        sentences = raw_contexts.get("sentences")
        if sentences is None:
            sentences = raw_contexts.get("content")
        if not isinstance(titles, list) or not isinstance(sentences, list):
            raise ValueError(
                "2wiki context dict must contain title list and sentences/content list."
            )
        if len(titles) != len(sentences):
            raise ValueError(
                "2wiki context title and sentences/content lists must have the same length."
            )
        return [
            normalize_2wiki_context([title, context_sentences], context_index)
            for context_index, (title, context_sentences) in enumerate(zip(titles, sentences))
        ]

    if isinstance(raw_contexts, list):
        return [
            normalize_2wiki_context(context, context_index)
            for context_index, context in enumerate(raw_contexts)
        ]

    raise ValueError("2wiki example context/ctxs must be a list or Hotpot-style context dict.")


def normalize_answers(raw_example: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    for key in keys:
        if key not in raw_example:
            continue
        raw_answers = raw_example[key]
        if raw_answers is None:
            return []
        if isinstance(raw_answers, list):
            return [str(answer) for answer in raw_answers]
        return [str(raw_answers)]
    return []


def normalize_example_id(raw_example: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        raw_id = raw_example.get(key)
        if raw_id is not None:
            return str(raw_id)
    return None


def normalize_optional_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def require_string_field(raw_example: dict[str, Any], key: str) -> str:
    if key not in raw_example:
        raise ValueError(f"Expected example to contain {key!r}.")
    return str(raw_example[key])


def build_dataset_metadata(
    raw_example: dict[str, Any],
    dataset_key: str,
    known_keys: set[str],
    context_metadata: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metadata = {
        "dataset": dataset_key,
        **{key: value for key, value in raw_example.items() if key not in known_keys},
    }
    if context_metadata:
        metadata["context_metadata"] = context_metadata
    return metadata
