from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class HFGenerationConfig:
    max_new_tokens: int = 32
    temperature: float = 0.0
    do_sample: bool | None = None
    top_p: float | None = None
    pad_token_id: int | None = None


@dataclass(slots=True)
class HFModelBundle:
    tokenizer: Any
    model: Any


@dataclass(slots=True)
class HFGenerationResult:
    output_text: str
    full_text: str
    input_ids: list[int]
    generated_ids: list[int]


def load_hf_causal_lm(
    model_name_or_path: str,
    torch_dtype: str | Any = "auto",
    device_map: str | dict[str, Any] | None = None,
    trust_remote_code: bool = False,
) -> HFModelBundle:
    """Load a HuggingFace causal LM and tokenizer for reference generation."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return HFModelBundle(tokenizer=tokenizer, model=model)


def generate_from_input_ids(
    model: Any,
    tokenizer: Any,
    input_ids: list[int],
    config: HFGenerationConfig | None = None,
) -> HFGenerationResult:
    """Run normal HuggingFace model.generate() from fully assembled input ids."""

    import torch

    generation_config = config or HFGenerationConfig()
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_tensor)

    do_sample = generation_config.do_sample
    if do_sample is None:
        do_sample = generation_config.temperature > 0

    pad_token_id = generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": generation_config.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = generation_config.temperature
        if generation_config.top_p is not None:
            generate_kwargs["top_p"] = generation_config.top_p

    with torch.inference_mode():
        output_ids = model.generate(
            input_tensor,
            attention_mask=attention_mask,
            **generate_kwargs,
        )[0].tolist()

    generated_ids = output_ids[len(input_ids) :]
    return HFGenerationResult(
        output_text=tokenizer.decode(generated_ids, skip_special_tokens=True),
        full_text=tokenizer.decode(output_ids, skip_special_tokens=True),
        input_ids=list(input_ids),
        generated_ids=generated_ids,
    )
