from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class HFModelBundle:
    tokenizer: Any
    model: Any


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
