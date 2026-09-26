"""Build fixed NEG and POS prototypes from symmetric CLIP text prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as functional


CLASS_NAMES = ("NEG", "POS")


def load_prompts(path: Path) -> dict[str, list[str]]:
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if tuple(prompts) != CLASS_NAMES:
        raise ValueError(f"Prompt classes must be ordered as {CLASS_NAMES}.")
    counts = {name: len(prompts[name]) for name in CLASS_NAMES}
    if not counts["NEG"] or counts["NEG"] != counts["POS"]:
        raise ValueError("NEG and POS must contain the same non-zero prompt count.")
    flat = [text.strip() for name in CLASS_NAMES for text in prompts[name]]
    if any(not text for text in flat) or len(flat) != len(set(flat)):
        raise ValueError("Prompts must be non-empty and unique.")
    return {name: [text.strip() for text in prompts[name]] for name in CLASS_NAMES}


def extract_text_features(output: Any) -> torch.Tensor:
    """Support Transformers versions returning a tensor or a model output."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    raise TypeError(f"Unsupported CLIP text-feature output: {type(output).__name__}")


def encode_texts(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        output = model.get_text_features(**encoded)
    return functional.normalize(extract_text_features(output), dim=-1)


def build_class_prototypes(
    model: Any,
    tokenizer: Any,
    prompts: dict[str, list[str]],
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    prototypes: list[torch.Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    for class_name in CLASS_NAMES:
        features = encode_texts(
            model, tokenizer, prompts[class_name], device, max_length
        )
        prototype = functional.normalize(features.mean(dim=0), dim=-1)
        prototypes.append(prototype)
        similarities = features @ prototype
        for index, (prompt, similarity) in enumerate(
            zip(prompts[class_name], similarities, strict=True), start=1
        ):
            diagnostics.append(
                {
                    "class_name": class_name,
                    "prompt_index": index,
                    "prompt": prompt,
                    "cosine_to_class_prototype": float(similarity.item()),
                }
            )
    return torch.stack(prototypes), diagnostics
