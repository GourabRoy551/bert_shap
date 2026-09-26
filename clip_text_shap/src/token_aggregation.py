"""Unpack CLIP SHAP explanations and aggregate BPE pieces into words."""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np


SPECIAL_TOKENS = {"<|startoftext|>", "<|endoftext|>"}
CONTRACTION = re.compile(r"^['’](?:s|t|re|ve|ll|d|m)$", re.IGNORECASE)
HYPHENS = {"-", "‐", "‑", "–"}


def unpack_explanation(explanation: Any) -> tuple[list[str], np.ndarray, np.ndarray]:
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3:
        values = values[0]
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected SHAP values shaped (features, 2), got {values.shape}.")

    data = explanation.data
    if isinstance(data, np.ndarray) and data.ndim > 1:
        data = data[0]
    feature_texts = [str(value) for value in np.asarray(data, dtype=object).reshape(-1)]

    base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    if base_values.size != 2:
        raise ValueError(f"Expected two base values, got {base_values.shape}.")
    if len(feature_texts) != values.shape[0]:
        raise ValueError("SHAP feature text and value counts differ.")
    return feature_texts, values, base_values


def calculate_additivity(
    scores: Sequence[float], base_values: np.ndarray, values: np.ndarray
) -> dict[str, np.ndarray]:
    scores_array = np.asarray(scores, dtype=float)
    shap_sum = values.sum(axis=0)
    reconstructed = base_values + shap_sum
    return {
        "shap_sum": shap_sum,
        "reconstructed": reconstructed,
        "residual": scores_array - reconstructed,
    }


def _row(indices: list[int], text: str, values: np.ndarray, special: bool) -> dict[str, Any]:
    clean = text.strip() or text
    return {
        "feature_indices": list(indices),
        "word": clean,
        "values": values[indices].sum(axis=0),
        "is_special": special,
        "is_content": (not special and bool(re.search(r"[A-Za-z0-9]", clean))),
    }


def aggregate_clip_bpe(
    feature_texts: Sequence[str],
    model_tokens: Sequence[str],
    values: np.ndarray,
) -> list[dict[str, Any]]:
    if len(feature_texts) != len(model_tokens) or len(feature_texts) != len(values):
        raise ValueError("Feature texts, model tokens, and SHAP values must align.")

    groups: list[dict[str, Any]] = []
    pending_indices: list[int] = []
    pending_text = ""

    def flush() -> None:
        nonlocal pending_indices, pending_text
        if pending_indices:
            groups.append(_row(pending_indices, pending_text, values, False))
            pending_indices = []
            pending_text = ""

    for index, (feature_text, model_token) in enumerate(
        zip(feature_texts, model_tokens, strict=True)
    ):
        if model_token in SPECIAL_TOKENS:
            flush()
            groups.append(_row([index], model_token, values, True))
            continue
        pending_indices.append(index)
        pending_text += feature_text
        if model_token.endswith("</w>"):
            flush()
    flush()

    contracted: list[dict[str, Any]] = []
    for group in groups:
        if (
            contracted
            and not group["is_special"]
            and CONTRACTION.match(group["word"])
            and not contracted[-1]["is_special"]
        ):
            previous = contracted[-1]
            previous["feature_indices"].extend(group["feature_indices"])
            previous["word"] += group["word"]
            previous["values"] = previous["values"] + group["values"]
            previous["is_content"] = True
        else:
            contracted.append(group)

    # Merge repeatedly so compounds such as "paint-by-numbers" remain one word.
    merged = list(contracted)
    index = 0
    while index + 2 < len(merged):
        if (
            merged[index + 1]["word"] in HYPHENS
            and merged[index]["is_content"]
            and merged[index + 2]["is_content"]
        ):
            left, hyphen, right = merged[index : index + 3]
            combined = {
                "feature_indices": left["feature_indices"]
                + hyphen["feature_indices"]
                + right["feature_indices"],
                "word": left["word"] + hyphen["word"] + right["word"],
                "values": left["values"] + hyphen["values"] + right["values"],
                "is_special": False,
                "is_content": True,
            }
            merged[index : index + 3] = [combined]
        else:
            index += 1

    for word_index, group in enumerate(merged):
        group["word_index"] = word_index
    return merged


def margin_values(values: np.ndarray) -> np.ndarray:
    return values[:, 1] - values[:, 0]
