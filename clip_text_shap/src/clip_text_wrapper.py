"""Batched CLIP Text Encoder wrapper returning NEG and POS similarities."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from prompt_prototypes import CLASS_NAMES, encode_texts


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_model_and_tokenizer(
    model_name: str, device: torch.device, local_files_only: bool
) -> tuple[Any, Any]:
    from transformers import CLIPModel, CLIPTokenizerFast

    tokenizer = CLIPTokenizerFast.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    model = CLIPModel.from_pretrained(
        model_name, local_files_only=local_files_only
    ).to(device)
    model.eval()
    return model, tokenizer


class ClipTextSentimentScores:
    """Map strings to cosine similarity with fixed NEG/POS prototypes."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prototypes: torch.Tensor,
        device: torch.device,
        max_length: int = 77,
    ) -> None:
        if tuple(prototypes.shape[:1]) != (len(CLASS_NAMES),):
            raise ValueError("Expected one prototype for NEG and one for POS.")
        self.model = model
        self.tokenizer = tokenizer
        self.prototypes = prototypes.to(device)
        self.device = device
        self.max_length = max_length
        self.forward_batches = 0
        self.text_evaluations = 0

    def __call__(self, texts: Sequence[str] | np.ndarray) -> np.ndarray:
        values = np.asarray(texts, dtype=object).reshape(-1).tolist()
        values = [str(text) for text in values]
        features = encode_texts(
            self.model, self.tokenizer, values, self.device, self.max_length
        )
        scores = features @ self.prototypes.T
        self.forward_batches += 1
        self.text_evaluations += len(values)
        return scores.detach().cpu().numpy().astype(np.float64, copy=False)


def uncalibrated_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / exp.sum()
