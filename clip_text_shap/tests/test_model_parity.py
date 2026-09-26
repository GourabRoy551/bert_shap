"""Integration check for the CLIP wrapper and direct prototype calculation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from clip_text_wrapper import ClipTextSentimentScores, load_model_and_tokenizer
from prompt_prototypes import build_class_prototypes, encode_texts, load_prompts


ROOT = Path(__file__).resolve().parents[1]


class ModelParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        cls.device = torch.device("cpu")
        cls.model, cls.tokenizer = load_model_and_tokenizer(
            cls.config["model_name"], cls.device, local_files_only=True
        )
        prompts = load_prompts(ROOT / "prompts.json")
        cls.prototypes, _ = build_class_prototypes(
            cls.model,
            cls.tokenizer,
            prompts,
            cls.device,
            int(cls.config["max_length"]),
        )

    def test_wrapper_matches_direct_cosine_similarity(self) -> None:
        texts = ["A beautiful and deeply moving film.", "A dull and terrible film."]
        wrapper = ClipTextSentimentScores(
            self.model,
            self.tokenizer,
            self.prototypes,
            self.device,
            int(self.config["max_length"]),
        )
        wrapped = wrapper(texts)
        direct_features = encode_texts(
            self.model,
            self.tokenizer,
            texts,
            self.device,
            int(self.config["max_length"]),
        )
        direct = (direct_features @ self.prototypes.T).detach().cpu().numpy()
        self.assertEqual(wrapped.shape, (2, 2))
        np.testing.assert_allclose(wrapped, direct, rtol=0.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

