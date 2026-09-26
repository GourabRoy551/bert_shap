"""One-sentence integration test for two-output Partition SHAP additivity."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import shap
import torch

from clip_text_wrapper import ClipTextSentimentScores, load_model_and_tokenizer
from masking_strategy import create_text_masker
from prompt_prototypes import CLASS_NAMES, build_class_prototypes, load_prompts
from token_aggregation import calculate_additivity, unpack_explanation


ROOT = Path(__file__).resolve().parents[1]


class AdditivityIntegrationTests(unittest.TestCase):
    def test_both_outputs_reconstruct_the_model_scores(self) -> None:
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        device = torch.device("cpu")
        model, tokenizer = load_model_and_tokenizer(
            config["model_name"], device, local_files_only=True
        )
        prototypes, _ = build_class_prototypes(
            model,
            tokenizer,
            load_prompts(ROOT / "prompts.json"),
            device,
            int(config["max_length"]),
        )
        model_function = ClipTextSentimentScores(
            model, tokenizer, prototypes, device, int(config["max_length"])
        )
        masker = create_text_masker(tokenizer, "")
        explainer = shap.Explainer(
            model_function,
            masker,
            algorithm="partition",
            output_names=list(CLASS_NAMES),
        )
        text = "The film is moving and beautifully performed."
        scores = model_function([text])[0]
        explanation = explainer([text], max_evals=100, batch_size=32, silent=True)
        _, values, base_values = unpack_explanation(explanation)
        result = calculate_additivity(scores, base_values, values)
        np.testing.assert_allclose(result["residual"], [0.0, 0.0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()

