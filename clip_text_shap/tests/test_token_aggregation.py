"""Unit tests for CLIP BPE-to-word aggregation and SHAP arithmetic."""

from __future__ import annotations

import unittest

import numpy as np

from token_aggregation import aggregate_clip_bpe, calculate_additivity, margin_values


class TokenAggregationTests(unittest.TestCase):
    def test_bpe_contraction_and_multi_hyphen_merge(self) -> None:
        feature_texts = ["", "It ", "'s ", "paint", "-", "by", "-", "numbers", ".", ""]
        model_tokens = [
            "<|startoftext|>",
            "it</w>",
            "'s</w>",
            "paint</w>",
            "-</w>",
            "by</w>",
            "-</w>",
            "numbers</w>",
            ".</w>",
            "<|endoftext|>",
        ]
        values = np.arange(20, dtype=float).reshape(10, 2)
        groups = aggregate_clip_bpe(feature_texts, model_tokens, values)
        words = [group["word"] for group in groups]
        self.assertIn("It's", words)
        self.assertIn("paint-by-numbers", words)
        compound = next(group for group in groups if group["word"] == "paint-by-numbers")
        np.testing.assert_allclose(compound["values"], values[3:8].sum(axis=0))

    def test_dual_additivity_and_margin_identity(self) -> None:
        values = np.asarray([[0.1, -0.2], [0.3, 0.4]])
        base = np.asarray([0.2, -0.1])
        scores = base + values.sum(axis=0)
        result = calculate_additivity(scores, base, values)
        np.testing.assert_allclose(result["residual"], [0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(
            float(scores[1] - scores[0]),
            float(base[1] - base[0] + margin_values(values).sum()),
        )


if __name__ == "__main__":
    unittest.main()

