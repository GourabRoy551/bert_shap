"""Validation tests for the balanced NEG/POS prompt definitions."""

from __future__ import annotations

import unittest
from pathlib import Path

from prompt_prototypes import CLASS_NAMES, load_prompts


ROOT = Path(__file__).resolve().parents[1]


class PromptTests(unittest.TestCase):
    def test_prompts_are_balanced_unique_and_ordered(self) -> None:
        prompts = load_prompts(ROOT / "prompts.json")
        self.assertEqual(tuple(prompts), CLASS_NAMES)
        self.assertEqual(len(prompts["NEG"]), len(prompts["POS"]))
        self.assertGreater(len(prompts["NEG"]), 0)
        self.assertEqual(
            len({text for values in prompts.values() for text in values}),
            len(prompts["NEG"]) + len(prompts["POS"]),
        )


if __name__ == "__main__":
    unittest.main()

