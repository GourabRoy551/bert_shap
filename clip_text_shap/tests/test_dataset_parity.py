"""Protect the approved dataset from accidental replacement or editing."""

from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DatasetParityTests(unittest.TestCase):
    def test_manifest_hash_and_counts(self) -> None:
        data_path = ROOT / "data" / "sentences_20_curated.csv"
        manifest = json.loads(
            (ROOT / "data" / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        self.assertEqual(digest, manifest["sha256"])
        with data_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(row["gold_label"] == "NEG" for row in rows), 10)
        self.assertEqual(sum(row["gold_label"] == "POS" for row in rows), 10)

    def test_records_identical_to_approved_bert_input(self) -> None:
        clip_data = ROOT / "data" / "sentences_20_curated.csv"
        bert_data = ROOT.parent / "bert_shap" / "dual_class_bert" / "data" / "sentences_20_curated.csv"
        with clip_data.open("r", encoding="utf-8-sig", newline="") as stream:
            clip_rows = list(csv.DictReader(stream))
        with bert_data.open("r", encoding="utf-8-sig", newline="") as stream:
            bert_rows = list(csv.DictReader(stream))
        self.assertEqual(clip_rows, bert_rows)


if __name__ == "__main__":
    unittest.main()

