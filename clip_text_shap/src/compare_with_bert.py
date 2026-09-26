"""Ranking-based comparison of CLIP and BERT dual-class SHAP outputs."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_plots import save_figure


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _normalise_word(word: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", word.casefold()))


def _with_occurrence_keys(rows: list[dict[str, str]]) -> dict[tuple[str, int], float]:
    counts: dict[str, int] = defaultdict(int)
    result: dict[tuple[str, int], float] = {}
    for row in sorted(rows, key=lambda item: int(item["word_index"])):
        word = _normalise_word(row["word"])
        if not word:
            continue
        occurrence = counts[word]
        counts[word] += 1
        result[(word, occurrence)] = float(row["margin_shap_value"])
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks, matching Spearman's treatment of ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2:
        return float("nan")
    first_ranks = _rankdata(first)
    second_ranks = _rankdata(second)
    if np.std(first_ranks) < 1e-12 or np.std(second_ranks) < 1e-12:
        return float("nan")
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def compare_runs(
    clip_summary_rows: list[dict[str, Any]],
    clip_word_rows: list[dict[str, Any]],
    bert_values_dir: Path,
) -> list[dict[str, Any]]:
    bert_summary = {
        row["sentence_id"]: row
        for row in _read_csv(bert_values_dir / "sentence_summary.csv")
    }
    bert_words_by_sentence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(bert_values_dir / "word_shap_values_wide.csv"):
        bert_words_by_sentence[row["sentence_id"]].append(row)
    clip_words_by_sentence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in clip_word_rows:
        clip_words_by_sentence[str(row["sentence_id"])].append(
            {key: str(value) for key, value in row.items()}
        )

    results: list[dict[str, Any]] = []
    for summary in clip_summary_rows:
        sentence_id = str(summary["sentence_id"])
        if sentence_id not in bert_summary:
            continue
        clip_values = _with_occurrence_keys(clip_words_by_sentence[sentence_id])
        bert_values = _with_occurrence_keys(bert_words_by_sentence[sentence_id])
        shared = sorted(set(clip_values) & set(bert_values))
        clip_array = np.asarray([clip_values[key] for key in shared], dtype=float)
        bert_array = np.asarray([bert_values[key] for key in shared], dtype=float)
        top_count = min(3, len(shared))
        clip_top = {
            shared[index]
            for index in np.argsort(np.abs(clip_array))[-top_count:]
        }
        bert_top = {
            shared[index]
            for index in np.argsort(np.abs(bert_array))[-top_count:]
        }
        union = clip_top | bert_top
        nonzero = (np.abs(clip_array) > 1e-12) & (np.abs(bert_array) > 1e-12)
        sign_agreement = (
            float(np.mean(np.sign(clip_array[nonzero]) == np.sign(bert_array[nonzero])))
            if np.any(nonzero)
            else float("nan")
        )
        bert_prediction = str(bert_summary[sentence_id]["prediction"])
        results.append(
            {
                "sentence_id": sentence_id,
                "gold_label": summary["gold_label"],
                "clip_prediction": summary["prediction"],
                "bert_prediction": bert_prediction,
                "prediction_agreement": int(summary["prediction"] == bert_prediction),
                "shared_word_count": len(shared),
                "spearman_margin_rank": _spearman(clip_array, bert_array),
                "top3_overlap_count": len(clip_top & bert_top),
                "top3_jaccard": float(len(clip_top & bert_top) / len(union))
                if union
                else float("nan"),
                "margin_sign_agreement": sign_agreement,
                "clip_top3_words": " | ".join(key[0] for key in sorted(clip_top)),
                "bert_top3_words": " | ".join(key[0] for key in sorted(bert_top)),
                "comparison_note": (
                    "Ranks/directions only; raw BERT logits and CLIP cosine "
                    "similarities are not magnitude-comparable."
                ),
            }
        )
    return results


def save_comparison_matrix(
    output_stem: Path, rows: list[dict[str, Any]]
) -> None:
    if not rows:
        return
    columns = ["Spearman", "Top-3 Jaccard", "Sign agreement", "Pred. agreement"]
    values = np.asarray(
        [
            [
                row["spearman_margin_rank"],
                row["top3_jaccard"],
                row["margin_sign_agreement"],
                row["prediction_agreement"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    shown = np.nan_to_num(values, nan=0.0)
    figure, axis = plt.subplots(figsize=(11, max(7.0, 0.42 * len(rows) + 3.0)))
    image = axis.imshow(shown, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_xticks(np.arange(len(columns)), labels=columns, rotation=25, ha="right")
    axis.set_yticks(
        np.arange(len(rows)), labels=[str(row["sentence_id"]) for row in rows]
    )
    axis.set_title("BERT versus CLIP text SHAP ranking comparison", pad=12)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                "n/a" if not np.isfinite(value) else f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color="white" if np.isfinite(value) and abs(value) > 0.58 else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025).set_label(
        "Agreement metric"
    )
    figure.text(
        0.5,
        0.01,
        "Comparison uses POS-minus-NEG word contribution ranks/directions; "
        "raw magnitudes are not compared.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(figure, output_stem)

