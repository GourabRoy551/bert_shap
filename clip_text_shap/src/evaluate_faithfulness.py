"""Top-k deletion faithfulness checks for dual-class CLIP text SHAP."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_plots import save_figure
from masking_strategy import apply_feature_mask
from prompt_prototypes import CLASS_NAMES


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _deleted_text(
    masker: Any,
    text: str,
    feature_count: int,
    groups: Sequence[dict[str, Any]],
    selected_group_indices: Sequence[int],
) -> str:
    kept = np.ones(feature_count, dtype=bool)
    for group_index in selected_group_indices:
        for feature_index in groups[group_index]["feature_indices"]:
            kept[int(feature_index)] = False
    return apply_feature_mask(masker, text, kept)


def evaluate_sentence(
    sentence_id: str,
    text: str,
    scores: np.ndarray,
    shap_values: np.ndarray,
    word_groups: Sequence[dict[str, Any]],
    model_function: Any,
    masker: Any,
    top_ks: Sequence[int],
    random_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Compare SHAP-ranked deletion with equally sized random deletion."""
    content_group_indices = [
        index for index, group in enumerate(word_groups) if group["is_content"]
    ]
    if not content_group_indices:
        return []

    feature_count = len(shap_values)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for class_index, class_name in enumerate(CLASS_NAMES):
        ranked = sorted(
            content_group_indices,
            key=lambda index: float(word_groups[index]["values"][class_index]),
            reverse=True,
        )
        for requested_k in top_ks:
            k = min(int(requested_k), len(ranked))
            selected = ranked[:k]
            top_text = _deleted_text(
                masker, text, feature_count, word_groups, selected
            )
            top_score = float(model_function([top_text])[0, class_index])
            top_drop = float(scores[class_index] - top_score)

            random_texts: list[str] = []
            for _ in range(random_repeats):
                random_indices = rng.choice(
                    content_group_indices, size=k, replace=False
                ).tolist()
                random_texts.append(
                    _deleted_text(
                        masker, text, feature_count, word_groups, random_indices
                    )
                )
            random_scores = model_function(random_texts)[:, class_index]
            random_drops = float(scores[class_index]) - np.asarray(random_scores)
            rows.append(
                {
                    "sentence_id": sentence_id,
                    "class_index": class_index,
                    "class_name": class_name,
                    "requested_k": int(requested_k),
                    "actual_k": k,
                    "original_score": float(scores[class_index]),
                    "top_k_score": top_score,
                    "top_k_score_drop": top_drop,
                    "random_mean_score": float(np.mean(random_scores)),
                    "random_mean_score_drop": float(np.mean(random_drops)),
                    "random_std_score_drop": float(np.std(random_drops)),
                    "top_minus_random_drop": float(top_drop - np.mean(random_drops)),
                    "selected_words": " | ".join(
                        str(word_groups[index]["word"]) for index in selected
                    ),
                    "top_k_masked_text": top_text,
                }
            )
    return rows


def summarise_faithfulness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["sentence_id"]), str(row["class_name"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (sentence_id, class_name), group in grouped.items():
        group.sort(key=lambda row: int(row["actual_k"]))
        summaries.append(
            {
                "sentence_id": sentence_id,
                "class_name": class_name,
                "evaluated_k_values": " | ".join(
                    str(row["actual_k"]) for row in group
                ),
                "top_k_aopc": float(
                    np.mean([float(row["top_k_score_drop"]) for row in group])
                ),
                "random_aopc": float(
                    np.mean([float(row["random_mean_score_drop"]) for row in group])
                ),
                "aopc_improvement": float(
                    np.mean([float(row["top_minus_random_drop"]) for row in group])
                ),
            }
        )
    return summaries


def save_faithfulness_curve(
    output_stem: Path, rows: list[dict[str, Any]]
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for class_index, class_name in enumerate(CLASS_NAMES):
        axis = axes[class_index]
        class_rows = [row for row in rows if row["class_name"] == class_name]
        ks = sorted({int(row["actual_k"]) for row in class_rows})
        top_means = [
            float(
                np.mean(
                    [
                        float(row["top_k_score_drop"])
                        for row in class_rows
                        if int(row["actual_k"]) == k
                    ]
                )
            )
            for k in ks
        ]
        random_means = [
            float(
                np.mean(
                    [
                        float(row["random_mean_score_drop"])
                        for row in class_rows
                        if int(row["actual_k"]) == k
                    ]
                )
            )
            for k in ks
        ]
        axis.plot(ks, top_means, "o-", linewidth=2, label="SHAP top-k")
        axis.plot(ks, random_means, "s--", linewidth=2, label="Random mean")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(ks)
        axis.set_xlabel("Number of deleted content words (k)")
        axis.set_title(f"Target: {class_name}")
        axis.grid(alpha=0.22)
        axis.margins(y=0.15)
        axis.legend()
        for x_value, top_value, random_value in zip(
            ks, top_means, random_means, strict=True
        ):
            axis.annotate(
                f"{top_value:+.4f}",
                (x_value, top_value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
            axis.annotate(
                f"{random_value:+.4f}",
                (x_value, random_value),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axes[0].set_ylabel("Mean target-score drop")
    figure.suptitle(
        "Faithfulness: deleting positive SHAP-ranked words versus random words",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, output_stem)


def save_aopc_matrix(output_stem: Path, summaries: list[dict[str, Any]]) -> None:
    sentence_ids = list(dict.fromkeys(str(row["sentence_id"]) for row in summaries))
    lookup = {
        (str(row["sentence_id"]), str(row["class_name"])): float(row["top_k_aopc"])
        for row in summaries
    }
    values = np.asarray(
        [[lookup.get((sentence_id, name), np.nan) for name in CLASS_NAMES] for sentence_id in sentence_ids]
    )
    finite = values[np.isfinite(values)]
    maximum = float(np.max(np.abs(finite))) if finite.size else 1.0
    maximum = maximum if maximum > 1e-12 else 1.0
    figure, axis = plt.subplots(
        figsize=(7.5, max(7.0, 0.4 * len(sentence_ids) + 2.5))
    )
    image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-maximum, vmax=maximum)
    axis.set_xticks([0, 1], labels=CLASS_NAMES)
    axis.set_yticks(np.arange(len(sentence_ids)), labels=sentence_ids)
    axis.set_title("Per-sentence AOPC for SHAP top-k deletion", pad=11)
    for row in range(values.shape[0]):
        for column in range(2):
            value = values[row, column]
            axis.text(
                column,
                row,
                "n/a" if not np.isfinite(value) else f"{value:+.4f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color="white" if np.isfinite(value) and abs(value) > 0.58 * maximum else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025).set_label(
        "Mean target-score drop (AOPC)"
    )
    figure.tight_layout()
    save_figure(figure, output_stem)

