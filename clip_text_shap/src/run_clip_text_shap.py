"""Run dual-class Partition SHAP on a CLIP Text Encoder sentiment proxy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    import shap
except ImportError:
    shap = None

from clip_text_wrapper import (
    ClipTextSentimentScores,
    choose_device,
    load_model_and_tokenizer,
    uncalibrated_softmax,
)
from compare_with_bert import compare_runs, save_comparison_matrix
from evaluate_faithfulness import (
    evaluate_sentence,
    save_aopc_matrix,
    save_faithfulness_curve,
    summarise_faithfulness,
)
from make_plots import (
    save_additivity_residual_matrix,
    save_all_sentence_summary_matrix,
    save_class_comparison_bars,
    save_class_value_matrix,
    save_dual_word_bar,
    save_global_word_class_matrix,
    save_sentence_class_magnitude_matrix,
    save_sentence_overview,
    save_sentence_summary_matrix,
)
from masking_strategy import create_text_masker, masking_diagnostics
from prompt_prototypes import CLASS_NAMES, build_class_prototypes, load_prompts
from token_aggregation import (
    aggregate_clip_bpe,
    calculate_additivity,
    margin_values,
    unpack_explanation,
)


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.json"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The top-level configuration must be a JSON object.")
    return config


def experiment_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else EXPERIMENT_DIR / path


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    known, _ = pre_parser.parse_known_args()
    settings = load_config(known.config.resolve())

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description="Explain NEG and POS CLIP text-prototype similarities with SHAP.",
    )
    parser.add_argument("--input", type=Path, default=experiment_path(settings["input_path"]))
    parser.add_argument(
        "--output-root", type=Path, default=experiment_path(settings["output_root"])
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--sentence-ids", nargs="+", default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--model-name", default=settings["model_name"])
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default=settings.get("device", "auto")
    )
    parser.add_argument("--max-length", type=int, default=int(settings.get("max_length", 77)))
    parser.add_argument("--max-evals", type=int, default=int(settings.get("max_evals", 500)))
    parser.add_argument("--batch-size", type=int, default=int(settings.get("batch_size", 32)))
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=bool(settings.get("local_files_only", True)),
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-faithfulness", action="store_true")
    parser.add_argument("--no-bert-comparison", action="store_true")
    return parser.parse_args()


def read_examples(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sentence_id", "text", "gold_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")
        examples: list[dict[str, str]] = []
        for row in reader:
            label = row["gold_label"].strip().upper()
            if label not in CLASS_NAMES:
                raise ValueError(f"Invalid label: {label!r}")
            examples.append(
                {
                    "sentence_id": row["sentence_id"].strip(),
                    "text": row["text"].strip(),
                    "gold_label": label,
                    "source": (row.get("source") or "").strip(),
                    "source_row_id": (row.get("source_row_id") or "").strip(),
                }
            )
    if len({row["sentence_id"] for row in examples}) != len(examples):
        raise ValueError("Sentence IDs must be unique.")
    return examples


def select_examples(
    examples: list[dict[str, str]],
    sentence_ids: Sequence[str] | None,
    max_sentences: int | None,
) -> list[dict[str, str]]:
    selected = examples
    if sentence_ids:
        requested = {value.upper() for value in sentence_ids}
        selected = [
            row for row in examples if row["sentence_id"].upper() in requested
        ]
        found = {row["sentence_id"].upper() for row in selected}
        if missing := requested - found:
            raise ValueError(f"Unknown sentence IDs: {sorted(missing)}")
    if max_sentences is not None:
        if max_sentences < 1:
            raise ValueError("--max-sentences must be at least one.")
        selected = selected[:max_sentences]
    if not selected:
        raise RuntimeError("No sentences selected.")
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(path: Path, manifest_path: Path, examples: list[dict[str, str]]) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hash = _sha256(path)
    if actual_hash.casefold() != str(manifest["sha256"]).casefold():
        raise ValueError("Dataset SHA-256 does not match the approved manifest.")
    if len(examples) != int(manifest["rows"]):
        raise ValueError("Dataset row count does not match the approved manifest.")
    label_counts = {name: sum(row["gold_label"] == name for row in examples) for name in CLASS_NAMES}
    if label_counts != {
        "NEG": int(manifest["negative_rows"]),
        "POS": int(manifest["positive_rows"]),
    }:
        raise ValueError("Dataset class counts do not match the approved manifest.")


def create_output_dirs(root: Path, run_name: str | None) -> dict[str, Path]:
    name = run_name or datetime.now().strftime("clip_text_dual_%Y%m%d_%H%M%S")
    directories = {
        category: root / category / name
        for category in ("values", "plots", "evaluation", "reports")
    }
    if existing := [path for path in directories.values() if path.exists()]:
        raise FileExistsError(
            "The requested run already exists: " + ", ".join(str(path) for path in existing)
        )
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=False)
    return directories


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


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def model_tokens(tokenizer: Any, text: str, max_length: int) -> list[str]:
    encoded = tokenizer(
        text, truncation=True, max_length=max_length, add_special_tokens=True
    )
    return [str(token) for token in tokenizer.convert_ids_to_tokens(encoded["input_ids"])]


def top_word(
    groups: Sequence[dict[str, Any]], class_index: int, largest: bool
) -> str:
    content = [group for group in groups if group["is_content"]]
    if not content:
        return ""
    selected = (
        max(content, key=lambda item: float(item["values"][class_index]))
        if largest
        else min(content, key=lambda item: float(item["values"][class_index]))
    )
    return f"{selected['word']} ({float(selected['values'][class_index]):+.4f})"


def build_report(
    summaries: list[dict[str, Any]],
    faithfulness: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    correct = sum(int(row["correct"]) for row in summaries)
    maximum_residual = max(
        max(
            abs(float(row["negative_additivity_residual"])),
            abs(float(row["positive_additivity_residual"])),
        )
        for row in summaries
    )
    lines = [
        "# CLIP Text Encoder Dual-Class SHAP Run Report",
        "",
        "## Experiment",
        "",
        f"- Model: `{config['model_name']}`",
        "- Explained outputs: cosine similarity to fixed NEG and POS prompt prototypes",
        "- Explainer: Partition SHAP with deletion masking (CLIP has no mask token)",
        f"- Sentences: {len(summaries)}",
        f"- Proxy-classification accuracy: {correct}/{len(summaries)}",
        f"- Maximum class additivity residual: {maximum_residual:.3e}",
        f"- Maximum evaluations per sentence: {config['max_evals']}",
        "- The displayed softmax share is descriptive and is not a calibrated probability.",
        "",
        "## Sentence results",
        "",
        "| ID | Gold | Pred. | NEG score | POS score | Top NEG increase | Top POS increase |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in summaries:
        lines.append(
            "| {sentence_id} | {gold_label} | {prediction} | "
            "{negative_score:.4f} | {positive_score:.4f} | "
            "{top_negative_increasing_word} | {top_positive_increasing_word} |".format(**row)
        )
    if faithfulness:
        top_aopc = [float(row["top_k_aopc"]) for row in faithfulness]
        random_aopc = [float(row["random_aopc"]) for row in faithfulness]
        improvements = [float(row["aopc_improvement"]) for row in faithfulness]
        lines.extend(
            [
                "",
                "## Faithfulness",
                "",
                "Positive AOPC means that deleting SHAP-ranked words lowers the target "
                "similarity. The random baseline deletes the same number of content words.",
                "",
                f"- Mean SHAP top-k deletion AOPC: {np.mean(top_aopc):+.4f}",
                f"- Mean random deletion AOPC: {np.mean(random_aopc):+.4f}",
                f"- Mean SHAP-minus-random AOPC improvement: {np.mean(improvements):+.4f}",
                f"- Positive sentence/class improvements: {sum(value > 0 for value in improvements)}/{len(improvements)}",
            ]
        )
    if comparisons:
        correlations = [
            float(row["spearman_margin_rank"])
            for row in comparisons
            if np.isfinite(float(row["spearman_margin_rank"]))
        ]
        lines.extend(
            [
                "",
                "## BERT comparison",
                "",
                "BERT and CLIP are compared by POS-minus-NEG word-contribution rankings "
                "and directions. Raw magnitudes are not compared because the explained "
                "outputs are BERT logits versus CLIP cosine similarities.",
                "",
                f"- Mean sentence-level Spearman rank correlation: {np.mean(correlations):+.3f}"
                if correlations
                else "- Mean sentence-level Spearman rank correlation: unavailable",
                f"- Mean top-3 Jaccard overlap: {np.mean([float(row['top3_jaccard']) for row in comparisons]):.3f}",
                f"- Mean margin-sign agreement: {np.mean([float(row['margin_sign_agreement']) for row in comparisons]):.3f}",
                f"- Prediction agreement: {sum(int(row['prediction_agreement']) for row in comparisons)}/{len(comparisons)}",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A positive SHAP value raises the named class-prototype similarity relative "
            "to the deletion-masked baseline; a negative value lowers it. Each sentence "
            "has values for both NEG and POS, irrespective of its gold or predicted class.",
            "",
            "All heatmap cells and bars display their numerical values, and every figure "
            "is saved as both PNG and PDF.",
            "",
            "## Limitation",
            "",
            "CLIP is a prompt-based sentiment proxy here, not a fine-tuned sentiment "
            "classifier. Shared prompt-anchor words such as *movie* and *film* can raise "
            "both similarities, so POS-minus-NEG contributions should be inspected for "
            "class discrimination. This 20-sentence run is not sufficient for a broad "
            "claim that CLIP is better than BERT or RFEM.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.max_evals < 3:
        raise ValueError("--max-evals must be at least three.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    settings = load_config(args.config.resolve())
    examples_all = read_examples(args.input)
    validate_dataset(
        args.input,
        experiment_path(settings["dataset_manifest_path"]),
        examples_all,
    )
    examples = select_examples(examples_all, args.sentence_ids, args.max_sentences)
    directories = create_output_dirs(args.output_root, args.run_name)
    values_dir = directories["values"]
    plots_dir = directories["plots"]
    evaluation_dir = directories["evaluation"]
    reports_dir = directories["reports"]

    start_time = time.time()
    seed = int(settings.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = choose_device(args.device)
    print(f"Loading {args.model_name} on {device}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, device, local_files_only=args.local_files_only
    )
    prompts = load_prompts(experiment_path(settings["prompts_path"]))
    prototypes, prompt_rows = build_class_prototypes(
        model, tokenizer, prompts, device, args.max_length
    )
    prototype_cosine = float(torch.dot(prototypes[0], prototypes[1]).item())
    model_function = ClipTextSentimentScores(
        model, tokenizer, prototypes, device, args.max_length
    )
    if shap is None:
        raise RuntimeError("SHAP is unavailable in the selected environment.")
    masker = create_text_masker(tokenizer, str(settings.get("mask_token", "")))
    explainer = shap.Explainer(
        model_function,
        masker,
        algorithm="partition",
        output_names=list(CLASS_NAMES),
    )

    run_config: dict[str, Any] = {
        "config": str(args.config.resolve()),
        "input": str(args.input.resolve()),
        "dataset_sha256": _sha256(args.input),
        "model_name": args.model_name,
        "device_requested": args.device,
        "device_resolved": str(device),
        "max_length": args.max_length,
        "max_evals": args.max_evals,
        "batch_size": args.batch_size,
        "selected_sentence_count": len(examples),
        "selected_sentence_ids": [row["sentence_id"] for row in examples],
        "output_names": list(CLASS_NAMES),
        "score_definition": "cosine similarity to mean-normalized class prompt prototype",
        "prediction_rule": "argmax over [NEG similarity, POS similarity]",
        "masking_strategy": "token deletion using empty replacement text",
        "softmax_note": "descriptive uncalibrated share; not a probability",
        "negative_positive_prototype_cosine": prototype_cosine,
        "values_dir": str(values_dir.resolve()),
        "plots_dir": str(plots_dir.resolve()),
        "evaluation_dir": str(evaluation_dir.resolve()),
        "reports_dir": str(reports_dir.resolve()),
        "versions": {
            "python": sys.version.split()[0],
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "shap": package_version("shap"),
            "numpy": package_version("numpy"),
            "matplotlib": package_version("matplotlib"),
        },
    }
    (values_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    write_csv(values_dir / "prompt_diagnostics.csv", prompt_rows)

    summary_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    word_rows: list[dict[str, Any]] = []
    wide_word_rows: list[dict[str, Any]] = []
    masking_rows: list[dict[str, Any]] = []
    faithfulness_rows: list[dict[str, Any]] = []

    print(
        f"Examples: {len(examples)} | max_evals={args.max_evals} | "
        f"batch_size={args.batch_size}"
    )
    for item_index, item in enumerate(examples, start=1):
        sentence_start = time.time()
        sentence_id = item["sentence_id"]
        text = item["text"]
        gold_label = item["gold_label"]
        print(f"[{item_index}/{len(examples)}] {sentence_id}: {text}")

        tokens = model_tokens(tokenizer, text, args.max_length)
        diagnostics = masking_diagnostics(masker, text)
        if int(diagnostics["feature_count"]) != len(tokens):
            raise ValueError(
                f"{sentence_id}: masker feature count {diagnostics['feature_count']} "
                f"does not equal model-token count {len(tokens)}."
            )
        scores = model_function([text])[0]
        shares = uncalibrated_softmax(scores)
        prediction = CLASS_NAMES[int(np.argmax(scores))]
        evaluations_before = model_function.text_evaluations
        explanation = explainer(
            [text], max_evals=args.max_evals, batch_size=args.batch_size, silent=True
        )
        feature_texts, shap_values, base_values = unpack_explanation(explanation)
        if len(feature_texts) != len(tokens):
            raise ValueError(
                f"{sentence_id}: SHAP features ({len(feature_texts)}) and "
                f"CLIP tokens ({len(tokens)}) do not align."
            )
        additivity = calculate_additivity(scores, base_values, shap_values)
        groups = aggregate_clip_bpe(feature_texts, tokens, shap_values)
        display_groups = [group for group in groups if not group["is_special"]]
        words = [str(group["word"]) for group in display_groups]
        word_values = np.asarray([group["values"] for group in display_groups], dtype=float)
        margin = float(scores[1] - scores[0])
        feature_margins = margin_values(shap_values)
        margin_base = float(base_values[1] - base_values[0])
        margin_reconstructed = float(margin_base + feature_margins.sum())

        for feature_index, (feature_text, token) in enumerate(
            zip(feature_texts, tokens, strict=True)
        ):
            for class_index, class_name in enumerate(CLASS_NAMES):
                value = float(shap_values[feature_index, class_index])
                token_rows.append(
                    {
                        "sentence_id": sentence_id,
                        "text": text,
                        "gold_label": gold_label,
                        "prediction": prediction,
                        "class_index": class_index,
                        "class_name": class_name,
                        "feature_index": feature_index,
                        "feature_text": feature_text,
                        "model_token": token,
                        "shap_value": value,
                        "absolute_shap_value": abs(value),
                        "increases_target_score": int(value > 0),
                        "margin_shap_value": float(feature_margins[feature_index]),
                    }
                )

        for word_index, group in enumerate(display_groups):
            negative_value = float(group["values"][0])
            positive_value = float(group["values"][1])
            wide_word_rows.append(
                {
                    "sentence_id": sentence_id,
                    "gold_label": gold_label,
                    "prediction": prediction,
                    "word_index": word_index,
                    "word": group["word"],
                    "feature_indices": " | ".join(
                        str(index) for index in group["feature_indices"]
                    ),
                    "is_content": int(group["is_content"]),
                    "negative_shap_value": negative_value,
                    "positive_shap_value": positive_value,
                    "margin_shap_value": positive_value - negative_value,
                }
            )
            for class_index, class_name in enumerate(CLASS_NAMES):
                value = float(group["values"][class_index])
                word_rows.append(
                    {
                        "sentence_id": sentence_id,
                        "text": text,
                        "gold_label": gold_label,
                        "prediction": prediction,
                        "class_index": class_index,
                        "class_name": class_name,
                        "word_index": word_index,
                        "word": group["word"],
                        "is_content": int(group["is_content"]),
                        "shap_value": value,
                        "absolute_shap_value": abs(value),
                        "increases_target_score": int(value > 0),
                    }
                )

        masking_rows.append(
            {
                "sentence_id": sentence_id,
                "feature_count": diagnostics["feature_count"],
                "original_text": text,
                "fully_unmasked_text": diagnostics["fully_unmasked_text"],
                "fully_masked_text": diagnostics["fully_masked_text"],
            }
        )
        content_groups = [group for group in display_groups if group["is_content"]]
        summary = {
            "sentence_id": sentence_id,
            "text": text,
            "gold_label": gold_label,
            "prediction": prediction,
            "correct": int(prediction == gold_label),
            "source": item["source"],
            "source_row_id": item["source_row_id"],
            "negative_score": float(scores[0]),
            "positive_score": float(scores[1]),
            "negative_softmax_share": float(shares[0]),
            "positive_softmax_share": float(shares[1]),
            "negative_base_value": float(base_values[0]),
            "positive_base_value": float(base_values[1]),
            "negative_sum_shap_values": float(shap_values[:, 0].sum()),
            "positive_sum_shap_values": float(shap_values[:, 1].sum()),
            "negative_reconstructed_score": float(additivity["reconstructed"][0]),
            "positive_reconstructed_score": float(additivity["reconstructed"][1]),
            "negative_additivity_residual": float(additivity["residual"][0]),
            "positive_additivity_residual": float(additivity["residual"][1]),
            "margin_positive_minus_negative": margin,
            "margin_base_value": margin_base,
            "margin_sum_shap_values": float(feature_margins.sum()),
            "margin_reconstructed": margin_reconstructed,
            "margin_additivity_residual": float(margin - margin_reconstructed),
            "feature_count": len(tokens),
            "display_word_count": len(display_groups),
            "content_word_count": len(content_groups),
            "negative_mean_abs_word_shap": float(np.mean(np.abs(word_values[:, 0]))),
            "positive_mean_abs_word_shap": float(np.mean(np.abs(word_values[:, 1]))),
            "negative_total_abs_word_shap": float(np.sum(np.abs(word_values[:, 0]))),
            "positive_total_abs_word_shap": float(np.sum(np.abs(word_values[:, 1]))),
            "top_negative_increasing_word": top_word(display_groups, 0, True),
            "top_negative_decreasing_word": top_word(display_groups, 0, False),
            "top_positive_increasing_word": top_word(display_groups, 1, True),
            "top_positive_decreasing_word": top_word(display_groups, 1, False),
            "shap_model_text_evaluations": model_function.text_evaluations - evaluations_before,
            "runtime_seconds_before_evaluation": time.time() - sentence_start,
        }
        summary_rows.append(summary)

        if not args.no_faithfulness:
            faithfulness_rows.extend(
                evaluate_sentence(
                    sentence_id,
                    text,
                    scores,
                    shap_values,
                    groups,
                    model_function,
                    masker,
                    [int(value) for value in settings.get("faithfulness_top_k", [1, 2, 3])],
                    int(settings.get("faithfulness_random_repeats", 10)),
                    seed + item_index,
                )
            )

        if not args.no_plots:
            sentence_dir = plots_dir / sentence_id
            token_labels = [f"{index}: {token}" for index, token in enumerate(tokens)]
            word_labels = [f"{index}: {word}" for index, word in enumerate(words)]
            common = {
                "sentence_id": sentence_id,
                "text": text,
                "gold_label": gold_label,
                "prediction": prediction,
            }
            save_dual_word_bar(
                sentence_dir / f"{sentence_id}_dual_class_word_bar",
                words=words,
                values=word_values,
                scores=scores,
                base_values=base_values,
                residuals=additivity["residual"],
                **common,
            )
            save_class_value_matrix(
                sentence_dir / f"{sentence_id}_token_class_matrix",
                row_labels=token_labels,
                values=shap_values,
                level_name="Token",
                **common,
            )
            save_class_value_matrix(
                sentence_dir / f"{sentence_id}_word_class_matrix",
                row_labels=word_labels,
                values=word_values,
                level_name="Word",
                **common,
            )
            save_sentence_summary_matrix(
                sentence_dir / f"{sentence_id}_summary_matrix",
                scores=scores,
                softmax_shares=shares,
                base_values=base_values,
                shap_values=shap_values,
                reconstructed=additivity["reconstructed"],
                residuals=additivity["residual"],
                **common,
            )
            save_sentence_overview(
                sentence_dir / f"{sentence_id}_presentation_overview",
                words=words,
                word_values=word_values,
                scores=scores,
                softmax_shares=shares,
                base_values=base_values,
                shap_values=shap_values,
                reconstructed=additivity["reconstructed"],
                residuals=additivity["residual"],
                **common,
            )

    write_csv(values_dir / "sentence_summary.csv", summary_rows)
    write_csv(values_dir / "token_shap_values.csv", token_rows)
    write_csv(values_dir / "word_shap_values.csv", word_rows)
    write_csv(values_dir / "word_shap_values_wide.csv", wide_word_rows)
    write_csv(values_dir / "masking_diagnostics.csv", masking_rows)

    faithfulness_summary = summarise_faithfulness(faithfulness_rows)
    write_csv(evaluation_dir / "faithfulness_deletion_rows.csv", faithfulness_rows)
    write_csv(evaluation_dir / "faithfulness_aopc_summary.csv", faithfulness_summary)

    comparisons: list[dict[str, Any]] = []
    bert_values_dir = experiment_path(settings["bert_values_path"])
    if not args.no_bert_comparison and bert_values_dir.exists():
        comparisons = compare_runs(summary_rows, wide_word_rows, bert_values_dir)
    write_csv(evaluation_dir / "bert_clip_ranking_comparison.csv", comparisons)

    if not args.no_plots:
        save_all_sentence_summary_matrix(
            plots_dir / "all_sentences_summary_matrix", summary_rows
        )
        save_sentence_class_magnitude_matrix(
            plots_dir / "sentence_class_shap_magnitude_matrix", summary_rows
        )
        save_global_word_class_matrix(
            plots_dir / "global_word_class_importance_matrix", word_rows
        )
        save_additivity_residual_matrix(
            plots_dir / "class_additivity_residual_matrix", summary_rows
        )
        save_class_comparison_bars(
            plots_dir / "global_signed_class_contributions", word_rows
        )
        if faithfulness_rows:
            save_faithfulness_curve(
                evaluation_dir / "faithfulness_deletion_curves", faithfulness_rows
            )
            save_aopc_matrix(
                evaluation_dir / "faithfulness_aopc_matrix", faithfulness_summary
            )
        if comparisons:
            save_comparison_matrix(
                evaluation_dir / "bert_clip_ranking_comparison_matrix", comparisons
            )

    run_config["total_runtime_seconds"] = time.time() - start_time
    run_config["total_model_text_evaluations"] = model_function.text_evaluations
    run_config["total_forward_batches"] = model_function.forward_batches
    run_config["proxy_correct_count"] = sum(int(row["correct"]) for row in summary_rows)
    run_config["maximum_absolute_additivity_residual"] = max(
        max(
            abs(float(row["negative_additivity_residual"])),
            abs(float(row["positive_additivity_residual"])),
        )
        for row in summary_rows
    )
    (values_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    report = build_report(summary_rows, faithfulness_summary, comparisons, run_config)
    (reports_dir / "run_report.md").write_text(report, encoding="utf-8")

    print(f"Done: {values_dir.name}")
    print(f"Values: {values_dir.resolve()}")
    print(f"Plots: {plots_dir.resolve()}")
    print(f"Evaluation: {evaluation_dir.resolve()}")
    print(f"Report: {reports_dir.resolve()}")
    print(
        "Maximum class additivity residual: "
        f"{run_config['maximum_absolute_additivity_residual']:.3e}"
    )


if __name__ == "__main__":
    main()

