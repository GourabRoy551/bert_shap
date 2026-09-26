"""Deletion masking for CLIP, which has no native mask token."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def create_text_masker(tokenizer: Any, mask_token: str = "") -> Any:
    import shap

    return shap.maskers.Text(
        tokenizer,
        mask_token=mask_token,
        collapse_mask_token=True,
    )


def apply_feature_mask(masker: Any, text: str, kept: Sequence[bool]) -> str:
    result = masker(np.asarray(kept, dtype=bool), text)
    values = result[0] if isinstance(result, tuple) else result
    return str(np.asarray(values, dtype=object).reshape(-1)[0])


def masking_diagnostics(masker: Any, text: str) -> dict[str, object]:
    feature_count = int(masker.shape(text)[1])
    all_kept = np.ones(feature_count, dtype=bool)
    none_kept = np.zeros(feature_count, dtype=bool)
    return {
        "feature_count": feature_count,
        "fully_unmasked_text": apply_feature_mask(masker, text, all_kept),
        "fully_masked_text": apply_feature_mask(masker, text, none_kept),
    }
