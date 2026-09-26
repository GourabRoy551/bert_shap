# CLIP Text Encoder Dual-Class SHAP Run Report

## Experiment

- Model: `openai/clip-vit-base-patch32`
- Explained outputs: cosine similarity to fixed NEG and POS prompt prototypes
- Explainer: Partition SHAP with deletion masking (CLIP has no mask token)
- Sentences: 2
- Proxy-classification accuracy: 2/2
- Maximum class additivity residual: 0.000e+00
- Maximum evaluations per sentence: 100
- The displayed softmax share is descriptive and is not a calibrated probability.

## Sentence results

| ID | Gold | Pred. | NEG score | POS score | Top NEG increase | Top POS increase |
|---|---:|---:|---:|---:|---|---|
| S1 | POS | POS | 0.9213 | 0.9338 | film (+0.0303) | film (+0.0395) |
| S2 | NEG | NEG | 0.9794 | 0.9771 | movie (+0.0477) | movie (+0.0573) |

## Faithfulness

Positive AOPC means that deleting SHAP-ranked words lowers the target similarity. The random baseline deletes the same number of content words.

- Mean SHAP-minus-random AOPC improvement: +0.0552

## BERT comparison

BERT and CLIP are compared by POS-minus-NEG word-contribution rankings and directions. Raw magnitudes are not compared because the explained outputs are BERT logits versus CLIP cosine similarities.

- Mean sentence-level Spearman rank correlation: +0.247

## Interpretation

A positive SHAP value raises the named class-prototype similarity relative to the deletion-masked baseline; a negative value lowers it. Each sentence has values for both NEG and POS, irrespective of its gold or predicted class.

All heatmap cells and bars display their numerical values, and every figure is saved as both PNG and PDF.
