# BERT Dual-Class SHAP Run Report

## Experiment

- Model: textattack/bert-base-uncased-SST-2
- Explained outputs: NEG logit and POS logit
- Explainer: Partition SHAP with the BERT text masker
- Sentences: 2
- Correct predictions: 2/2
- Maximum NEG additivity residual: 8.882e-16
- Maximum POS additivity residual: 8.882e-16
- Maximum evaluations per sentence: 100

## Sentence results

| ID | Gold | Pred. | NEG logit | POS logit | Top NEG increase | Top POS increase |
|---|---:|---:|---:|---:|---|---|
| S1 | POS | POS | -4.1938 | 4.0527 | of (+0.0583) | beautiful (+1.0294) |
| S2 | NEG | NEG | 3.8285 | -3.3188 | waste (+1.8682) | and (+1.0829) |

## Existing margin comparison

For S1-S10, POS SHAP minus NEG SHAP was compared with the saved scalar POS-logit-minus-NEG-logit explanation.

- Maximum absolute token difference: 2.225e+00

## Interpretation

A positive value in the NEG column increases the NEG logit; a positive value in the POS column increases the POS logit. A negative value decreases the corresponding target-class logit.

All matrix cells display raw numerical values. Sentence-summary matrix colours are normalized within a row or column only to make different metric scales readable.
