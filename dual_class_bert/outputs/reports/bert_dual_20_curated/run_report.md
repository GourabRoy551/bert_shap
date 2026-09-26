# BERT Dual-Class SHAP Run Report

## Experiment

- Model: textattack/bert-base-uncased-SST-2
- Explained outputs: NEG logit and POS logit
- Explainer: Partition SHAP with the BERT text masker
- Sentences: 20
- Correct predictions: 20/20
- Maximum NEG additivity residual: 1.776e-15
- Maximum POS additivity residual: 8.882e-16
- Maximum evaluations per sentence: 500

## Sentence results

| ID | Gold | Pred. | NEG logit | POS logit | Top NEG increase | Top POS increase |
|---|---:|---:|---:|---:|---|---|
| S1 | POS | POS | -4.1938 | 4.0527 | of (+0.1058) | beautiful (+1.1037) |
| S2 | NEG | NEG | 3.8285 | -3.3188 | waste (+3.0587) | and (+1.0829) |
| S3 | POS | POS | -3.5206 | 3.4425 | slow (+0.5956) | outstanding (+1.8437) |
| S4 | NEG | NEG | 3.7820 | -3.4976 | dull (+2.1185) | a (+0.1452) |
| S5 | POS | POS | -4.1543 | 4.0123 | the (+0.2730) | brilliant (+1.9438) |
| S6 | POS | POS | -4.1971 | 4.0228 | emotion (+0.9633) | breathtaking (+0.8913) |
| S7 | NEG | NEG | 3.8791 | -3.4845 | boring (+1.8844) | charm (+0.7449) |
| S8 | POS | POS | -3.5065 | 3.4310 | weak (+2.1135) | captivating (+2.3337) |
| S9 | NEG | NEG | 3.7930 | -3.3585 | hollow (+1.9841) | a (+0.2269) |
| S10 | POS | POS | -4.3161 | 4.0646 | to (+0.2217) | entertaining (+1.3631) |
| D01 | NEG | NEG | 3.4692 | -3.0463 | disappointingly (+2.1804) | more (+0.1576) |
| D02 | POS | POS | -4.2102 | 4.0334 | it (+0.3256) | grand (+0.5495) |
| D03 | NEG | NEG | 3.6049 | -3.3456 | bad (+1.0789) | when (+0.0849) |
| D04 | POS | POS | -4.2597 | 4.0685 | serry (+0.1216) | effective (+0.4213) |
| D05 | NEG | NEG | 3.1897 | -3.0792 | boring (+1.9484) | about (+0.3607) |
| D06 | POS | POS | -4.0629 | 3.9532 | brooding (+1.0908) | compelling (+1.7124) |
| D07 | NEG | NEG | 3.8316 | -3.4158 | hackneyed (+1.4294) | visual (+0.3446) |
| D08 | POS | POS | -4.2348 | 4.0577 | accessible (+0.2297) | powerful (+1.9980) |
| D09 | NEG | NEG | 3.7212 | -3.3083 | meaningless (+1.8118) | watching (+0.1961) |
| D10 | NEG | NEG | 2.7701 | -2.3037 | awful (+3.6012) | god (+0.4034) |

## Existing margin comparison

For S1-S10, POS SHAP minus NEG SHAP was compared with the saved scalar POS-logit-minus-NEG-logit explanation.

- Maximum absolute token difference: 2.306e-02

## Interpretation

A positive value in the NEG column increases the NEG logit; a positive value in the POS column increases the POS logit. A negative value decreases the corresponding target-class logit.

All matrix cells display raw numerical values. Sentence-summary matrix colours are normalized within a row or column only to make different metric scales readable.
