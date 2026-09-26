# CLIP Text Encoder Dual-Class SHAP Run Report

## Experiment

- Model: `openai/clip-vit-base-patch32`
- Explained outputs: cosine similarity to fixed NEG and POS prompt prototypes
- Explainer: Partition SHAP with deletion masking (CLIP has no mask token)
- Sentences: 20
- Proxy-classification accuracy: 17/20
- Maximum class additivity residual: 0.000e+00
- Maximum evaluations per sentence: 500
- The displayed softmax share is descriptive and is not a calibrated probability.

## Sentence results

| ID | Gold | Pred. | NEG score | POS score | Top NEG increase | Top POS increase |
|---|---:|---:|---:|---:|---|---|
| S1 | POS | POS | 0.9213 | 0.9338 | film (+0.0310) | film (+0.0402) |
| S2 | NEG | NEG | 0.9794 | 0.9771 | movie (+0.0500) | movie (+0.0593) |
| S3 | POS | NEG | 0.8976 | 0.8952 | but (+0.0155) | but (+0.0162) |
| S4 | NEG | NEG | 0.9157 | 0.9130 | forgettable (+0.0171) | forgettable (+0.0188) |
| S5 | POS | POS | 0.9320 | 0.9388 | The (+0.0260) | The (+0.0249) |
| S6 | POS | POS | 0.9193 | 0.9276 | masterpiece (+0.0201) | masterpiece (+0.0238) |
| S7 | NEG | NEG | 0.9097 | 0.8979 | Painfully (+0.0141) | and (+0.0111) |
| S8 | POS | POS | 0.9363 | 0.9435 | but (+0.0211) | but (+0.0216) |
| S9 | NEG | NEG | 0.9552 | 0.9498 | sequel (+0.0577) | sequel (+0.0619) |
| S10 | POS | POS | 0.9245 | 0.9337 | Funny (+0.0228) | entertaining (+0.0245) |
| D01 | NEG | POS | 0.9070 | 0.9184 | movie (+0.0345) | movie (+0.0433) |
| D02 | POS | POS | 0.9060 | 0.9116 | of (+0.0236) | of (+0.0235) |
| D03 | NEG | NEG | 0.9491 | 0.9490 | movie (+0.0300) | movie (+0.0350) |
| D04 | POS | POS | 0.9071 | 0.9177 | effective (+0.0226) | writer-director (+0.0229) |
| D05 | NEG | NEG | 0.9575 | 0.9514 | movie (+0.0568) | movie (+0.0648) |
| D06 | POS | POS | 0.9322 | 0.9331 | compelling (+0.0234) | compelling (+0.0257) |
| D07 | NEG | POS | 0.9173 | 0.9283 | film's (+0.0341) | film's (+0.0429) |
| D08 | POS | POS | 0.9550 | 0.9692 | film (+0.0346) | film (+0.0453) |
| D09 | NEG | NEG | 0.9085 | 0.8897 | this (+0.0253) | got (+0.0263) |
| D10 | NEG | NEG | 0.9183 | 0.9003 | god-awful (+0.0131) | this (+0.0119) |

## Faithfulness

Positive AOPC means that deleting SHAP-ranked words lowers the target similarity. The random baseline deletes the same number of content words.

- Mean SHAP top-k deletion AOPC: +0.0403
- Mean random deletion AOPC: +0.0090
- Mean SHAP-minus-random AOPC improvement: +0.0313
- Positive sentence/class improvements: 40/40

## BERT comparison

BERT and CLIP are compared by POS-minus-NEG word-contribution rankings and directions. Raw magnitudes are not compared because the explained outputs are BERT logits versus CLIP cosine similarities.

- Mean sentence-level Spearman rank correlation: +0.277
- Mean top-3 Jaccard overlap: 0.380
- Mean margin-sign agreement: 0.580
- Prediction agreement: 17/20

## Interpretation

A positive SHAP value raises the named class-prototype similarity relative to the deletion-masked baseline; a negative value lowers it. Each sentence has values for both NEG and POS, irrespective of its gold or predicted class.

All heatmap cells and bars display their numerical values, and every figure is saved as both PNG and PDF.

## Limitation

CLIP is a prompt-based sentiment proxy here, not a fine-tuned sentiment classifier. Shared prompt-anchor words such as *movie* and *film* can raise both similarities, so POS-minus-NEG contributions should be inspected for class discrimination. This 20-sentence run is not sufficient for a broad claim that CLIP is better than BERT or RFEM.
