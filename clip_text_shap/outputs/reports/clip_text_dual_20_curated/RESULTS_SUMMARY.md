# Results Summary: CLIP Text Encoder Dual-Class SHAP

## Outcome

The experiment was technically successful. Partition SHAP produced a separate NEG and POS attribution for every token and aggregated word in all 20 approved sentences. Both explained scores reconstructed the CLIP wrapper outputs exactly within the saved floating-point precision: the maximum observed additivity residual was `0.000e+00`.

This result confirms that the implementation can plot and tabulate Shapley values for **both sentiment classes for every sentence**, rather than explaining only the sentence's actual or predicted class.

## Quantitative results

| Measure | Result |
|---|---:|
| Sentences explained | 20 |
| NEG/POS explanations | 40 |
| Proxy predictions matching the gold label | 17/20 (85%) |
| Maximum class additivity residual | 0.000e+00 |
| Mean SHAP top-k deletion AOPC | 0.0403 |
| Mean random deletion AOPC | 0.0090 |
| Mean SHAP-minus-random AOPC improvement | +0.0313 |
| Sentence/class cases with positive AOPC improvement | 40/40 |
| Mean BERT–CLIP margin-rank Spearman correlation | +0.277 |
| Mean BERT–CLIP top-3 Jaccard overlap | 0.380 |
| Mean BERT–CLIP contribution-sign agreement | 0.580 |
| BERT–CLIP prediction agreement | 17/20 |

The three CLIP proxy-prediction errors were `S3` (gold POS, predicted NEG), `D01` (gold NEG, predicted POS), and `D07` (gold NEG, predicted POS).

## Interpretation

The positive faithfulness result is the clearest validation of the explanation procedure. Across all 40 sentence/class combinations, deleting words selected by positive class-specific SHAP values reduced the relevant CLIP target score more than deleting the same number of random content words. The mean improvement over the random baseline was `+0.0313` cosine-similarity units.

The BERT–CLIP agreement is modest rather than strong. A mean Spearman correlation of `+0.277` shows that the models often rank word contributions differently. This is plausible because BERT was fine-tuned for sentiment classification, while CLIP is a multimodal contrastive model and the present experiment turns it into a sentiment proxy through text prompts.

Several plots show neutral prompt-anchor words such as *movie* or *film* contributing strongly to both class similarities. These words are shared by the class prompts and therefore raise general text-to-prompt similarity without necessarily deciding the sentiment margin. For class discrimination, the POS-minus-NEG contribution in `word_shap_values_wide.csv` should be considered alongside the two raw class columns.

## What the result does and does not establish

The run establishes that:

- the CLIP Text Encoder can be wrapped as a reproducible two-score sentiment proxy;
- Partition SHAP can explain NEG and POS scores simultaneously;
- raw BPE values can be aggregated into readable words without losing class additivity;
- the saved explanations pass a top-k deletion faithfulness check on this 20-sentence set;
- the numerical results and figures are available for presentation and later reporting.

The run does **not** establish that CLIP is a better sentiment classifier than BERT, or that CLIP explanations detect sentiment words better than BERT/RFEM. The evaluation set is small, the prompt-based class scores are not calibrated probabilities, and important shared prompt nouns can dominate the raw class-specific values. A larger evaluation (for example, the proposed 500 samples) would be needed for a comparative performance claim.

## Output inventory

- `sentence_summary.csv`: sentence scores, reconstruction checks, margins, and top words.
- `token_shap_values.csv`: raw CLIP BPE contributions for NEG and POS.
- `word_shap_values.csv`: long-format word contributions.
- `word_shap_values_wide.csv`: NEG, POS, and POS-minus-NEG word contributions.
- `faithfulness_deletion_rows.csv` and `faithfulness_aopc_summary.csv`: top-k and random deletion metrics.
- `bert_clip_ranking_comparison.csv`: ranking, top-3 overlap, sign, and prediction agreement.
- 108 PNG figures and 108 corresponding PDF figures, with no missing or empty pairs.

