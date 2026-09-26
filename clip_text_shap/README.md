# CLIP Text Encoder Dual-Class SHAP Experiment

This independent experiment explains **both sentiment targets for every sentence**. It does not modify the existing BERT/RFEM projects, and it does not use the CLIP Vision Encoder.

## What is being explained

CLIP is not a sentiment classifier, so fixed negative and positive text prototypes are constructed from the balanced prompt sets in `prompts.json`. For each class (c\in\{NEG,POS\}), the normalized prompt embeddings are averaged and normalized:

\[
p_c = \frac{\frac{1}{K}\sum_{k=1}^{K} e(t_{c,k})}
{\left\|\frac{1}{K}\sum_{k=1}^{K} e(t_{c,k})\right\|_2}.
\]

For an input sentence (x), the explained score is its cosine similarity to the class prototype:

\[
s_c(x)=e(x)^\top p_c.
\]

The model wrapper always returns two columns in the fixed order `[NEG, POS]`. The class with the larger score is the proxy prediction. A two-score softmax is saved only as a descriptive share; it is **not** called a calibrated sentiment probability.

Partition SHAP uses CLIP's token hierarchy and estimates a contribution \(\phi_{i,c}\) for token feature \(i\) and target class \(c\):

\[
s_c(x) \approx \phi_{0,c} + \sum_{i=1}^{M}\phi_{i,c}.
\]

CLIP has no native mask token, so unavailable features are deleted with an empty-string text masker. Raw CLIP BPE contributions are saved and then summed into readable word groups. The class-margin identity is also checked:

\[
s_{POS}(x)-s_{NEG}(x)
= (\phi_{0,POS}-\phi_{0,NEG})
+ \sum_i(\phi_{i,POS}-\phi_{i,NEG}).
\]

## Project structure

```text
clip_text_shap/
├── README.md
├── requirements.txt
├── config.json
├── prompts.json
├── run_clip_text_shap.bat
├── data/
│   ├── sentences_20_curated.csv
│   └── dataset_manifest.json
├── src/
│   ├── prompt_prototypes.py
│   ├── clip_text_wrapper.py
│   ├── masking_strategy.py
│   ├── token_aggregation.py
│   ├── run_clip_text_shap.py
│   ├── make_plots.py
│   ├── evaluate_faithfulness.py
│   └── compare_with_bert.py
├── tests/
└── outputs/
    ├── values/
    ├── plots/
    ├── evaluation/
    └── reports/
```

## Outputs

For every sentence, the runner saves these figures in both PNG and PDF:

1. signed NEG/POS word bars;
2. token-by-class SHAP matrix;
3. word-by-class SHAP matrix;
4. sentence-summary numerical matrix;
5. combined presentation overview.

The run also produces five annotated global figures, two annotated faithfulness figures, and an annotated BERT–CLIP ranking-comparison matrix. Every bar or heatmap cell displays its numerical value.

CSV tables preserve the sentence scores, base values, SHAP sums, reconstructions, residuals, raw token values, aggregated word values, masking diagnostics, prompt diagnostics, deletion metrics, and BERT comparison metrics.

## Faithfulness metric

For each sentence and target class, content words with the largest positive class-specific SHAP values are deleted for \(k\in\{1,2,3\}\). Their target-score drop is compared with the mean drop from ten equally sized random deletions. The area over the perturbation curve is reported as:

\[
AOPC_c = \frac{1}{|K|}\sum_{k\in K}
\left[s_c(x)-s_c(x_{\setminus top-k})\right].
\]

A positive value means the selected words supported the target score. `aopc_improvement` is the SHAP-ranked AOPC minus the random-deletion AOPC.

## BERT comparison

Raw BERT and CLIP SHAP magnitudes are not compared because BERT explains logits while CLIP explains cosine similarities. The comparison instead uses the POS-minus-NEG word contribution, reporting sentence-level Spearman rank correlation, top-three overlap/Jaccard, sign agreement, and prediction agreement.

## Run

The included Windows launcher uses the already available local model and Python environment:

```bat
run_clip_text_shap.bat
```

To run a smaller check manually, use the same environment variables as the launcher and run:

```text
python src/run_clip_text_shap.py --run-name smoke --max-sentences 2 --max-evals 100
```

The runner refuses to overwrite an existing run name. It also verifies the approved dataset hash and class counts before loading the model.

## Tests

The tests cover dataset parity, prompt validation, word aggregation, dual-class margin arithmetic, wrapper parity, and one-sentence SHAP additivity:

```text
python -m unittest discover -s tests
```

