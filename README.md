# When Does Factorization Enable OCR? An Optimizer Stress Test

A completed, reproducible mini-project testing optimizer sensitivity in a controlled fixed-attention symbolic OCR task. All results in this repository are measured, not illustrative. See [the two-page research note](research_note.pdf) and [raw results](results/summary.csv).

## Main results

Mean hidden-implication accuracy [minimum, maximum across five initialization seeds]:

| Optimizer | Direct, CE <= 0.01 | Factorized, CE <= 0.01 | Direct, 20k steps | Factorized, 20k steps |
|---|---|---|---|---|
| gd | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] |
| adam | 3.0% [0.0, 6.7] | 87.7% [78.3, 95.0] | 6.3% [1.7, 10.0] | 100.0% [100.0, 100.0] |
| adamw0.01 | 3.0% [0.0, 6.7] | 87.7% [78.3, 95.0] | 6.3% [3.3, 8.3] | 100.0% [100.0, 100.0] |
| adamw0.03 | 3.3% [0.0, 6.7] | 88.7% [78.3, 95.0] | 4.7% [3.3, 6.7] | 99.7% [98.3, 100.0] |

The factorization advantage persists across the tested optimizers. At CE <= 0.01, factorized Adam trails factorized GD by 12.3 percentage points (87.7% versus 100%), but both reach 100% at 20,000 updates. Adam has a lower centered spectral ratio (4.121 versus 4.287) despite lower OCR at matched fit, so the results do not support a simple lower-ratio-means-better-OCR rule. Actual first-crossing CE spans 0.00964-0.01000 across conditions. All 40 substantive runs reach all three training CE thresholds, and all finish with 100% training accuracy. Direct models' small nonzero OCR under Adam should not be interpreted as robust reasoning. The task has 40 candidates (uniform chance 2.5%); selecting only implication candidates uniformly would give 5%, a different baseline.

These are descriptive paired-seed comparisons, not significance tests. The first-crossing checkpoint is the first discrete update with CE <= the target, not an interpolated exact-loss solution. See `summary.csv` for actual CE and step values. We do not select learning rates by OCR performance.

## Reproduce

Python 3.11 was used on an Apple Silicon CPU. The complete pinned environment is in `requirements.txt`; software and hardware metadata are in `results/environment.json`. No GPU, API keys, downloaded datasets or services are required. Installation needs internet access.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python checks.py
python sweep.py
python verify_results.py
python plot.py
python report.py
```

The default sweep overwrites the bundled results. To preserve them, run `python sweep.py --results results/reproduction` and then `python plot.py --results results/reproduction --figures figures/reproduction`. The report generator uses the default `results/aggregates.json` and `figures/` paths. The recorded training loops took about 181 seconds in total on this machine (excluding setup, figure generation and audits). CPU runtime depends on hardware; measured per-run times are included in every JSON record. Running all experiments involves 1,060,000 full-batch updates: twelve 5k-step pilot runs plus fifty 20k-step main/check runs.

A single smoke run:

```bash
python train.py --kind factorized --optimizer gd --lr 1 --steps 500 --output results/smoke.json
```

## Experimental controls

- **Data:** 20 groups, four subjects each, one observed implication per group. Training has 80 facts and 20 implications; evaluation has 60 hidden implications. All 40 answers are candidates. Subjects have independently permuted token IDs. Group labels never enter the model. The split is fixed (data seed 1729); training facts for evaluation subjects are intentionally included, while their implications are withheld. `results/dataset.json` stores every triple.
- **Models:** x = e_subject + e_relation; direct logits Wx versus factorized logits AB^T x. Only the active 40 output rows and 82 input columns exist. Inner width 40 is full expressivity for a 40-by-82 map. There are no biases or learned attention weights.
- **Initialization:** independent Gaussian W0 entries with standard deviation 0.01, seeds 0-4; direct W0 versus balanced SVD factors. Paired initial functions agree below 1e-6 across all 160 queries. SVD chooses a particular latent basis: coordinate-adaptive Adam need not be invariant to other equivalent factor rotations. Seeds vary initialization, not the data mapping.
- **Optimization:** full-batch SGD with momentum zero is called GD; Adam and AdamW use betas (0.9, 0.999), epsilon 1e-8, no scheduler, clipping or early stopping. CPU float64, one thread and deterministic algorithms. Equal update budgets do not mean equal compute or equivalent update sizes.
- **Pilot:** seed 100 is excluded from final seed summaries. Twelve candidates, all saved in `results/pilot/`. GD rates 0.01, 0.1, 1; Adam rates 0.0001, 0.001, 0.01; 5,000 steps each. Select the stable candidate with lowest final training CE, independently per parameterization. Selected GD = 1, Adam = 0.01. AdamW inherits Adam's rate, including nonzero decay. This is a deliberately limited grid, not a claim of optimal tuning.
- **Sweep:** GD/Adam at zero decay (20 runs); AdamW at 0.01/0.03 (20 runs); AdamW at zero decay (10 equivalence runs). Equivalence runs are excluded from figures/aggregates to avoid double counting.
- **Checkpointing:** CE 0.1, 0.03, 0.01 checked every update; first crossing saved. Continue until 20,000 updates. Histories logged every 100 updates plus crossing/final steps. Each `.pt` contains model states at crossings and final, not optimizer state for exact resume. Missing thresholds would be absent from `crossings` and marked `reached=False` in the summary.

## Metrics and figures

Training/test CE and accuracy; minimum training margin; mean and minimum hidden-implication margin. A margin is the correct logit minus the maximum incorrect logit. Center each active matrix over answer rows: Wc = W - W.mean(axis=0). Report its Frobenius and nuclear norms and R = ||Wc||_* / ||Wc||_F. R is scale invariant; R^2 is a spectral effective dimension, not literal rank. At positive minimum training margin gamma, also report ||Wc||_*/gamma and ||Wc||_F/gamma.

1. [OCR versus training CE](figures/01_accuracy_vs_training_loss.png): seed trajectories and threshold means; CE decreases to the right. Curves below 1e-5 are clipped from this view.
2. [Parameterization and optimizer](figures/02_parameterization_optimizer.png): all seed points and means at CE <= 0.01 and 20k updates. Small horizontal offsets separate seeds; vertical accuracy is not jittered.
3. [Spectrum versus OCR](figures/03_spectrum_vs_accuracy.png): one matched-fit checkpoint per substantive run. Overlapping points reflect similar values; they are not extra independent replications.

SVG versions are included for editing. `results/summary.csv` is the tidy per-seed table; `aggregates.json` is derived from it/the same raw logs. Raw run JSONs include configuration, metrics, crossings and elapsed time. `checks.json` and `final_audit.json` document verification. `verify_results.py` reloads all saved model checkpoints, recomputes every metric and checks complete Adam/zero-decay AdamW trajectories for equality.

## Interpretation and limitations

The measured factorization advantage is robust across this chosen GD/Adam/AdamW matrix. This does not show that all optimizers behave alike, or that a nuclear-norm mechanism causes the observed behavior. Spectral concentration does not identify the correct fact-group structure; low rank alone does not guarantee OCR. No exact SVM solution, numerical nuclear-norm optimum or direction-similarity reference is computed in this version.

Equal AdamW coefficients do not imply equal functional shrinkage: factor decay alone multiplies AB^T by (1 - lr * wd)^2, while direct W shrinks by (1 - lr * wd). Thus decay comparisons combine regularization and optimizer dynamics. They are not an isolated test of adaptive preconditioning. Matching functions also does not match parameter-space geometry, which is precisely the parameterization intervention.

Fixed attention, small noiseless data, one split, five seeds, one pilot seed, finite training and a restricted rate grid limit generalization. Balanced random matched initialization differs from special symmetric initializations and asymptotic theorem assumptions. No causal mechanism is proven and no realistic LLM experiment is included. Minor cross-platform numerical differences, especially SVD bases and argmax near ties, may change results.

## Relation to the paper

Huang et al., *Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers*, [arXiv:2506.10887v4](https://arxiv.org/html/2506.10887v4), especially Section 4.1, Appendix A.3 and Appendix E. Section 4.1 supplies the equal fixed-attention logit model. Appendix E already reports AdamW use, so this project does **not** claim to introduce AdamW to the paper's experiments. This is an independent compact implementation inspired by the paper and the supplied experiment brief, not a reproduction of the authors' full one-layer transformer or LLM runs. The supplied brief's specific author-code commit claim was not needed or independently audited.

## Files

`data.py`, `models.py`, `train.py`, `sweep.py`, `plot.py`: core pipeline. `configs/experiment.json`: experiment grid. `checks.py` and `verify_results.py`: scientific validation. `report.py`: regenerated two-page PDF plus editable text. `results/`: raw and derived measurements. `figures/`: three figures as PNG/SVG. `research_note.pdf`: final report; `research_note.md`: editable prose, table and figure-link companion. `requirements.txt`: installed dependency lock. `MANIFEST.sha256`: delivery checksums.
