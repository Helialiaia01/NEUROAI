# Multiobjective synthetic benchmark — 21 September 2026

## Purpose

Test the official CEBRA 0.6.0 multiobjective implementation before deciding
whether to build a trial-aware IBL integration. This is a separate prototype;
existing IBL results still come from independent per-variable regularized CEBRA.
No reserved IBL sessions or published RRR results are used here.

## Frozen design

`scripts/benchmark_multiobjective.py` compares two predefined observation graphs:
one signal neuron per latent plus ten noise neurons (anchor), and six signal
neurons per latent (redundant). Two independent autoregressive latent variables
drive a tanh observation model with additive Gaussian noise. Train, validation
and test are separate continuous sequences of 1,200/400/400 samples, each with
independent burn-in. Neural scaling and latent scaling use training data only.
Data seed is 872; training seeds are 2025 and 2026.

Both families use offset10 encoders, 32 hidden units, batch size 64, Adam learning
rate 0.0003, temperature 1, and 1,000 steps per encoder. The adaptation trains
two separate 2D encoders with constant Jacobian regularization 0.01 and its
existing label-difference lag 10. The candidate uses the official
`ContrastiveMultiObjectiveLoader`, `MultiObjectiveConfig` and multiobjective
solver with two behavior objectives (slices 0:2 and 2:4, label-difference lag 1),
plus a temporal objective (slice 0:8, time offset 10). Disjoint output blocks are
renormalized; regularization ramps from zero to 0.01 between steps 250 and 500.
Temporal positives outside the synthetic recording are clamped at its boundary.
This continuous-sequence adapter is not an implementation of IBL trial boundaries.

This is a comparison of model families, not a controlled isolation of one
architectural change: capacity, label lag, objectives, regularization schedule
and total compute differ. It is not a reproduction of the paper's benchmark.

## Evaluation and safeguards

Only receptive-field-interior samples are decoded. Ridge hyperparameters are
selected on validation, then evaluated on test. AUROC and average precision
measure neuron support recovery against the known graph. AUROC chance is 0.5;
positive prevalence is 1/12 for anchor and 1/2 for redundant (the average-precision
baseline depends on the ranking distribution at this small sample size).

For both families, attribution uses a double-precision copy of the trained
encoder, signed temporal-mean Jacobians, full pseudoinversion at relative cutoff
1e-5, then variable-slice selection and mean absolute aggregation. The first 256
interior test windows are identical across methods and seeds. Double precision
avoids treating roundoff in block-normalization null directions as neuron signal.

Outputs under `outputs/multiobjective_benchmark_seeded_20260921/` contain the frozen
design, data, model weights, losses, attribution vectors and incremental/final
metrics. Outputs are Git-ignored. Training runs on CPU and uses no cluster jobs.
Two seeds and one data realization cannot establish robustness or identifiability.
A successful synthetic result alone would not authorize biological conclusions
or full-cohort training.

## Verification

The four-step smoke exercised both training families, decoding, inverse-Jacobian
attribution and artifact writing. The suite has 42 passing tests, including an
analytic full-inverse-before-slice check and training-only scaling check.
The installed official Jacobian regularizer emits a PyTorch deprecated-call
warning; it is not a failed fit or numerical warning.

## Sampler reproducibility correction

A repeated smoke initially failed exact reproducibility for both families.
Inspection of installed CEBRA 0.6 `distributions/base.py` showed that
`HasGenerator.__init__` calls `generator.seed()` regardless of the supplied seed.
Global NumPy/PyTorch seeds therefore left private distribution generators random.
`xcebra_ibl.models.randomness.seed_loader_generators` now seeds the loader's
nested distributions and priors explicitly, including their saved device-transfer
seed. Both the production per-variable fit and this prototype use the correction.
Two fresh smoke processes then produced exactly identical metrics and attribution
arrays. A regression test verifies repetition and sensitivity to a changed seed.

The earlier interrupted benchmark directory is exploratory and superseded; it
must not be pooled with corrected outputs. Existing IBL fits are still completed
observations, but their claimed deterministic replay was not established. This
does not by itself explain or repair poor cross-model attribution stability.
