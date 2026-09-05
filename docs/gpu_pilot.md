# GPU pilot protocol — 5 September 2026

## Purpose and method decision

The executable pilot addresses held-out behavioural decoding and representation
stability, the first and third thesis research questions. Use the name
**regularized per-variable CEBRA adaptation**. Each variable has an independent
supervised encoder with Jacobian regularization and inverted-gradient
attribution. This is not canonical multiobjective xCEBRA, whose official demo
uses multiple objectives and output slices in a shared encoder:
https://cebra.ai/docs/demo_notebooks/Demo_xCEBRA_RatInABox.html

Attributions describe sensitivity in training-standardized neural coordinates.
They are neither causal effects nor unique contributions of correlated labels,
and their magnitudes are not directly comparable to RRR coefficients.

## Run sequence

From the repository root, first measure runtime and memory on one session:

```sh
python -m xcebra_ibl.experiments --max-sessions 1 --seeds 2025 --dimensions 4 --iterations 500 --output outputs/gpu_calibration
```

Then run the controlled pilot:

```sh
python -m xcebra_ibl.experiments --max-sessions 3 --seeds 2025 2026 2027 --dimensions 2 4 8 --iterations 500 --output outputs/gpu_pilot
```

Kaggle now defaults to the one-session calibration run. `KAGGLE_PIPELINE_ARGS`
overrides its arguments. For the full controlled grid, set it to
`--max-sessions 3 --seeds 2025 2026 2027 --dimensions 2 4 8 --iterations 500`.
`KAGGLE_EXPERIMENT_MODE=legacy` explicitly restores the previous pipeline CLI.
The pilot does not require the unavailable original RRR result export.

The full pilot grid is **432 encoder fits**, including controls, not three fits.
Use measured per-fit runtime before committing GPU allocation. Checkpoints are
written every 100 iterations, retaining one per variable; final models are also
saved. Completed seed/dimension/control combinations are reused only when the
manifest (code, inputs, configuration and runtime) matches. Interrupted
combinations restart; optimizer-step resume is not implemented. A changed
configuration requires a new output directory. Existing results are preserved.

Only increase iterations toward 10,000 after checking loss trajectories,
validation scores, observed-versus-null decoding, seed stability, runtime and
memory. Do not choose iteration count by repeatedly inspecting test scores.
For confirmatory evaluation after pilot-driven choices, use new sessions or a
separately reserved test set. The selected first three sorted sessions are an
engineering sample, not a representative brain-wide sample.

## Scientific controls and outputs

- A fixed 60/20/20 split of retained trials is shared by all seeds and dimensions.
  Neuron inclusion, movement lag fitting, and normalization use training trials
  only. Smoothing stays within trials. Raw trial/neuron indices and fitted
  preprocessing parameters are archived.
- Neural-context Ridge/logistic decoding baselines receive the same
  neurons and receptive-field windows as the encoders. Embeddings receive both
  linear and kNN decoders. Raw-neural kNN is omitted to avoid a costly quadratic
  distance computation in thousands of input dimensions. Decoding evaluation uses
  interior bins. Scaling and decoder fits use training data; decoder parameters
  and embedding dimension use validation scores only. Test scores are reported
  for the selected dimension, separately for linear and kNN decoders.
- Time-resolved behaviour-to-neuron Ridge and reduced-rank Ridge encoding are
  reported separately in `encoding.json`. These are local baselines, not a
  reproduction of the published global RRR fit. Coefficients (time × variable ×
  neuron) and test predictions are saved as NPZ files.
- Each null jointly permutes complete training-trial label trajectories, then
  retrains the encoder and decoder. Test/validation labels retain their original
  assignment. This preserves within-trial trajectories and dependencies between
  variables. Trial exchangeability is imperfect for task blocks and drift;
  these are diagnostics, not calibrated permutation p-values. One null is a
  pilot minimum; increase `--nulls` before estimating a null distribution.
- `scores.json` reports R² or balanced accuracy and 95% trial-bootstrap intervals.
  Degenerate targets yield an unavailable score rather than fabricated evidence.
  Intervals are conditional on the fitted model and selected dimension.
- `stability.json` reports validation-fitted linear alignment evaluated on test
  embeddings, and neuron-attribution Spearman agreement across seed pairs at
  each dimension. Stability does not identify exact biological dimensionality.
- `summary.json` averages seeds within each session, then bootstraps sessions.
  Three sessions give weak population uncertainty; sessions from one animal are
  not independent animals. No across-mouse claim is supported by these files.
- Each combination saves all embeddings, loss histories, final models,
  checkpoints, held-out attributions and test predictions. Data hashes, code
  hashes, package versions and device information are in `manifest.json`.
  Non-finite output fails instead of silently becoming a successful result.

## Remaining interpretation work

The second thesis question needs neuron-level within-area clustering,
reproducibility filtering, Gaussian nulls, and multiple-testing correction.
The pilot saves neuron IDs, area metadata and per-seed attributions for that
analysis, but does not automatically declare clusters or cortical specialization.
Area comparisons also require appropriate session weighting and neuron-count
controls. Training a known-ground-truth synthetic generative model is still
needed before making xCEBRA recovery/identifiability claims.

The local raw dataset contains 205 NPZ sessions. The original RRR JSON paths
contain placeholders, so direct paper-result comparisons remain unavailable.
A real GPU run, runtime budget, and the actual RRR export remain external needs.

## Verification

`python -m unittest discover -s tests -v` checks training/test isolation, joint
trial shuffling, and attribution invariance to batch partition. A short CPU
smoke run exercises continuous and categorical models, two seeds, two dimensions,
observed/null training, checkpoint saving, selection, stability and result export.
This checks execution, not convergence or biological validity. The local NumPy
2.2/macOS numerical stack emits matrix-multiplication runtime warnings despite
finite saved results; inspect the GPU environment independently before scaling.
