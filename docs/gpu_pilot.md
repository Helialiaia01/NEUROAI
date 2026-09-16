# GPU pilot protocol — updated 9 September 2026

The local implementation and CPU checks are complete for the pre-GPU plan.
GPU convergence, numerical compatibility and scientific recovery remain to be
validated. See [PLAN.md](../PLAN.md), the [analysis protocol](analysis_protocol.md)
and the [verification record](pre_gpu_verification_2026-09-09.md).

## Purpose and method decision

Use the name **regularized per-variable CEBRA adaptation**. Each task variable
supervises an independent encoder, with Jacobian regularization and pseudoinverse
attribution. The implementation does not claim to be canonical multiobjective
xCEBRA with shared output slices; see the
[official demonstration](https://cebra.ai/docs/demo_notebooks/Demo_xCEBRA_RatInABox.html).
The thesis compares held-out decoding, neuron-profile structure and consistency
across dimensions/seeds. Encoding and decoding answer different questions.

Attribution is sensitivity in training-standardized neural coordinates, not a
causal effect or a unique contribution of correlated labels. The learned synthetic
check decoded its latent variables but did not reliably recover neuron support.
Use longer synthetic calibration before interpreting neuron-level findings.

## Data and evaluation

The frozen cohort has 164 exploratory and 41 reserved confirmatory sessions from
205 raw NPZ files. Existing model sessions were excluded from the reserve. Subject
IDs are unknown; add verified metadata before freezing an animal-level design.
Supplying `--cohort` selects its entire requested phase unless `--session-ids` is
also supplied; `--max-sessions` only applies without a cohort. Always select the
small pilot explicitly. Confirmatory runs require a cohort and one frozen dimension.

Each session uses a fixed 60/20/20 trial split. Training trials determine neuron
filters, movement alignment and scaling; smoothing stays within trials. Cortical
areas are explicitly selected by default. QC records raw identities, retained
counts, class coverage, units and outcome semantics. This is offline decoding:
symmetric smoothing and neural temporal context can use future bins.

Neural activity is standardized per neuron and time bin using training trials.
When a time bin has negligible training variation, its scale falls back to 10%
of that neuron's pooled training scale. The raw scale, fallback scale and affected
bins are saved in `preprocessing.npz`; this prevents rare held-out spikes from
being amplified by an unstable near-zero denominator while preserving the
per-time-bin scaling where it is supported by the training data.

Training windows use centre-preserving replication at each trial edge, matching
inference. Continuous label differences use within-trial transitions only.
Decoding uses interior bins. Matched neural-context Ridge/logistic baselines and
linear/kNN embedding decoders fit scaling on training data and select parameters
on validation data. Dimensions are selected by mean validation decoding across
seeds. kNN measures the representation/decoder combination; the primary contrast
uses linear decoders on both neural windows and embeddings.

Behaviour-to-neuron Ridge and rank-projected Ridge encoding are saved separately.
They are local time-resolved baselines, not the published global RRR estimator.
Original RRR comparison strictly requires a genuine identity-matched export.

The null jointly permutes whole training-trial label trajectories and retrains
encoders/decoders. Validation/test labels stay unchanged. `--shuffle within_block`
restricts permutation to original prior-block runs; saved `changed_supervision`
reveals labels that this control preserves. One null is a pilot diagnostic, not a
calibrated significance distribution. Trial bootstrap intervals include paired
improvement over the matched baseline; `--bootstrap-unit block` assesses dependence
sensitivity. Aggregation averages seeds within sessions and, when all subject IDs
are known, sessions within subjects. One resampling unit has no interval.

## Run sequence after compute access is available

Build the [training environment](../environments/README.md). From the repository
root, inspect the selected data and environment without training:

```sh
python -m xcebra_ibl.experiments --cohort xcebra_ibl/configs/cohort.json --session-ids 044be2f4-e898-404c-91e2-1285cbada2cd --preflight-only --device cuda --output outputs/preflight
```

Calibrate one exploratory session (16 encoder fits, including controls):

```sh
python -m xcebra_ibl.experiments --cohort xcebra_ibl/configs/cohort.json --session-ids 044be2f4-e898-404c-91e2-1285cbada2cd --seeds 2025 --dimensions 4 --iterations 500 --device cuda --output outputs/gpu_calibration
```

Then run an explicitly selected three-session pilot:

```sh
python -m xcebra_ibl.experiments --cohort xcebra_ibl/configs/cohort.json --session-ids 1a507308-c63a-4e02-8f32-3239a07dc578 288bfbf3-3700-4abe-b6e4-130b5c541e61 6c6b0d06-6039-4525-a74b-58cfaa1d3a60 --seeds 2025 2026 2027 --dimensions 2 4 8 --iterations 500 --device cuda --output outputs/gpu_pilot
python -m xcebra_ibl.analysis.pilot --input outputs/gpu_pilot --output outputs/gpu_pilot_analysis --null-draws 99
```

These session choices are an engineering sample, not a representative brain-wide
sample. The full grid is **432 encoder fits** before exclusions. Inspect total and
regularization loss, gradients, validation decoding, nulls, stability, seconds and
VRAM before increasing toward 10,000 iterations. Freeze choices before evaluating
reserved sessions. Do not tune by repeatedly inspecting test scores.

For multiple workers and Slurm, use the [portable job instructions](../cluster/README.md).
Each worker owns one session's full candidate grid. Merge verifies every planned
worker and preserves its provenance; explicit preprocessing exclusions remain
visible. Run clustering once over the merged study so its correction family
covers all session-area-method tests. Use at least 999 Gaussian draws for improved
resolution in a larger analysis, and inspect the resulting minimum attainable p-value.

Kaggle defaults to one-session calibration with explicit CUDA and pinned training
requirements. `KAGGLE_PIPELINE_ARGS` replaces the default arguments; include the
cohort/session selection and `--device cuda` in an override. The launcher clones
GitHub, so local changes must be committed/pushed before a new remote run can use
them. `KAGGLE_EXPERIMENT_MODE=legacy` selects the historical path and requires its
own legacy CLI arguments; it does not consume the controlled pilot protocol.

## Artifacts, recovery and verification

Each completed variable has an integrity-checked recovery model and diagnostics.
A matching rerun reuses completed variables and combinations. An interrupted
variable restarts from its seed; optimizer-step continuation is not implemented.
Checkpoints are saved every 100 iterations with one retained per variable.
Configuration, code, input or environment changes require a new output directory.
Corruption fails explicitly instead of being accepted as a completed result.

Outputs include fitted preprocessing, raw neuron IDs/UUIDs, models, decoders and
scalers, encoding coefficients/intercepts/predictions, embeddings, full training
diagnostics, held-out attributions, paired scores/intervals, seed stability,
runtime/memory profiles, numerical-warning counts and hashed completion manifests.
`warnings.json` retains emitted warning details/counts without changing filters.
Non-finite training/output fails. Final merged output retains each worker manifest.

The offline analysis writes neuron/session-area tables, reproducible KMeans
selection, covariance-matched Gaussian nulls, BH correction and cross-seed cluster
agreement. These are explicitly defined extensions to the reference protocol;
Gaussian rejection does not establish biological categories. The actual published
RRR export remains unavailable, and NumPy/macOS matrix-operation warnings remain
unresolved despite finite CPU outputs. Linux/CUDA validation is still required.

Runnable local checks:

```sh
python -m unittest discover -s tests -v
python -m scripts.verify_pipeline --output outputs/new_cpu_verification
python -m scripts.validate_synthetic_recovery --help
```

The verification script uses tiny fixtures, three iterations per fit and an empty
output directory. It is an execution check, not a convergence benchmark.
