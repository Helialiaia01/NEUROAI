# Pre-GPU implementation plan

**Loki execution proposal, 15 September:** [Review the cluster setup and training
plan](cluster/LOKI_PLAN.md). Approved with user edits; execution is in progress. The completed implementation
record below is preserved.

**Current decision, 25 September:** the initial 48-session study and its offline
analysis are complete. The remaining 157 sessions are running under `nohup` on
both P6000 GPUs (95/116 exploratory sessions complete at the latest check; the
41 reserved confirmatory sessions follow automatically). Dr Shuqi Wang supplied
the published `RRR_selectivity.json` and authoritative code. The comparison now
uses the released schema, the published `RRR_r2 - null_r2 > 0.015` filter and
sum-over-time absolute coefficient magnitude for comparison with unsigned
Jacobian attribution. This is a post-training analysis correction and does not
require restarting the CEBRA run. See
[published RRR integration](docs/published_rrr_integration_2026-09-25.md).

Updated and reviewed: 9 September 2026. Scope: complete locally verifiable
scientific and engineering fixes before GPU activation. Existing data and
historical outputs are preserved. No NAS login or GPU job was attempted.

## Review and adjustments

The previous lowercase Kaggle plan is preserved in
[docs/legacy_kaggle_plan_2026-09-03.md](docs/legacy_kaggle_plan_2026-09-03.md).
This plan was created before implementation, reread against the actual code and
adjusted as the checks exposed concrete issues:

- Preserve the explicitly named **regularized per-variable CEBRA adaptation**.
  Canonical multiobjective xCEBRA is a separate scientific experiment.
- Use one session's full candidate grid per worker, with recovery of completed
  variables. This keeps preprocessing/baselines and validation selection together
  and avoids duplicate work and concurrent output writes.
- Treat learning a known synthetic graph as a diagnostic, not an automatic pass.
  The measured inconsistent attribution recovery leaves scientific interpretation
  open even though implementation/execution checks pass.
- Record preprocessing exclusions as integrity-checked outcomes so the complete
  planned cohort can be accounted for during merging.
- Keep missing published RRR data, subject metadata and compute allocation explicit;
  do not invent them or label the local Ridge baseline as published RRR.

## Completed local work

- [x] 1. Sampling and alignment: vectorized centre-preserving within-trial padding
  matching inference; valid within-trial continuous differences; validation of
  short/discontiguous trials; regression and actual CEBRA parity checks.
- [x] 2. Model execution: regularized loss/gradient diagnostics and finite checks;
  model/decoder/scaler/encoding-intercept persistence; artifact validation;
  per-variable recovery, deterministic seeds and interruption records.
- [x] 3. Experiment design: resolved model/preprocessing metadata, cortical selection,
  label semantics/class coverage, paired trial/block uncertainty and null controls;
  frozen 164/41 exploratory/reserved session cohort, subject-aware aggregation.
- [x] 4. Analysis: neuron/area tables from pilot outputs, reproducible clustering,
  covariance-matched Gaussian nulls, BH correction and cross-seed agreement;
  strict identity/schema-checked published RRR comparison, without a fallback.
- [x] 5. Execution/environment: bounded context memory, profiling, job manifests,
  independent worker outputs, provenance-preserving merge and explicit exclusions;
  scheduler-neutral jobs, configurable Slurm example, pinned training recipe,
  preflight and local package snapshot. Linux/CUDA compatibility remains unverified.
- [x] 6. Local validation: 24 tests pass; final 32-fit worker integration passes;
  full raw-session CPU smoke and offline analysis pass; learned synthetic diagnostic
  measured; compilation, shell syntax and whitespace checks pass.
- [x] 7. Documentation: current methods/launch protocol, historical audit notices,
  execution instructions and measured verification record updated and reviewed.

See [verification evidence](docs/pre_gpu_verification_2026-09-09.md),
[current training methods](docs/gpu_pilot.md),
[analysis methods](docs/analysis_protocol.md),
[portable jobs](cluster/README.md) and [environment recipe](environments/README.md).
The new runner is the supported controlled-study entry point; legacy scripts and
historical figures are not automatically promoted to final thesis evidence.

## Open external and scientific verification

Update 19 September: Loki calibration, the corrected three-session pilot and
merged offline analysis have completed. The historical checklist below is
superseded for those execution milestones by
[the pilot review](docs/pilot_review_2026-09-19.md). Scientific acceptance,
attribution validation and full-cohort dispatch remain pending.

The subsequent [attribution audit and training readiness decision](docs/training_readiness_2026-09-19.md)
records the tested numerical corrections, synthetic limitations and the prepared
48-fit calibration launcher. Full-cohort scientific acceptance remains open.

The [21 September corrections](docs/attribution_fixes_2026-09-21.md) fix a verified
normalized-Jacobian roundoff failure and separate attribution sampling from
training randomness. Saved-model tests confirm improved numerical repeatability;
between-model neuron-rank stability remains weak, so full-cohort acceptance is
still open.

The [multiobjective benchmark and sampler correction](docs/multiobjective_benchmark_2026-09-21.md)
adds a separate official-solver prototype and fixes CEBRA private sampler seeds.
Fresh smoke runs repeat exactly after the correction; prior global seeds alone
did not guarantee replay of historical training. The corrected full pipeline
integration and 42 regression tests pass locally.
The corrected synthetic benchmark is complete: multiobjective decoding improves
on the redundant graph, but mean support AUROC does not (0.701 versus 0.708).
The same tests pass on Loki. No production model replacement or full-cohort
dispatch is justified by this small benchmark.

The subsequent [fresh-data and neuron-masking validation](docs/independent_validation_2026-09-21.md)
supports a deadline-aware thesis run. The existing adaptation passed the
predefined sensitivity direction in 7/8 fresh cases and the multiobjective
prototype in 8/8, but the latter is not yet an IBL-ready production pipeline.
Proceed with 48 frozen exploratory sessions using decoding as the primary result
and attribution as exploratory sensitivity. Keep all 41 confirmatory sessions
reserved.

- [x] Confirm Loki login, two P6000 GPUs and writable HDD storage. User authorized
  assuming no allocation limits unless the supervisor advises otherwise.
- [x] Execute Linux/CUDA training and measure calibration cost; the 48-fit
  calibration completed in 69.1 minutes. Local and Loki numerical regression
  tests passed after the 21 September attribution correction.
- [ ] Validate learned attribution. The 1,000-step adaptation diagnostic and
  real-data cross-seed checks do not yet establish reliable neuron recovery.
  Compare the official multiobjective solver on predefined synthetic graphs
  before integrating it into the IBL runner.
- [ ] Obtain verified subject IDs before animal-level inference; refreeze the
  cohort if changing the split unit to animal. Current reservation is session-level.
- [x] Obtain and verify the published RRR export. The Git-LFS media artifact contains
  59,820 neurons across 178 sessions and has SHA-256
  `e73993443982e3df5c0d0c86c7f032ce1cdeddf1d5f18fa16804d19df1442042`.
  Exact-unit comparison remains limited by IBL spike-sorting version changes;
  session-balanced area-level comparison is therefore the main published-result
  comparison, while local Ridge remains the matched decoding baseline.
- [x] Run one-session GPU calibration and the controlled three-session pilot
  (432 encoder fits); both completed. Their completion is not scientific acceptance.
- [ ] Freeze design/iterations using training and validation evidence before
  reserved confirmatory evaluation and full-cohort analysis.
- [ ] Promote findings to thesis results only after convergence, observed/null
  comparisons, uncertainty, stability and interpretation checks are saved.

## Handoff

GPU pilot, calibration and the first 48-session study are complete. The remaining
157-session run is active on Loki. Decoding and its matched linear baseline remain
the primary performance result; neuron attribution and agreement with the
published RRR selectivity structure remain secondary analyses with explicit
stability and multiple-testing controls. The current run should finish before any
optional exact-data sensitivity study is considered. NAS account metadata remains
in a local Git-ignored note; no password is included in project documentation.
