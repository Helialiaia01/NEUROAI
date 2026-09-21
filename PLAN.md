# Pre-GPU implementation plan

**Loki execution proposal, 15 September:** [Review the cluster setup and training
plan](cluster/LOKI_PLAN.md). Approved with user edits; execution is in progress. The completed implementation
record below is preserved.

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

- [ ] Confirm compute host, login/VPN, scheduler/allocation, GPU/VRAM, storage and
  wall-time limits. The NAS invitation alone does not establish compute access.
- [ ] Build/test Linux/CUDA, check numerical warnings, execute real kernels and
  measure calibration cost. Local NumPy 2.2.0/2.2.6 warnings remain unresolved;
  an optional 2.4.1 test install was blocked by automatic approval review.
- [ ] Validate attribution on longer learned synthetic experiments. The current
  100-iteration diagnostic has AUROC 0.47–0.78 and is not reliable support recovery.
- [ ] Obtain verified subject IDs before animal-level inference; refreeze the
  cohort if changing the split unit to animal. Current reservation is session-level.
- [ ] Obtain the actual published RRR export if direct published-result comparison
  is required. The public LFS media check returned 404; local baselines stay separate.
- [ ] Run one-session GPU calibration, then the controlled 1–3-session pilot.
  Freeze design/iterations using training and validation evidence before reserved
  confirmatory evaluation and full-cohort analysis. The 3-session default grid is
  432 encoder fits before exclusions; do not infer its budget from a CPU smoke.
- [ ] Promote findings to thesis results only after convergence, observed/null
  comparisons, uncertainty, stability and interpretation checks are saved.

## Handoff

Local pre-GPU implementation is complete. No full GPU run, biological attribution
validation or published RRR replication is claimed. Changes are in the working
tree; a remote launcher that clones GitHub will need the reviewed changes pushed
before use. NAS account metadata remains in a local Git-ignored note; no password
is included in project documentation.
