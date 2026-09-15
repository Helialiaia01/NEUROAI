# Pre-GPU verification — 9 September 2026

**Status:** the planned local implementation is complete and the checks below
passed. This is engineering readiness for GPU calibration, not a completed
scientific training result. The [machine-readable record](pre_gpu_verification_2026-09-09.json)
saves measured values, code/data fingerprints and checksums of the local evidence.

## Changes verified

- Centre-preserving trial-edge padding agrees with actual CEBRA inference;
  continuous label differences exclude cross-trial transitions.
- Full regularized objective/gradient diagnostics, finite checks, saved models,
  decoder/scaler persistence and per-variable integrity-checked recovery.
- Training-only preprocessing, matched baselines, joint/block-conditional nulls,
  paired trial/block uncertainty, explicit cohort and subject-aware aggregation.
- Bounded context construction, per-session workers, provenance-preserving merge,
  interruption records and explicit preprocessing exclusions.
- Neuron/area profile export, reproducible KMeans selection, covariance-matched
  Gaussian nulls, BH correction, seed agreement and strict published-RRR matching.

## Measured checks

| Check | Outcome and scope |
|---|---|
| Unit/regression suite | **24 passed**, including real CEBRA training/recovery, alignment parity, corruption, paired bootstrap, exclusions and strict RRR schema/identity tests. |
| Worker integration | **Passed:** two eligible synthetic sessions, one explicit exclusion, 32 encoder fits at three iterations, same-configuration recovery, merge, subject aggregation and offline analysis. |
| Real raw-session CPU smoke | **Passed finite execution:** 402 retained trials, 54 cortical neurons, 100 time bins, train/validation/test counts 242/80/80; eight variables and one null, 16 fits at three iterations. |
| Real-session cost | Approximately **44.0 seconds**, process peak RSS **789,020,672 bytes**. This is a CPU smoke measurement, not a GPU/full-training estimate. |
| Real-session analysis | Exercised four session-area-method tests with 99 Gaussian draws on the short-run artifacts. No scientific clustering conclusion is drawn. |
| Learned synthetic diagnostic | Eight fits: two variables, two seeds, regularization 0 and 0.01, 100 iterations. Decoding R² approximately **0.67–0.79**; attribution AUROC **0.47–0.78**. Support recovery was inconsistent. |
| Static validation | Changed Python sources compile; Slurm shell syntax and Git whitespace checks pass. |

The real-session smoke predates the last exclusion/warning-bookkeeping changes;
its exact code fingerprint is retained separately. The final worker integration
exercises the final training/analysis code fingerprint. Tiny integration fixtures
are designed to exercise paths, not imitate the complete IBL population.

## Reproducing local verification

Activate a compatible isolated training environment and run from the repository
root, selecting a new empty output directory:

```sh
python -m unittest discover -s tests -v
python -m scripts.verify_pipeline --output outputs/new_cpu_verification
python -m scripts.validate_synthetic_recovery --help
```

The measured local tests used the repository `.venv`, with NumPy 2.2.6 isolated
under ignored `outputs/pre_gpu_validation/numpy226`, selected through `PYTHONPATH`.
The original `.venv` was not modified. `LOKY_MAX_CPU_COUNT=1` and a writable
`MPLCONFIGDIR` were used for the local checks. Full artifacts remain in the ignored
`outputs/pre_gpu_validation/` directory; the small verification record is suitable
for version control. All final checked manifests identify their actual packages.

## Remaining gates

1. **Compute:** obtain the actual node/login/scheduler/allocation and GPU details;
   test the Linux/CUDA environment, real kernels, convergence, runtime and VRAM.
   The NAS invitation is retained locally as connection metadata only. No login
   or GPU job was attempted.
2. **Attribution:** use the supplied learned synthetic diagnostic to examine longer
   training, regularization and assumptions. Its current recovery is not a passed
   identifiability result. Useful decoding alone cannot validate neuron support.
3. **Reference data:** obtain the genuine published RRR export for that comparison.
   The public LFS media URL returned HTTP 404; placeholders are rejected.
4. **Population inference:** obtain verified subject IDs if making animal-level
   claims. The current reserve is at session level and does not guarantee unseen
   animals. Keep calibration/tuning within the exploratory cohort.
5. **Numerical environment:** NumPy 2.2.0 and 2.2.6 on this Mac emitted matrix-operation
   warnings despite finite results. The runner preserves emitted warnings in
   `warnings.json`; the warnings have not been declared resolved. Validate the
   compute environment before scaling.

An optional NumPy 2.4.1 compatibility test installation was rejected by automatic
approval review citing an account usage limit. It was not retried through another
route, and the existing environment remained unchanged. The completed checks use
the available environment; this optional test is not represented as passed.
