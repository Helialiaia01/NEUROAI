# Training readiness and attribution audit — 19 September 2026

## Decision

The implementation is ready for a **bounded exploratory calibration**, once a
P6000 is available. It is not scientifically cleared for a full-cohort run that
claims reliable individual-neuron identification. No implementation change can
guarantee that scientific result. Reserved sessions remain untouched.

## Attribution audit

Sources: [CEBRA attribution API](https://cebra.ai/docs/api/pytorch/attribution.html),
[official multiobjective xCEBRA example](https://cebra.ai/docs/demo_notebooks/Demo_xCEBRA_RatInABox.html),
and the installed CEBRA 0.6.0 source. The official example combines time and
behavior objectives in embedding slices. Our independent per-variable encoders
remain a regularized CEBRA adaptation, not that multiobjective implementation.

The numerical audit used the installed official `_jacobian.py` backend against
all eight saved 1,000-iteration synthetic encoders. It used SciPy pseudoinversion
with the same explicit relative threshold as the runner; this isolates derivative
and inversion agreement from cutoff differences. Maximum absolute attribution
discrepancy was 1.30e-7. Temporal finite differences also passed, including an
incomplete last batch. Loading only that backend avoided unrelated optional
CVXPY/Captum dependencies; this was not a full official-package pipeline test.

Repairs made:

- Corrected a stale squared-forward-gradient docstring.
- Rejected ambiguous 2D temporal inputs, empty/nonfinite inputs, invalid windows,
  invalid output slices and invalid inversion/batch parameters.
- Joint attribution now inverts the full Jacobian before selecting output
  coordinates. Existing per-variable pilot scores are unaffected by this fix.
- Added regression tests for the joint inverse and temporal finite differences.

All eight normalized 2D encoders had median effective Jacobian rank one. Unit
normalization constrains local movement to the unit circle's tangent direction;
the inverse is therefore a truncated, minimum-norm local mapping. It is not a
unique reconstruction of neural causes.

The current absolute-entry (L1) aggregation depends on embedding coordinates.
An L2 alternative is invariant to orthogonal output rotations but did not
consistently resolve the synthetic support-recovery weakness. A forward-gradient
variant also had mixed results. Cutoffs 1e-3 and 1e-5 produced the same AUROCs;
1e-7 changed some rankings, potentially by admitting near-null numerical
directions. No cutoff or attribution variant was promoted on this evidence.

The synthetic data have substantial redundancy: six correlated neurons encode
each latent. Good decoding therefore need not imply unique recovery of every
generating neuron. This is a limitation of this diagnostic as well as a caution
for correlated IBL recordings. The evidence supports sensitivity analysis, not
causal or unique-neuron claims.

Audit artifact: `outputs/attribution_audit_20260919/audit.json` includes model and
official-backend hashes, reference errors, effective ranks and all tested variants.

## Prepared next calibration

`xcebra_ibl/configs/calibration_1000.json` freezes the bounded design: one previously
exploratory session (`1a507308-c63a-4e02-8f32-3239a07dc578`), three seeds,
dimension four, 1,000 iterations, eight variables subject to support checks, one
retrained null, regularization 0.01 and 500 bootstrap draws. Maximum: 48 encoder
fits. The purpose is convergence and seed stability at a fixed design, not
optimizing held-out test scores. Do not extrapolate the earlier runtime linearly
without measuring this job's preprocessing, decoding and attribution costs.

Run from the Mac repository root:

```bash
bash cluster/run_loki_calibration.sh --check
```

The launcher logs into Loki, activates the correct environment, requires matching
commits and clean training code, checks input availability, and refuses an occupied
P6000. The check does not start training. When the scientific purpose above is
accepted and the GPU is free, use `--launch`. The job uses `nohup`, creates a fresh
exclusive run directory, writes its PID/log and records the exit status.
This availability check is not a resource reservation; coordinate shared GPU use.

## Full-cohort launch criteria

1. Bounded calibration completes with verified artifacts and acceptable finite
   diagnostics; inspect training and validation loss/decoding by variable/seed.
2. Fix the thesis claim: decoding plus exploratory sensitivity is currently
   supportable. Stronger neuron recovery requires further synthetic evidence and
   a reviewed method change, potentially canonical multiobjective xCEBRA.
3. Freeze dimensions, iterations, null protocol, primary contrasts and a cohort
   small enough for measured runtime plus rerun/reporting time before 8 October.
4. Obtain verified animal metadata before animal-level inference. Paper RRR
   comparison additionally requires identity-matched original results and methods.
5. Do not interpret a failed cluster test as proof of no structure, or select a
   method because it makes the already inspected test results significant.

The evaluation-only review and its corrected intervals are documented in
[the pilot report](pilot_review_2026-09-19.md). Original Loki outputs are preserved.
