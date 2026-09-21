# Attribution corrections and verification — 21 September 2026

## Verified corrections

1. Attribution sampling now has its own seed (default 42), independent of the
   encoder-training seed. All candidates receive identical sample centers.
   Centers, original trial IDs and time-bin IDs are saved in
   `attribution_samples.npz`. Stability rows explicitly report whether both
   models used identical samples. Loading historical checkpoints retains their
   original sampling seed unless overridden for the new audit.
2. For explicitly normalized encoders (`net.normalize`), the temporally averaged
   Jacobian is projected onto the output's tangent space before pseudoinversion:
   `J_corrected = J - z (z.T J)`, using a float64 unit vector `z`.
   This is an identity in exact arithmetic because differentiating `z.T z = 1`
   gives `z.T J = 0`. It removes a demonstrably false radial derivative caused by
   float32 roundoff, without selecting a cutoff to improve behavioral scores.
   Unnormalized encoders are unchanged. This explicit numerical safeguard goes
   beyond the original CEBRA implementation and is recorded in provenance.
3. Attribution outputs now include effective-rank and single-sample-contribution
   diagnostics. These are numerical sensitivity measures, not biological quality
   thresholds. Future controlled-run CLI defaults use 1,024 samples, supported
   by the precision comparison below. Tests with tiny fixtures override this.

## Actual failure and independent check

The saved 1,000-step outcome encoder for seed 2027 had one sample whose numerical
radial singular value was 2.21e-8. Its largest tangent singular value was 0.00201,
so the false radial direction survived the relative cutoff 1e-5. Its inverse
dominated the result: one of 1,024 samples supplied 97.77% of the summed absolute
inverse magnitude. The normalized four-dimensional output should have rank at
most three, but that sample appeared to have rank four.

An independent float64 derivative reduced the radial singular value to 6.04e-18;
the tangent projection reduced it to 4.11e-19. The corrected attribution for this
sample matches float64 with relative L2 error 1.71e-6. No sample was discarded.
The largest sample contribution after correction was 1.10%, with rank three
throughout. A regression test recreates false radial inversion analytically.

## Saved-model evaluation, no retraining

The audit covers all 24 observed encoders (eight variables, three training seeds)
from the completed calibration, using held-out neural inputs reconstructed with
the original training split and retained neuron identities. Checkpoint hashes are
verified before loading. The CPU audit loads encoder weights directly because
CEBRA 0.6 estimator loading otherwise restores saved CUDA device metadata.

| Measurement | Result |
|---|---:|
| Median cross-model rho, historical 256 sampling | 0.1144 |
| Median cross-model rho, shared 256 samples, seed 42 | 0.1032 |
| Median cross-model rho, corrected shared 1,024 samples, seed 42 | 0.1152 |
| Median cross-model rho, corrected shared 1,024 samples, seed 43 | 0.1180 |
| Median same-model sampling rho, 256 samples | 0.9515 |
| Median same-model sampling rho, corrected 1,024 samples | 0.9890 |
| Worst same-model sampling rho, uncorrected 1,024 samples | 0.2418 |
| Worst same-model sampling rho, corrected 1,024 samples | 0.9479 |

The correction fixes the identified numerical failure and improves measurement
repeatability. It does **not** fix the weak agreement between independently
trained encoders. Claims of reliable individual-neuron identification remain
unsupported. This exploratory audit does not substitute for synthetic recovery
across independent graphs, a canonical multiobjective method comparison, or a
held-out confirmatory study. No full-cohort run is justified by these fixes alone.

The saved synthetic-model Jacobian audit passed again with the tangent safeguard
explicitly applied to both implementations. This verifies numerical agreement;
it is not a claim that synthetic recovery became reliable.

## Evidence and reproduction

- `scripts/audit_attribution_sampling.py`: paired sampling audit, no fitting.
- `outputs/attribution_sampling_20260921/review/sampling_audit.json`: before correction.
- `outputs/attribution_sampling_20260921/review_corrected/sampling_audit.json`: after correction.
- `outputs/attribution_sampling_20260921/radial_roundoff_check.json`: singular values.
- `outputs/attribution_sampling_20260921/float64_validation.json`: independent precision check.
- `outputs/attribution_audit_20260921_corrected/audit.json`: synthetic reference agreement.

Run the current saved-model audit with:

```bash
python -m scripts.audit_attribution_sampling \
  --models outputs/attribution_sampling_20260921/models \
  --inputs outputs/attribution_sampling_20260921/inputs.npz \
  --output outputs/attribution_sampling_20260921/new_review
```

Use a fresh output directory. The earlier calibration artifacts and decoder
scores were not overwritten. New attribution computations use the recorded
`normalized_tangent_projection_v1` numerical policy, including when old models
are loaded; old numerical results remain available in the archived audit.
