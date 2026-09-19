# Corrected pilot review — 19 September 2026

The completed Loki pilot used code `164fe45`, three sessions, three seeds,
dimensions 2/4/8, observed and shuffled conditions, and 500 iterations.
All three session completion markers reported complete and merging succeeded.
The offline analysis produced neuron profiles, area profiles and clustering
results on 18 September at 12:53. These are pilot findings, not confirmatory
evidence or a reproduction of published RRR results.

## Findings and limits

The primary linear-decoder comparison did not establish a general improvement
over the matched neural-context baseline. Mean wheel R² was 0.035 for CEBRA
versus 0.042 for the baseline; whisker R² was 0.049 versus -0.153, with the
paired improvement interval including zero. Three sessions give weak support
for population inference, and verified animal identities are still missing.

Eight session-area-method clustering tests produced no BH-significant result
(minimum q = 0.08 with 99 Gaussian draws). No CEBRA result passed the recorded
cross-seed cluster criterion. This does not prove that neural structure is absent.
Do not increase null draws merely to seek significance; use a prospectively
specified resolution for the final study.

The pathological continuous decoding values disappeared after the neural scale
floor fix. One session recorded 18 `y_pred contains classes not in y_true`
warnings. Such warnings indicate missing target class support; the previous
status report attributed all of them to bootstrap draws without tracing each
call. The follow-up below now verifies that source by replaying the affected draw.

## Repairs after this run

- Categorical trial/block bootstrap draws now require all classes present in
  the original evaluated target. Missing-class draws are counted and excluded
  for both model and paired baseline. Intervals are conditional on full class
  support; requested, valid and excluded counts must accompany interpretation.
  No extra draws are silently substituted. Point estimates remain unchanged.
- Analysis logs session/area/method and Gaussian-draw progress, refuses to
  overwrite a nonempty output, and writes `analysis_complete.json` only after
  all results are saved.
- The uncommitted watcher reports completed combinations out of 18 per session.
  It previously divided combination markers by 144 encoder fits. Earlier
  percentages and derived timing estimates from that counter were incorrect.
  Session completion markers still indicated completion correctly.

These local repairs do not modify the completed remote artifacts or their
provenance. Old bootstrap intervals retain the old policy and must be labeled
accordingly; the new policy is not retroactively applied by rerunning clustering.

## Completed follow-up checks

`python -m scripts.review_saved_pilot` now reviews saved predictions without
training. It checks copied inputs against the original session checksums,
preserves dimension selections and point estimates, records source and code
hashes, and writes revised intervals to a fresh destination. A fixture review
also passed with two included sessions and one explicit exclusion.

The completed real-data review contains 312 unchanged point estimates. Exactly
one of 500 draws omitted a contrast-level class in session `288bfbf3`; this draw
affected 13 score rows. Replaying the original calculations reproduced all 18
warnings. The revised policy retains 499 draws in those rows; ten score intervals
change. Session-level aggregate point estimates and their original session
bootstrap intervals are unaffected by this within-session revision.

Artifacts are in `outputs/pilot_review_20260919_verified/`, including
`revised_intervals.json`, `review_complete.json`, diagnostic summaries, PNG/PDF
decoding and attribution-stability figures, and `synthetic_1000/report.json`.
These generated outputs are Git-ignored. Original remote results remain intact.

Session runtimes were 77.3, 94.8 and 109.1 minutes. Median attribution rank
correlations across seed pairs and dimensions were 0.072, 0.105 and 0.081
for the three sessions, respectively. Training-loss tails are heterogeneous;
their aggregate cannot establish convergence for every encoder.

An eight-fit, two-seed synthetic check at 1,000 iterations finished on CPU in
188 seconds. With regularization 0.01, neuron-support AUROC ranged from 0.417 to
0.778 while linear decoding R² ranged from 0.593 to 0.822. Unregularized AUROC
ranged from 0.528 to 0.833. This is one small synthetic graph, not a formal
comparison, and does not establish reliable neuron-support recovery. Longer
training alone has not resolved that concern.

Both Loki P6000s were heavily occupied during this follow-up. The newly started
synthetic process was stopped after verifying its command, then run locally;
other processes were untouched. No full-cohort job was launched.

## Before scaling

1. Inspect saved training/validation diagnostics for convergence; benchmark a
   bounded longer run only if those diagnostics justify it. Avoid choosing
   settings by repeatedly inspecting held-out test performance.
2. Use the newly recalculated, labeled intervals for within-session uncertainty.
3. Audit attribution assumptions and compare methods on predefined synthetic
   graphs with known support before interpreting individual neurons biologically.
   The longer diagnostic above did not establish reliable recovery; agree the
   thesis claims and method scope with the advisors before a large run.
4. Freeze settings, null protocol, cohort size and a measured time budget that
   reserves rerun and writing time before 8 October. Full-cohort dispatch remains
   pending; low pilot scores alone do not mandate a code change.
5. Keep the method named regularized per-variable CEBRA adaptation. Canonical
   multiobjective xCEBRA and direct published RRR comparison are not established.
   Obtain genuine identity-matched RRR results and verified animal metadata.
