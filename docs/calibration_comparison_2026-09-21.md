# Matched calibration comparison — 21 September 2026

## Decision

The 1,000-step calibration completed successfully in 69.1 minutes with zero recorded warnings. It does not support launching the full cohort for reliable individual-neuron attribution. Longer optimization produced mixed validation changes and only a small increase in seed stability. This is a scientific limitation, not a failed GPU job.

## Matched design

Compared session `1a507308-c63a-4e02-8f32-3239a07dc578`, dimension 4, seeds 2025/2026/2027, observed and null conditions at 500 versus 1,000 iterations. Raw input fingerprints and package manifests match; the preprocessing artifact hashes are identical, including saved trial splits and retained neuron metadata. The cohort file paths/hashes differ because the pilot used a subset manifest. Session membership and all relevant fitting settings match apart from iteration count.

The two code releases differ (`164fe45` and `cacab25`), including evaluation and defensive attribution fixes; the per-variable formula is unchanged. These are independently restarted fits, not resumed checkpoints. Their first 500 loss values are not bitwise identical (maximum absolute total-loss difference 0.0787). Thus this is a matched practical comparison, not an exact continuation experiment or proof that iteration count alone caused each score change.

## Validation and attribution stability

Linear-decoder validation scores are averaged across three seeds. Categorical scores are balanced accuracy; wheel, whisker and lick use R². Attribution stability is the median neuron-rank Spearman correlation over three seed pairs, at dimension 4 only. Test scores and the pilot’s best-dimension selection are not used in this comparison.

| Variable | Validation 500 | Validation 1,000 | Change | Attribution rho 500 | Attribution rho 1,000 |
|---|---:|---:|---:|---:|---:|
| block | 0.615 | 0.614 | -0.002 | 0.074 | 0.105 |
| side | 0.605 | 0.635 | +0.030 | 0.060 | 0.103 |
| contrast_level | 0.366 | 0.351 | -0.015 | -0.023 | 0.097 |
| choice | 0.701 | 0.693 | -0.007 | 0.008 | 0.109 |
| outcome | 0.596 | 0.598 | +0.002 | 0.099 | 0.120 |
| wheel | 0.083 | 0.025 | -0.058 | 0.148 | 0.181 |
| whisker_max | 0.109 | 0.108 | -0.001 | 0.227 | 0.095 |
| lick | 0.018 | -0.035 | -0.053 | 0.184 | 0.204 |

Validation improves for 3 of 8 variables and decreases for 5. At 1,000 steps the observed validation mean exceeds the shuffled mean for 7 of 8 variables; lick is the exception. One shuffled condition is a diagnostic, not a significance distribution.

Across all 24 variable/seed-pair comparisons, median attribution correlation increases from 0.084 to 0.114. This remains weak reproducibility; a small increase does not establish trustworthy neuron rankings. Sample centers are seeded separately for each model seed, so this stability measure includes both fitting and attribution-sampling variation.

## Convergence

All saved optimization diagnostics are finite. Categorical losses generally decline rapidly then level off with occasional spikes; contrast-level behavior differs appreciably between seeds. Wheel, whisker and lick losses continue to decline toward step 1,000. The median change from steps 801–900 to 901–1,000 is -0.0070 across the 24 observed fits, with range -0.1012 to +0.0503. This descriptive statistic is not a convergence criterion. Lower training loss coexists with worse validation decoding for several variables, so blindly extending to 10,000 iterations is not justified.

![Training loss trajectories](../outputs/calibration_comparison_20260921/review/convergence.png)

## Recommended next decision

Do not launch all 164 exploratory or 205 total sessions merely because this run finished. The confirmatory reserve must remain separate. For a thesis centered on decoding, freeze a bounded cohort and primary contrasts, with attribution labeled exploratory. For a thesis centered on meaningful individual-neuron attribution, first benchmark a suitable method (including a canonical multiobjective xCEBRA candidate) on predefined synthetic graphs and separate sampling variability from fitting variability. Do not describe the existing adaptation as canonical xCEBRA or claim superiority to published RRR.

No additional training was started for this comparison. The completed calibration, source JSONs, reproducible comparison code and machine-readable results are preserved. Relevant artifacts: `scripts/compare_calibration.py`, `outputs/calibration_comparison_20260921/source500/`, `source1000/`, and `review/comparison.json`.
