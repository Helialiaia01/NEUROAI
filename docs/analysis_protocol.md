# Analysis protocol implemented before GPU activation

The primary decoding contrast is regularized CEBRA plus a linear decoder versus
matched neural-window linear decoding. A separate kNN result reports the combined
representation/decoder choice. Decoder parameters and embedding dimensions are
selected on validation trials; cross-seed stability is descriptive and is not a
second test-based model-selection rule.

Trial bootstrap intervals and paired improvements use the same resampled units
for the model and baseline. `--bootstrap-unit block` resamples complete held-out
prior-block runs to assess sensitivity to within-block dependence. One unit has
no confidence interval. Subject IDs, when complete, define the aggregate resampling
unit after averaging seeds within sessions and sessions within subjects. Missing
subjects produce an explicitly labelled session-level summary, not an animal claim.

`--shuffle trial` is the diagnostic joint whole-trial permutation.
`--shuffle within_block` permutes whole trajectories only inside each original
prior-block run. This leaves block identity unchanged and can leave other labels
unchanged in small groups. `changed_supervision` in each combination records that
fact. Such a control cannot assess information that the conditioning preserves.
Neither option automatically provides calibrated permutation significance.

## Neuron profiles and clustering

`python -m xcebra_ibl.analysis.pilot` consumes integrity-checked completed pilot
sessions. It chooses each variable's dimension from the saved **linear decoder**
selection, uses observed models, and retains session/raw-neuron IDs and UUIDs.
The JSON tables include mean raw attributions (`xcebra_attr_*`), explicit normalized
profiles (`profile_*`), and local Ridge coefficient magnitudes (`local_ridge_*`).
The legacy area pipeline is not called on these normalized features.

For clustering, each variable is z-scored over neurons within each session/seed;
normalized profiles are averaged across seeds. Tests run separately per session
and area, preventing changes in session mixture from creating pooled clusters.
The local time-resolved Ridge fit is analyzed separately. At least two variables
and 20 neurons per session/area are required by default. This is a defined extension,
not an exact reimplementation of the published RRR clustering protocol.

For k=2 through min(6, neuron_count/3), KMeans uses ten initializations. Five fits
to independently sampled 80% subsets predict labels for all neurons; median
adjusted Rand agreement with the full fit must reach 0.8. Among qualifying k,
the highest silhouette is selected. If none qualifies, the statistic is zero.
The same complete search and reproducibility filter is repeated on each unimodal
Gaussian draw with the observed mean/covariance and sample size. Monte Carlo
p-values use `(1 + exceedances)/(1 + draws)`. Benjamini–Hochberg adjustment covers
**all session-area-method tests in that analysis invocation**, not each area alone.
Use one invocation on the merged final study to preserve this correction family.

Observed CEBRA profiles additionally report cross-seed cluster agreement when at
least two seeds are available. `passes_distributional_test` and
`cross_seed_reproduced` are separate fields. Gaussian-null rejection can also
reflect skew/heavy tails or other model structure. It does not establish natural
biological categories or causality. The default 99 draws is a pilot diagnostic;
999 or more improve resolution, especially with many simultaneous tests.

## Published RRR and interpretation gates

Published comparisons require a real result table with session and neuron UUIDs,
consistent areas, beta coefficients, and performance columns. The reader removes
the **last** coefficient row as the intercept, matching the reference export,
and applies the published delta-R² > 0.015 filter after identity matching.
Direct comparison to the paper's full estimator still requires the original
export: the local Ridge/rank-projection fit has its own explicitly stated scope.

A learned synthetic diagnostic trains the adaptation on a sparse nonlinear
observation model and reports held-out decoding, attribution AUROC and average
precision across seeds and regularization settings. The initial 100-iteration CPU
check decoded the latent variables but showed inconsistent attribution recovery
(AUROC approximately 0.47–0.78). This is **not a passed identifiability benchmark**.
Use it to assess assumptions, convergence and regularization before interpreting
neuron support. A decoder can work while attribution remains unreliable.

The full original RRR LFS object is listed as 1,268,022,439 bytes with SHA256
`39cc53467eb1dbbe3f74ab3f7dd81a4657da48a182a826ade188b52bc83f4a49`.
The public media URL checked on 9 September returned HTTP 404. The real export
therefore remains an external dependency; no placeholder was relabelled as data.
