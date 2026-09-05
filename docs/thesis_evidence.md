# Thesis evidence baseline

**Snapshot date:** 4 September 2026  
**Scope:** repository, the three supplied papers, and the pasted project brief.

The pasted brief is treated as an unverified project proposal from another language model. It is not evidence that an experiment was run, nor does it override the papers or the repository. Claims in the final thesis must be promoted from “planned” to “result” only after the corresponding experiment, configuration, and output are saved.

## Current code update (5 September 2026)

The historical audit below describes the earlier implementation. The new
[GPU pilot protocol](gpu_pilot.md) supersedes its implementation-status claims:
training-only preprocessing, matched decoding and local encoding baselines,
retrained trial-shuffle controls, seed/dimension sweeps, trial/session intervals,
and seed stability are implemented in `xcebra_ibl.experiments`. The current
method remains a regularized per-variable CEBRA adaptation. CPU smoke tests are
engineering evidence; no new scientific GPU results are claimed. Canonical
multiobjective xCEBRA, calibrated clustering inference and the original RRR
export remain outstanding.

## Continuity and advisor alignment

The attached mail thread provides the project history and advisor expectations:

- The thesis collaboration began with Prof. Spinelli and Dr. Bardella in July 2025, with an expectation of bi-weekly progress updates.
- CEBRA was selected as the main direction in October 2025, initially with replication, embedding visualisation, and kNN decoding as the practical scope.
- xCEBRA was identified in November 2025 as a possible attribution extension after the core CEBRA work.
- On 27 August 2026, Prof. Spinelli confirmed that running on the full IBL dataset is expected and that the three proposed research questions are acceptable, while noting that the deadline is close.
- On 28 August 2026, Helia reported finding the exact `alf` data folder used for the IBL analysis and beginning its download, and asked whether university GPU/cloud computing is available.

The current advisor email is therefore a progress follow-up and a request for computing support. It should not reopen the already answered question of whether a full IBL run is expected.

## Scientific sources

| Source | What it establishes | What it does not establish for this thesis |
|---|---|---|
| Posani et al., *Rarely categorical, always high-dimensional* | A linear RRR analysis of 14,283 neurons across 43 cortical regions and eight task variables; clustering is assessed with reproducible k-means, silhouette scores, and a Gaussian null; within-area categorical structure is strongest in early sensory areas. | It is the baseline to reproduce/extend. It does not provide nonlinear xCEBRA results, nor does it show that higher-order prefrontal areas are categorically modular. |
| Schneider, Lee & Mathis, *CEBRA* | Behaviour- and/or time-conditioned contrastive embeddings, consistency across runs/sessions, and downstream decoding; the original paper uses kNN decoding in its benchmarks. | It does not provide xCEBRA inverted-neuron-gradient attribution or a causal claim about this IBL analysis. |
| Schneider et al., *xCEBRA* | Regularized contrastive learning plus the Inverted Neuron Gradient; theoretical identifiability of zero/non-zero connectivity entries under stated generative assumptions and limiting conditions; synthetic and rat-data demonstrations. | The guarantees are not automatic for arbitrary finite-data neural recordings. They require matching assumptions, training, dimensionality, regularization, and validation. |

## Historical repository snapshot (4 September 2026)

The current checkout contains:

- 205 raw `.npz` files under `data/downloaded/`.
- 8 processed sessions under `xcebra_ibl/data_processed/`.
- 8 saved model metadata files under `xcebra_ibl/trained_models/`.
- 7,162 rows in `xcebra_ibl/results/xcebra_neuron_results.json`, spanning 8 sessions and 45 recorded acronyms.
- 14 area profiles in `xcebra_ibl/results/xcebra_selectivity_profiles.csv`.
- A strict RRR comparison is not currently reproducible from the local artifacts: the available `RRRglobal_full.json` files are missing or Git-LFS pointer files.

These numbers describe the current files, not the 14,000+ neuron / 43-cortical-region result in the baseline paper.

## Historical implementation audit (4 September 2026)

| Topic | Current implementation | Required before claiming the proposed contribution |
|---|---|---|
| Embedding | Separate CEBRA model per variable, `offset10-model`, 4 dimensions per variable, 32 dimensions in joint metadata; current saved runs report 500 iterations. | Implement and document the intended supervised/hybrid xCEBRA setup, including which dimensions are time-only and which are auxiliary-variable conditioned. |
| Jacobian regularization | `JACOBIAN_REG_WEIGHT` is defined in configuration but is not passed into the CEBRA training call. | Add the regularized objective and log the effective regularization value and fit diagnostics. |
| Attribution | `_jacobian_attribution` averages squared encoder-Jacobian gradients, `E[||∂f/∂x_n||²]`. | For an xCEBRA claim, compute and validate the pseudo-inverse Jacobian `J_f^+`, aggregate/threshold it as specified by the paper, and include synthetic recovery controls. |
| Decoding | `cross_validate_session` uses trial-based K-fold splits and Ridge regression on embeddings. | Add the planned kNN regressor/classifier benchmark, held-out metrics, baselines, confidence intervals, and leakage checks. |
| Clustering | Current analysis computes cosine similarity and hierarchical area clustering. | Add neuron-level k-means, reproducible-cluster filtering, silhouette scores, Gaussian null draws, multiple-comparison correction, and a pre-registered interpretation rule. |
| Dimensionality | Configuration has a fixed 4 dimensions per group; no dimension sweep or cross-seed consistency output is present. | Sweep candidate dimensions, train independent seeds, align embeddings, compute cross-seed `R^2`, and report the selected dimension with uncertainty. |
| Data split | Current CV splits trials within a session. The processed metadata does not retain a mouse-ID split. | If generalization across animals is claimed, retain subject IDs and split by mouse/session as appropriate; otherwise state the scope as within-session/trial-held-out. |

## Interpretation guardrails

1. “Attribution” is not automatically “causal attribution.” The xCEBRA paper defines a ground-truth connectivity target under a generative model; applying the method to recorded neural data supports an attribution hypothesis only to the extent that the assumptions and controls are satisfied.
2. Higher Silhouette Score alone does not prove biological specialization. It must exceed an appropriate null, survive the stated correction, and be robust to session composition, area size, preprocessing, and clustering choices.
3. A local linear fallback is useful for engineering, but it must be labelled as a linear baseline, not as the original RRR result.
4. Existing preliminary figures and JSON outputs are not final thesis evidence until their generating command, seed, data snapshot, and configuration are recorded.

## Immediate evidence priorities

1. Make the current pipeline scientifically executable end-to-end on a small controlled subset and fix the demo/training-path inconsistencies.
2. Add held-out decoding and baseline metrics before interpreting area structure.
3. Implement/validate regularized hybrid xCEBRA and Inverted Neuron Gradient on synthetic data with a known ground-truth graph.
4. Scale the validated pipeline to the complete intended IBL sample and archive configs, logs, seeds, and figures.
5. Update the graduation report after each evidence milestone, keeping preliminary and final results visibly separate.
