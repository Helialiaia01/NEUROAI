# Published RRR integration decision — 25 September 2026

## Authoritative material

Dr Shuqi Wang provided the published
[`RRR_selectivity.json`](https://github.com/realwsq/brainwide-RRR-encoding-model/blob/main/trained_RRR_model/RRR_selectivity.json),
the [RRR training and trial-split code](https://github.com/realwsq/brainwide-RRR-encoding-model/blob/main/utils/RRRGD_main_CV.py#L110),
the [preprocessing code](https://github.com/realwsq/brainwide-RRR-encoding-model/blob/main/utils/save_and_load_data.py),
and the [published Methods](https://www.nature.com/articles/s41586-026-10668-4#Sec17).
The downloaded Git-LFS media artifact is 1,270,848,264 bytes with SHA-256
`e73993443982e3df5c0d0c86c7f032ce1cdeddf1d5f18fa16804d19df1442042`.

The artifact contains 59,820 neurons from 178 sessions. Its released fields are
`eid`, `uuids`, `acronym`, `RRR_r2`, `RRR_beta`, `RRR_U`, `RRR_V`, `RRR_b`, and
`null_r2`. The authors' code defines selective neurons as
`RRR_r2 - null_r2 > 0.015`. For the area-level magnitude result it removes the
last, intercept row of `RRR_beta` and computes `sum(abs(beta), time)`. For
single-neuron clustering it instead sums signed coefficients over time.

## Valid comparison

CEBRA Jacobian attribution is an unsigned sensitivity magnitude. It is therefore
compared with the paper's unsigned coefficient magnitude. The signed RRR
clustering input cannot be treated as equivalent to an unsigned Jacobian.

Published `RRR_r2` measures behaviour-to-neural encoding. The CEBRA decoding
score measures neural-embedding-to-behaviour prediction. Their numerical values
answer different questions and are not ranked against each other. CEBRA decoding
is compared with the locally fitted linear decoder using identical inputs and
splits. Published RRR supplies an external selectivity-structure comparison.

## Data-version audit

The frozen 205-session cohort shares 141 session IDs with the published RRR
artifact. In the completed 46 analyzable sessions, 37 session IDs overlap, but
only 54 unit UUIDs match and 31 pass the published delta-R2 filter; all matched
units come from one session. The official repository warns that public IBL data
can differ across versions. The current downloads and the paper's archived spike
sorting therefore do not support a broad exact-neuron comparison.

Two analyses are retained:

1. exact-UUID neuron-level correlations, reported with their small coverage;
2. session-balanced area-level correlations between local CEBRA profiles and the
   full published RRR artifact, reported over shared cortical areas.

For the completed subset, 35 cortical areas are shared. Side and lick have
uncorrected positive correlations, but none of the eight area-profile tests pass
Benjamini-Hochberg correction. These are preliminary subset results and must be
recomputed after all sessions finish.

## Training decision

No CEBRA retraining is required because the correction changes only the external
comparison loader and analysis. Restarting the active GPU run would not recover
the paper's historical unit UUIDs. A true broad neuron-matched rerun would require
the exact archived IBL input files used by the authors. Given the deadline, this
is optional follow-up work rather than a prerequisite for the current full study.
