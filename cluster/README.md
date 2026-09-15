# Portable training jobs

No cluster job is submitted by these files. The Synology invitation supplies a
storage account, not a verified compute hostname, scheduler or GPU allocation.

The job unit is **one session with its full seed/dimension/control grid**. This
is a reviewed change from splitting every fit into a separate scheduler job:
preprocessing and CPU baselines run once per session, model selection has all its
candidates, and each worker owns its output directory. Completed individual
variables are recovered inside that job. This avoids races and repeated baseline
work. If measured session cost exceeds allocation wall time, reduce the grid or
use repeated runs with per-variable recovery before adding finer-grained workers.

The provided `xcebra_ibl/configs/cohort.json` contains 164 exploratory and 41
reserved sessions, allocated deterministically with seed 20260909. Sessions with
historical model directories were excluded from the reserve. Subject IDs are
currently null; populate them only from verified IBL metadata, then freeze a new
cohort before running. Reservations are a protocol, not a filesystem access ban.
No existing data were moved or deleted.

Create a small calibration cohort by selecting one exploratory session from the
frozen cohort and preserving its ID/phase/subject. Alternatively use the normal
CLI `--session-ids` option to select a subset of the full cohort. Do not dispatch
all 164 exploratory sessions for a pilot.

For scheduler-neutral jobs, write a cohort JSON containing the intended 1–3 pilot
session entries, then prepare:

```sh
python -m xcebra_ibl.jobs prepare --cohort my_pilot_cohort.json --study xcebra_ibl/configs/pilot.json --output outputs/job_plan
python -m xcebra_ibl.jobs run --manifest outputs/job_plan/jobs.json --index 0 --data-dir /path/to/raw --root /path/to/workers
```

The example study requires CUDA. Local CPU verification should use a separate
study file with `--device cpu`, few iterations and a small grid.

On a Slurm cluster, set `THESIS_REPO`, `THESIS_PYTHON`, `THESIS_JOB_MANIFEST`,
`THESIS_DATA`, and `THESIS_OUTPUT`. Submit `cluster/slurm_worker.sh` as an array
with the lab's actual partition/account/GPU/time/memory settings. The script has
no invented resource directives. Shell paths are quoted and the worker propagates
failures. SIGTERM/SIGINT records interruption and retains completed variables.

After every planned worker has completed:

```sh
python -m xcebra_ibl.jobs merge --manifest outputs/job_plan/jobs.json --root /path/to/workers --output /path/to/merged
python -m xcebra_ibl.analysis.pilot --input /path/to/merged --output /path/to/analysis --null-draws 999
```

Merge checks source completion hashes and consistent code, package versions and
scientific configuration. Missing/failed sessions cause an error; no silent
partial aggregate is produced. Verified preprocessing exclusions are included with
their reasons and excluded from score aggregation. Every worker manifest is saved
in `worker_manifests.json`; the merge manifest records the common code fingerprint,
packages and excluded sessions. The output directory must be empty. Published RRR
comparison is explicitly requested with `--rrr /path/to/RRRglobal_full.json`; a
placeholder, missing required columns, ambiguous UUIDs or inconsistent areas is
an error. It is never substituted with the local Ridge fit.
