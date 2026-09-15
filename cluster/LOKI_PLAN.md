# Loki setup and training plan for review

Prepared 15 September 2026. Status: approved by the user with edits; execution
started. Submission deadline: 8 October 2026. The record at the end distinguishes
verified steps from work still planned.

## Proposed decisions

Use the existing installed dependencies if they pass compatibility checks. Place
code, data, caches and outputs in a permitted project directory on a suitable
mounted volume under `/media`. Run independent session jobs on the two allocated
P6000s, measure their actual throughput, then freeze a study that leaves time for
reruns and writing. Preserve the controlled scientific method and cohort reserve.

## 1 Connect and inspect storage

The user supplied a successful connection to `mohammadi@100.75.110.13`, whose
remote hostname is `loki`. The other supplied address timed out. After approval,
attempt the known working connection with existing authentication. If interactive
authentication is needed, the user completes it locally; no password is written
into a command, document or repository.

Connection command runs on the Mac:

```bash
ssh -o ConnectTimeout=10 mohammadi@100.75.110.13
```

Initial inspection commands run on Loki after the prompt changes to
`mohammadi@loki`:

```bash
hostname
id
ls -lahr /media
ls -ld /media/*
df -h / /home/mohammadi /media/*
df -i /media/*
findmnt -r -o TARGET,SOURCE,FSTYPE,OPTIONS
```

`ls` shows directory ownership and permission bits; `df` shows available filesystem
space. `du` measures occupied space, not permission or free capacity. Use bounded
`du -h --max-depth=1` only within a relevant permitted directory if needed; do not
recursively inventory other users' data across all mounted drives.

For candidate project locations, check parent traversal/write access, mount
read-only flags, quotas and ACLs (`getfacl` when available). A writable directory
must also be an appropriate lab-authorized location, not another user's workspace.
After selecting it, create only our project directory and verify a small temporary
file can be written and removed there. Do not change shared ownership/permissions.

If no suitable writable directory exists, report the exact path and permission
failure. The user will ask the supervisor for a directory or access. Continue
independent local preparation while that request is pending.

Acceptance: one verified writable project directory with enough capacity and a
record of its filesystem, quota if applicable, and free space. The earlier login
banner showed only 33 GB free on `/`, versus 3.2 TB on `/media/hdd`; recheck these
figures before allocating space.

## 2 Verify the installed environment and GPU allocation

Record the active Python executable/version, environment location, `pip check`,
package versions, CPU cores, RAM and available scheduler/session tools. Verify
PyTorch, CEBRA and the scientific packages against the training recipe. Do not
reinstall working dependencies simply because they were installed separately.

Check PyTorch's CUDA build, visible GPU names, compute capabilities and compiled
architectures, then execute small real CUDA operations. P6000 needs a compatible
Pascal build; the intended pinned package is PyTorch 2.10.0 from the CUDA 12.6
wheel index. The driver reporting CUDA 13.0 does not select the application runtime.
If installed packages differ, assess and document the difference before freezing
the environment. Repair incompatibilities inside an isolated environment.

Identify the two P6000s by GPU UUID. The supplied snapshot lists them at physical
indices 0 and 2; the K620 at index 1 is excluded. Establish whether both GPUs are
allocated to this account, for how long, and whether a scheduler governs use.
Idle GPUs are not proof of allocation. Do not alter drivers or reboot the server.

Acceptance: a recorded environment that runs CUDA on each allocated P6000, a
confirmed persistence mechanism for jobs, and known CPU/RAM/GPU resource limits.

## 3 Publish and retrieve the intended code version

The readiness changes remain uncommitted locally. A clone of GitHub alone will
not contain them. After plan approval, review the task's diff, run the local
checks, and prepare a release commit containing only the intended code, tests,
configuration and documentation. Push the reviewed release and record its SHA.
Preserve unrelated local files; exclude credentials, environments, raw data and
generated outputs from Git.
User edit accepted: automatic selection uses CUDA, otherwise MPS, otherwise CPU.
CEBRA 0.6 already implements that order. Add a regression check and record the
resolved device in each experiment manifest; explicit CUDA remains required for
the allocated GPU jobs so a missing accelerator fails early.
Clone `https://github.com/Helialiaia01/NEUROAI.git` inside the verified project
directory on Loki and check out that exact SHA. If a checkout already exists,
inspect it first and preserve local changes. Verify required scripts and matching
commit identity before installing the project into the chosen environment.
User edit accepted: if cloning fails, transfer the exact committed release over
SSH (for example, a Git bundle), verify its SHA, and transfer raw data separately.
Acceptance: the same reviewed source version on the Mac and Loki, with an
auditable environment snapshot. No training while its source files are changing.

## 4 Locate or transfer the data

Cloning source does not supply the Git-ignored dataset. The local raw data occupies
approximately 72 GB (`du -sh data/downloaded`, checked 15 September). First locate
an existing authorized copy on Loki or mounted lab storage. If absent, transfer
the selected calibration sessions first using resumable copying, then the complete
required cohort. Verify session IDs, file counts and SHA256 hashes.

Budget separately for raw data, environments/caches, worker outputs, merged copies,
analysis and one rerun. Current merging copies session artifacts; do not count the
merge as free space. Measure output size on representative sessions before the
full run, including embeddings and encoding predictions. Keep persistent outputs
on the chosen volume and record storage consumption throughout execution.

Acceptance: verified calibration data first, followed by the required complete
cohort without a second unnecessary raw-data copy. Transfer time enters the budget.

## 5 Use both P6000s without changing the scientific method

The cards do not automatically combine into a single 48 GB device. A model can
use multiple GPUs only through explicit parallel implementation. Data parallel
training replicates the model and synchronizes gradients; it does not simply pool
their memory. The current CEBRA runner is a single-GPU implementation.

Recommended approach: two independent workers, each owning a different session's
full seed/dimension/control grid. Give each process visibility of just its assigned
P6000, preferably by UUID, and use `--device cuda`. In a scheduler, respect the
visibility assigned by the scheduler rather than overriding it with host indices.
Keep one shared immutable job manifest, disjoint job indices and separate outputs.
Limit CPU threads so simultaneous CPU baselines do not oversubscribe the machine.

Use the existing per-session job mechanism. Check or add a small reliable dispatch
wrapper after review if needed to keep both workers fed, record exit status and
retry interrupted jobs without dispatching the same session twice. Use the lab's
scheduler if present; otherwise a permitted persistent terminal/session mechanism.

Benchmark one worker and two concurrent workers. Throughput could approach twice
the single-worker rate when GPU work dominates, but CPU, disk and scheduling can
reduce that benefit. We will report measured speedup rather than assume 2x.

A distributed single-model rewrite is deferred: contrastive sampling, loss
normalization and Jacobian regularization need additional correctness checks.
It offers no established deadline advantage for this many-independent-fits study.

Sources: [PyTorch 2.10 distributed data parallel documentation](https://docs.pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html),
[NVIDIA device visibility documentation](https://docs.nvidia.com/deploy/topics/topic_5_2_1.html).

## 6 Test and measure before scaling

1. Run the 24-test suite on Loki's actual environment. Prior Mac success is not
   evidence of Linux/CUDA success.
2. Run the tiny worker/merge integration and an actual short real-session CUDA
   smoke test that exercises regularization, attribution, saving and recovery.
3. Time 500- and 1,000-iteration runs on small and larger exploratory sessions.
   Capture total session time, fit times, VRAM, CPU RAM, warnings and output size.
   Measure dual-worker contention as well. Fixed CPU baselines mean total runtime
   does not scale exactly in proportion to iteration count.
4. Run the 1–3-session controlled pilot and longer synthetic attribution diagnostic.
   Check losses, validation decoding, null controls, attribution stability and
   synthetic support recovery. The earlier AUROC 0.47–0.78 is not a passed recovery
   benchmark. Do not tune the method on reserved-session test results.
5. Freeze dimension and training settings using exploratory/validation evidence.
   Keep the dimension sweep in the pilot; use the frozen dimension for the main
   study. Keep three seeds, matched baselines and retrained null controls unless
   a scientifically justified design change is explicitly reviewed.
6. Recompute time/storage budgets before committing to the main cohort. Test
   interruption/recovery on a disposable calibration run first. Completed variables
   are reusable; an interrupted variable restarts from its seed, not optimizer step.

Acceptance: finite complete outputs, validated recovery/merge, measured hardware
compatibility, and a study budget with time reserved for a rerun and final analysis.
If attribution remains weak, preserve and report that limitation; useful decoding
must not be presented as evidence of biological support recovery.

## 7 Deadline and decision points

| Dates | Target |
|---|---|
| 15–16 September | Storage/access, source release, installed-environment verification and first timed GPU checks |
| 17–19 September | Representative pilot, recovery test, measured budget and frozen main design |
| 20–25 September | Target completion of first main pass |
| 26–30 September | Reserved reruns, corrections and analysis checks |
| 1–7 October | Final figures, interpretation, writing and submission checks |
| 8 October | Submission deadline |

These are proposed targets, not promised execution times. Adjust them from actual
allocation and calibration; do not silently consume the rerun or writing reserve.
At one frozen dimension, eight variables, three seeds and one null, 205 sessions
require up to 9,840 encoder fits before exclusions. The 3-session/3-dimension
pilot requires up to 432 fits.

For illustration only, two continuously available GPUs and an average complete
session cost of one hour imply 4.3 days for 205 sessions, or 8.5 days including one
full rerun. At two hours per session those figures become 8.5 and 17.1 days.
These calculations exclude transfer, final clustering and downtime. If calibration
exceeds the first-pass budget, review allocation or scientific scope before launch;
do not remove controls or reduce iterations blindly.

Merge only validated worker outcomes, retain explicit preprocessing exclusions,
and run final clustering on the merged study with its stated correction family.
Keep published-RRR comparison conditional on obtaining the genuine reference.
Unknown subject IDs still limit animal-level claims.

## Review and later writer handoff

The user approved this plan and subsequently authorized the available
**GPT-5.6 Sol, medium** writer instead of the unavailable GPT-6 Sol setting.
Give the writer a bounded documentation task, the approved plan, frozen methods
and verified results. The writer must distinguish planned runs from completed
results and must not change training design, access the cluster or invent findings.

## Execution record

- SSH succeeded; `/media/hdd/mohammadi/thesis` was created privately and passed
  an actual temporary-file write/read/remove check. `/media/hdd` has about 3.2 TB
  free; root has about 19 GB free (96% used). No supervisor permission is needed
  for this verified project path.
- Loki has 40 logical CPUs and 110 GiB RAM, with about 85 GiB available during
  inspection. `tmux` is installed; `sbatch` and `srun` were not on PATH.
- User authorized assuming both P6000s available with no CPU/time limits; this
  is an explicit working assumption, not a supervisor-confirmed allocation.
- Writer started with the authorized GPT-5.6 Sol medium setting and is restricted
  to the runbook. Local regression suite now passes 25 tests, including automatic
  CUDA/MPS/CPU selection and failure for unavailable explicitly requested CUDA.
- Initial dependency discovery used system Python 3.12, which was not the Python
  in which the user installed Torch; the correct executable is
  `/opt/miniconda/bin/python`.
- The exact release `14ab2b2384a9491df4f08e98d6d9e48bd9047a33` was pushed and
  cloned on Loki. Pinned dependencies installed in `.venv313`; `pip check`, all
  25 Loki tests and the 32-fit worker/merge integration passed.
- Both P6000s passed forward, backward and second-order CUDA operations using
  Torch 2.10.0+cu126. Real-session CUDA calibration remains pending its data.
- The 201 MB calibration NPZ compresses to about 6 MB. A compressed payload was
  prepared locally after stopping the slower uncompressed transfer. Automatic
  approval review then blocked the transfer restart because the Codex account hit
  its usage limit. No workaround was attempted; the partial remote file remains.
