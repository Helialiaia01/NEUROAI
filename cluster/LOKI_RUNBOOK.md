# Loki GPU runbook

Prepared 15 September 2026. This is an operator checklist for the approved Loki
pilot workflow. Commands marked **Mac** run on the local Mac. Commands marked
**Loki** run only after the prompt is `mohammadi@loki`.

## Current status

Verified:

- `ssh mohammadi@100.75.110.13` works noninteractively.
- `/media/hdd` is read-write ext4 with about 3.2 TB free. The private directory
  `/media/hdd/mohammadi/thesis` exists, and a temporary write/read/remove test
  passed there.
- The selected training interpreter is `/opt/miniconda/bin/python` at Python
  3.13.2. The system Python 3.12.3 is not the training interpreter.
- User Torch is 2.10.0+cu126 at
  `/home/mohammadi/.local/lib/python3.13/site-packages`. CUDA is available, and
  the build contains sm50, sm60, sm70, sm75, sm80, sm86 and sm90 support.
- Loki has 40 logical CPUs, 110 GiB RAM (about 85 GiB available), `nohup` and
  `tmux`. Use `nohup` for real training as requested. No `sbatch` or `srun` was
  found.
- The two intended P6000s are
  `GPU-ddcddfcb-9e9a-2fec-a848-077ca2c870b5` and
  `GPU-13cc3970-fd10-b6f2-630b-75459933e640`. Exclude the K620.

Working assumption authorized by the user: both P6000s are available with no
time or CPU limit. This has not been confirmed by the supervisor or a scheduler.

The 26 local tests and the same 26 tests on Loki pass. The pinned scientific
dependencies installed successfully in `.venv313`, and `pip check` reports no
broken requirements. Both P6000s passed forward, backward and second-order CUDA
operations. The tiny Loki integration also passed 32 encoder fits, recovery,
artifact integrity, merge, subject aggregation, explicit exclusions and offline
analysis. The first real calibration completed in 704.99 seconds with exit status
0, 145 MB of output, 1,884,418,048 bytes peak process RSS, 130 verified artifacts
and no captured warnings. It revealed an unsupported lick target split, leading
to the raw-target support safeguard described below. Still pending: a calibration
rerun under that corrected release, interruption recovery under load, and
scientific pilot acceptance.

Loki is checked out at exact release
`161df97f2a04819d10796dcccaf7a677c5380156`, which includes the two-GPU
dispatcher, its regression check and the GPU-selectable synthetic recovery
diagnostic.

The root filesystem has only about 19 GB free and is 96% used. Keep the checkout,
environment, data, temporary files, caches and outputs under `/media/hdd`.

## 1. Connect and set the workspace

**Mac**

```sh
ssh -o ConnectTimeout=10 mohammadi@100.75.110.13
```

**Loki**

```sh
cd /media/hdd/mohammadi/thesis
mkdir -p cache/pip cache/torch cache/xdg tmp outputs data
source /media/hdd/mohammadi/thesis/activate.sh
df -h / /media/hdd
```

The activation script selects `/media/hdd/mohammadi/thesis/.venv313`, redirects
pip, Torch, XDG and temporary caches to the HDD, and limits CPU threads to four per
worker. Source it in every new shell. Do not redirect shared system caches or
change shared permissions.

## 2. Publish and retrieve one exact source version

The local readiness changes must be reviewed, tested, committed and pushed before
Loki clones them. Record the resulting full commit SHA as `RELEASE_SHA`.

**Mac, from `/Users/helialiaia/ACSAI/NeuroAI/thesis`**

```sh
git status --short --branch
python -m unittest discover -s tests -v
git rev-parse HEAD
```

Review the intended diff, create the release commit, push it, then record its SHA:

```sh
git rev-parse HEAD
```

**Loki**

```sh
cd /media/hdd/mohammadi/thesis
source /media/hdd/mohammadi/thesis/activate.sh
git clone https://github.com/Helialiaia01/NEUROAI.git
cd NEUROAI
git fetch --all --tags
git checkout --detach RELEASE_SHA
git rev-parse HEAD
```

Replace `RELEASE_SHA` with the recorded full SHA and require the two machines to
match. If the clone fails, transfer a Git bundle or copy over SSH, preserving that
exact commit; verify `git rev-parse HEAD` again before continuing. Inspect and
preserve any existing checkout instead of overwriting it.

## 3. Transfer and verify data separately

Git does not include the approximately 72 GB in `data/downloaded`. First check for
an authorized existing copy on Loki. If none exists, transfer calibration sessions
first, then the required cohort with a resumable tool.

**Mac, example full transfer**

```sh
rsync -a --info=progress2 --partial data/downloaded/ mohammadi@100.75.110.13:/media/hdd/mohammadi/thesis/data/downloaded/
```

For these internally uncompressed NPZ files, SSH compression is substantially
faster. The prepared calibration payload can be sent from the Mac with:

```sh
scp /tmp/data_044be2f4-e898-404c-91e2-1285cbada2cd.npz.gz mohammadi@100.75.110.13:/media/hdd/mohammadi/thesis/data/downloaded/
```

Then on Loki, decompress and verify it without overwriting an existing final NPZ:

```sh
cd /media/hdd/mohammadi/thesis/data/downloaded
gzip -dc data_044be2f4-e898-404c-91e2-1285cbada2cd.npz.gz > data_044be2f4-e898-404c-91e2-1285cbada2cd.npz.new
sha256sum data_044be2f4-e898-404c-91e2-1285cbada2cd.npz.new
mv data_044be2f4-e898-404c-91e2-1285cbada2cd.npz.new data_044be2f4-e898-404c-91e2-1285cbada2cd.npz
```

Require SHA256 `45e19684bcffdef9490d9fe498525404029f7456067611dc4cdd853f051ef6f5`
before the `mv` command. Delete the compressed copy only after the final file verifies.

Before training, compare session IDs and file counts and verify SHA-256 hashes for
the selected files on both machines. Avoid a second raw-data copy.

## 4. Complete and gate the training environment and both GPUs

The HDD environment `.venv313` uses `--system-site-packages` so it can reuse the
verified user Torch described above. It is therefore not isolated from that user
Torch dependency: record Torch's exact path and version with every environment
snapshot. The pinned scientific dependencies are installed in this venv and reuse
the working Torch. Do not replace that Torch unless a compatibility check requires
a reviewed change.

**Loki — completed dependency installation command**

```sh
source /media/hdd/mohammadi/thesis/activate.sh
cd /media/hdd/mohammadi/thesis/NEUROAI
python -m pip install -r environments/requirements-training.txt
python -m pip install -e .
```

**Loki**

```sh
source /media/hdd/mohammadi/thesis/activate.sh
cd /media/hdd/mohammadi/thesis/NEUROAI
which python
python --version
python -m pip --version
python -m pip check
python -m pip freeze
nvidia-smi -L
python -c 'import torch, cebra; print("torch", torch.__version__, "torch path", torch.__file__, "cuda build", torch.version.cuda, "cuda available", torch.cuda.is_available(), "architectures", torch.cuda.get_arch_list(), "cebra", cebra.__version__)'
```

Then run a small real tensor operation once with each P6000 visible:

```sh
CUDA_VISIBLE_DEVICES=GPU-ddcddfcb-9e9a-2fec-a848-077ca2c870b5 python -c 'import torch; x=torch.ones(1024,device="cuda"); print(torch.cuda.get_device_name(), float((x*x).sum()))'
CUDA_VISIBLE_DEVICES=GPU-13cc3970-fd10-b6f2-630b-75459933e640 python -c 'import torch; x=torch.ones(1024,device="cuda"); print(torch.cuda.get_device_name(), float((x*x).sum()))'
```

Stop if either command does not use a P6000, CUDA fails, `pip check` fails, Torch
does not resolve to the recorded user installation, or the installed package
combination conflicts with the pinned recipe. Resolve other dependencies inside
`.venv313` under `/media/hdd/mohammadi/thesis`; never install the macOS freeze
wholesale on Linux and do not alter drivers or reboot.

## 5. Validate before any long run

The unit suite and tiny CPU integration in this section passed on Loki. A
1,000-iteration synthetic CUDA diagnostic also completed on one P6000 in 553.0
seconds, writing 3.2 MB with finite recorded diagnostics. Mean test R2 was 0.778
for linear decoding and 0.819 for k-NN. Attribution recovery remained variable
(AUROC 0.472--0.861, mean 0.663; average precision 0.501--0.856, mean 0.692),
and the 0.01 Jacobian penalty did not improve mean recovery in this two-seed toy
test. Treat that as a reason to retain repeated seeds and uncertainty reporting.
It is not evidence about IBL biology. The first real-session CUDA preflight and
500-iteration calibration passed operationally. The calibration also found that
the session had one raw lick-event trial in train, three in validation and none in
test. The earlier run therefore produced invalid, extreme negative lick R2 values.
The corrected pipeline checks raw target variation separately in every split,
records support in `qc.json` and `scores.json`, and excludes unsupported
session-variable pairs from fitting and scoring. Require a fresh output directory
after this code change.

**Loki, from the exact release checkout and selected environment**

```sh
source /media/hdd/mohammadi/thesis/activate.sh
cd /media/hdd/mohammadi/thesis/NEUROAI
python -m unittest discover -s tests -v
python -m scripts.verify_pipeline --output /media/hdd/mohammadi/thesis/outputs/new_cpu_verification
python -m xcebra_ibl.experiments --cohort xcebra_ibl/configs/cohort.json --session-ids 044be2f4-e898-404c-91e2-1285cbada2cd --preflight-only --device cuda --output /media/hdd/mohammadi/thesis/outputs/preflight
```

Preflight records configuration and CUDA availability but does not execute a GPU
kernel. The tensor checks above and the calibration below provide actual GPU
evidence. CEBRA 0.6 otherwise resolves CUDA, then MPS, then CPU when asked for an
available device, so retain explicit `--device cuda` on Loki jobs to prevent a
silent CPU fallback. Use a new empty output directory whenever code, input,
configuration or environment changes.

## 6. Calibrate on one P6000

**Loki — proposed run; execute only after Sections 1–5 pass**

```sh
source /media/hdd/mohammadi/thesis/activate.sh
cd /media/hdd/mohammadi/thesis/NEUROAI
mkdir -p /media/hdd/mohammadi/thesis/logs /media/hdd/mohammadi/thesis/run
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=GPU-ddcddfcb-9e9a-2fec-a848-077ca2c870b5 python -m xcebra_ibl.experiments --cohort xcebra_ibl/configs/cohort.json --data-dir /media/hdd/mohammadi/thesis/data/downloaded --session-ids 044be2f4-e898-404c-91e2-1285cbada2cd --seeds 2025 --dimensions 4 --iterations 500 --device cuda --output /media/hdd/mohammadi/thesis/outputs/gpu_calibration > /media/hdd/mohammadi/thesis/logs/gpu_calibration.log 2>&1 < /dev/null &
echo $! > /media/hdd/mohammadi/thesis/run/gpu_calibration.pid
```

Confirm finite outputs, regularization and attribution execution, warnings,
recovery behavior, elapsed time, peak VRAM, CPU RAM and output size. Repeat a
representative timing at 1,000 iterations before estimating the full budget.

## 7. Run the controlled pilot with two independent workers

Use one independent full-session-grid worker per P6000. Do not split one model
across GPUs and do not introduce DDP. Each worker must have one GPU UUID, disjoint
session IDs, a shared immutable manifest and a separate output directory. Use
`nohup` with a log and PID file for persistence because no Slurm commands were
found. Limit CPU threads if the two workers contend during CPU baselines.

The approved three-session pilot is three seeds, dimensions 2/4/8 and 500
iterations: 432 encoder fits before exclusions. Assign whole sessions between the
two workers through the repository's portable job mechanism; do not manually
divide a session's grid. First benchmark one worker, then two concurrent workers,
and record measured throughput rather than assuming a 2x speedup.

After preparing `jobs.json`, launch the tested dispatcher through `nohup`. The
log and PID file allow the SSH session to close without terminating training:

```sh
source /media/hdd/mohammadi/thesis/activate.sh
cd /media/hdd/mohammadi/thesis/NEUROAI
mkdir -p /media/hdd/mohammadi/thesis/logs /media/hdd/mohammadi/thesis/run
nohup env PYTHONUNBUFFERED=1 python -m xcebra_ibl.jobs dispatch --manifest JOB_PLAN/jobs.json --data-dir /media/hdd/mohammadi/thesis/data/downloaded --root /media/hdd/mohammadi/thesis/outputs/workers --gpus GPU-ddcddfcb-9e9a-2fec-a848-077ca2c870b5 GPU-13cc3970-fd10-b6f2-630b-75459933e640 > /media/hdd/mohammadi/thesis/logs/pilot_dispatch.log 2>&1 < /dev/null &
echo $! > /media/hdd/mohammadi/thesis/run/pilot_dispatch.pid
```

Replace `JOB_PLAN` with the prepared plan directory. The dispatcher gives each
process one GPU UUID, assigns each session index once, and stops scheduling new
sessions after a failure. The per-session recovery mechanism handles a safe rerun.

Check a detached job from a later SSH login:

```sh
PID=$(cat /media/hdd/mohammadi/thesis/run/pilot_dispatch.pid)
kill -0 "$PID" 2>/dev/null && echo RUNNING || echo NOT_RUNNING
tail -n 100 /media/hdd/mohammadi/thesis/logs/pilot_dispatch.log
nvidia-smi
```

`NOT_RUNNING` means the process exited; it does not by itself distinguish success
from failure. Require the job completion records and artifact integrity checks
before merging. Keep the log and PID file with the run record.

After all planned workers pass their completion and integrity checks, merge their
outputs once and run:

```sh
source /media/hdd/mohammadi/thesis/activate.sh
cd /media/hdd/mohammadi/thesis/NEUROAI
python -m xcebra_ibl.analysis.pilot --input MERGED_PILOT_OUTPUT --output /media/hdd/mohammadi/thesis/outputs/gpu_pilot_analysis --null-draws 99
```

Replace `MERGED_PILOT_OUTPUT` with the verified merged directory. Inspect losses,
gradients, validation decoding, retrained nulls, attribution stability, synthetic
support recovery, warnings, runtime and storage. Do not tune using reserved-session
test results. Freeze the dimension and settings only after pilot review; keep three
seeds, matched baselines and retrained null controls unless a scientific change is
explicitly reviewed.

## 8. Recovery and launch gate

Test interruption and recovery on disposable calibration output before scaling.
Completed variables may be reused; an interrupted variable restarts from its seed,
not from an optimizer step. Merge only validated worker results and retain worker
manifests and preprocessing exclusions.

Do not start the main cohort until both P6000s have passed real CUDA checks, the
pilot is scientifically accepted, dual-worker runtime and storage are measured,
and the schedule still preserves rerun and writing time before 8 October 2026.

## Transfer status

The first calibration session is 210,730,333 bytes with SHA256
`45e19684bcffdef9490d9fe498525404029f7456067611dc4cdd853f051ef6f5`.
The gzip payload and decompressed NPZ were transferred and verified on Loki. The
compressed SHA256 is
`b78a1ae86aeaaff277b77f8082304a7afdf87a2d435b20cc888ef12598efff1b`.
An earlier 66,650,112-byte partial NPZ was preserved as
`data_044be2f4-e898-404c-91e2-1285cbada2cd.npz.partial-66650112`; it is not a
training input and may be removed after the calibration result is secured.

The complete 205-session dataset was also archived locally after the sample check.
It contains 77,051,581,379 uncompressed bytes and is 2,516,271,304 bytes as
`/tmp/ibl_downloaded_205_2026-09-15.tar.gz`. `gzip -t` passed, the archive lists
205 NPZ files, and its SHA256 is
`004d9a8f2fcae8be372ab1891ad71c114314f8aaca252e7deca86cac248f0879`.
Once transfer is permitted, prefer this verified archive over sending 77 GB of
uncompressed NPZ containers. Send it from the Mac with resumable partial-file
retention:

```sh
rsync -avP /tmp/ibl_downloaded_205_2026-09-15.tar.gz mohammadi@100.75.110.13:/media/hdd/mohammadi/thesis/
```

On Loki, verify the archive hash and count before extraction:

```sh
cd /media/hdd/mohammadi/thesis
sha256sum ibl_downloaded_205_2026-09-15.tar.gz
tar -tzf ibl_downloaded_205_2026-09-15.tar.gz | grep -c '/data_.*\.npz$'
tar -xzf ibl_downloaded_205_2026-09-15.tar.gz -C data
```

Require the recorded SHA256 and count 205 before extraction. This produces
`data/downloaded`; retain the archive until file-count and calibration checks pass.
