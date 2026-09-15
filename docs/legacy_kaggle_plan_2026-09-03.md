# Kaggle GPU Training via GitHub + Launcher Kernel — Implementation Plan (Option C)

> **Goal:** Run a multi-file training project on Kaggle GPU from a dev PC, keeping all real code in a GitHub repo and pushing only a tiny launcher script to Kaggle.

**Architecture:** The dev PC holds the dataset + the repo. The dataset is uploaded once to Kaggle as a private Dataset. All training code lives in a GitHub repo (private or public). A 10-line launcher script (`kernel_type: "script"`, so no notebook) is pushed with `kaggle kernels push`; when Kaggle runs it, it clones the repo at HEAD, imports the package, trains on the attached dataset, and writes artifacts to `/kaggle/working`. The dev PC polls status and downloads artifacts.

**Tech stack:** Kaggle CLI, GitHub (git), Python, PyTorch (or whatever framework), bash polling loop.

**Current values from this machine:** `helialiaia` = Kaggle username from `kaggle config view`; `NEUROAI` = GitHub repo name from `git remote -v`; `GH_TOKEN` = GitHub PAT secret name, please tell me the exact secret name you want to use; dataset and kernel slugs should be replaced with the actual Kaggle dataset/kernel IDs you create later. Example project: image classifier "flowers", but the flow is framework-agnostic.

---

## Constraints that shape this plan (verified against the Kaggle CLI source)

- `kaggle kernels push` uploads **exactly one file** (the `code_file`) + metadata. Other files in the folder are ignored. → All logic must come from the repo clone.
- Dataset files mount read-only at `/kaggle/input/<dataset-slug>/`.
- Only files written under `/kaggle/working/` become downloadable output (`kaggle kernels output`).
- Kaggle has **no live log streaming via CLI** and no shell. Progress must be tee'd to `/kaggle/working/train.log`.
- GPU: use `NvidiaTeslaT4` (`machine_shape`). **Do not use `NvidiaTeslaP100`** — the default PyTorch image (cu128) has no Pascal (`sm_60`) kernels; first CUDA op fails.
- Datasets and kernels default to **private**.
- Each `kaggle kernels push` = one queued run. Re-push same `id` → new version + new run.

---

## Phase 0 — Setup on the dev PC (one time)

**Files:** none (environment only)

1. Install CLI: `pip install kaggle`
2. Authenticate: `kaggle auth login` (browser OAuth) — or `export KAGGLE_API_TOKEN=...` from https://www.kaggle.com/settings/api
3. Verify: `kaggle kernels list -m` prints your (empty) kernel list without auth errors.
4. Create the GitHub repo `current directory` (public for now; private needs Phase 6 step). Note the HTTPS URL.
5. Create working dir on the dev PC:

```bash
mkdir -p ~/ml-sync/{flower-train,flowers-dataset,flowers-launcher,results}
```

**Acceptance:** auth works, repo exists on GitHub, four local dirs exist.

---

## Phase 1 — Repo layout & entry point

**Files:** all inside `~/ml-sync/flower-train/` (this whole folder gets `git push`ed)

```
current-proj/
├── pyproject.toml            # optional but recommended (pip install -e works)
├── requirements.txt          # exact pin list for the run
└── flower_train/
    ├── __init__.py
    ├── config.py             # paths, hyperparams
    ├── data.py               # loads /kaggle/input/<dataset-slug>/...
    ├── model.py
    ├── train.py              # `main()` — the real training loop
    └── resume.py             # (Phase 6) checkpoint discovery + resume
```

### Task 1: Write the code so Kaggle paths are parameters

In `config.py`, read paths from the environment, never hardcode a PC path:

```python
import os

DATA_DIR = os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input/flowers-dataset")
OUT_DIR  = os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")
CHECKPOINT_PATH = os.path.join(OUT_DIR, "model.pt")
LOG_PATH = os.path.join(OUT_DIR, "train.log")
```

This makes the same code runnable locally (set the env vars to local dirs) and on Kaggle.

### Task 2: Write `train.py::main()` with three hard rules

1. Read data from `DATA_DIR` (e.g. `os.path.join(DATA_DIR, "train.csv")` — or unzip first if you uploaded archives).
2. Write **checkpoints + a log** to `OUT_DIR`:

```python
import sys

def main():
    log = open(LOG_PATH, "w")                 # tee, don't rely on stdout
    ...
    print("epoch 3/20 loss=0.42", file=log, flush=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    ...
```

3. Exit code 0 only when finished; `sys.exit(1)` on failure so the run shows `error`, not `complete`.

### Task 3: Commit & push

```bash
cd ~/ml-sync/flower-train
git init && git add . && git commit -m "init: training package"
git branch -M main
git remote add origin https://github.com/bob/flower-train.git
git push -u origin main
```

**Acceptance:** `git ls-remote https://github.com/bob/flower-train.git` shows the branch.

---

## Phase 2 — Upload the dataset (one time)

**Files:** `~/ml-sync/flowers-dataset/` (data + metadata)

### Task 1: Create the upload folder

```
flowers-dataset/
├── dataset-metadata.json
├── train.csv          # or images, or archives
└── val.csv
```

`dataset-metadata.json`:

```json
{
  "title": "Flowers Training Split",
  "id": "bob/flowers-dataset",
  "licenses": [{"name": "CC0-1.0"}]
}
```

### Task 2: Upload & wait for processing

```bash
cd ~/ml-sync/flowers-dataset
kaggle datasets create -p .                      # private by default; add --public only if intended
kaggle datasets status bob/flowers-dataset       # poll until "complete"/processed — do NOT push the kernel before this
```

Notes:
- Subdirectories are **skipped** by default on upload. Nested data → `kaggle datasets create -p . -r zip` and unzip in code.
- Updating the data later is a new **version**, not a new create:
  `kaggle datasets version -p . -m "added 200 samples"`

**Acceptance:** `kaggle datasets status bob/flowers-dataset` reports processed/complete.

---

## Phase 3 — Launcher kernel (the only thing Kaggle stores)

**Files:** `~/ml-sync/flowers-launcher/` — two files total, nothing else in this folder.

`kernel-metadata.json`:

```json
{
  "id": "bob/flowers-train",
  "title": "Flowers Train",
  "code_file": "run_kaggle.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["bob/flowers-dataset"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

`run_kaggle.py` (public repo version):

```python
"""Kaggle entry point: clone repo at HEAD, import package, train."""
import subprocess
import sys

REPO = "https://github.com/bob/flower-train.git"

subprocess.run(
    ["git", "clone", "--depth", "1", REPO, "/kaggle/working/repo"],
    check=True,
)
sys.path.insert(0, "/kaggle/working/repo")

from flower_train.train import main  # noqa: E402

main()
```

> Keep this file **dumb and stable**. All evolution happens in the GitHub repo. If the repo is pip-installable, an equivalent is `pip install git+https://github.com/bob/flower-train.git` + plain `import flower_train`.

**Acceptance:** folder contains exactly `kernel-metadata.json` + `run_kaggle.py`.

---

## Phase 4 — First run (push → poll → fetch)

### Task 1: Push (this triggers the run)

```bash
cd ~/ml-sync/flowers-launcher
kaggle kernels push -p .
# expected: "Kernel version N successfully pushed. Please check progress at https://www.kaggle.com/code/bob/flowers-train"
```

### Task 2: Poll until terminal state

```bash
kaggle kernels status bob/flowers-train
# -> running ... then complete | error
```

### Task 3: Fetch artifacts

```bash
kaggle kernels files bob/flowers-train                    # list outputs (should include model.pt, train.log)
kaggle kernels output bob/flowers-train -p ~/ml-sync/results
```

**Acceptance:** `~/ml-sync/results/train.log` exists and shows epochs advancing; `model.pt` downloads. If status = `error`, open the kernel page (URL from the push output) and read the traceback — the repo clone line is the usual first failure point.

---

## Phase 5 — Iteration loop (daily workflow)

```bash
# 1. change code on the dev PC
cd ~/ml-sync/flower-train
git add -A && git commit -m "experiment: lr 1e-3, wider head"
git push

# 2. re-run on Kaggle (same launcher, fresh HEAD)
cd ~/ml-sync/flowers-launcher
kaggle kernels push -p .

# 3. wait & fetch
kaggle kernels status bob/flowers-train
kaggle kernels output bob/flowers-train -p ~/ml-sync/results -o
```

Optional helper `~/ml-sync/wait_and_fetch.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
KERNEL="bob/flowers-train"
while kaggle kernels status "$KERNEL" | grep -qiE 'running|pulling'; do sleep 60; done
echo "== terminal state =="
kaggle kernels files "$KERNEL"
kaggle kernels output "$KERNEL" -p ~/ml-sync/results -o
```

**Acceptance:** an edit → `git push` → `kaggle kernels push` cycle ends with new artifacts in `~/ml-sync/results` without touching the web UI.

---

## Phase 6 — Optional hardening

### Private repo

1. GitHub → Settings → Developer settings → Personal access token (repo scope). Save value.
2. Kaggle web UI → your kernel → **Add-ons → Secrets** → add `GH_TOKEN` = the PAT (no CLI for secrets).
3. Launcher becomes:

```python
import subprocess
import sys

from kaggle_secrets import UserSecretsClient  # only available inside the Kaggle runtime

token = UserSecretsClient().get_secret("GH_TOKEN")
repo = f"https://x-access-token:{token}@github.com/bob/flower-train.git"
subprocess.run(["git", "clone", "--depth", "1", repo, "/kaggle/working/repo"], check=True)
sys.path.insert(0, "/kaggle/working/repo")

from flower_train.train import main

main()
```

### Long runs / quota (weekly GPU quota ~30 h, per-run cap)

- Checkpoint every epoch into `OUT_DIR`; write a `resume.py` that scans `/kaggle/input/...` **or** re-downloads the previous kernel output via `kaggle kernels output` in the launcher before training, then starts from the last checkpoint.
- Keep each run under the cap by design; resume = re-push.

### Deterministic environment

Pin `requirements.txt`; optionally set `docker_image` / `docker_image_pinning_type` in kernel metadata to a pinned Kaggle image.

---

## Verification checklist (run before calling it done)

- [ ] `kaggle datasets status bob/flowers-dataset` → processed
- [ ] `kaggle kernels status bob/flowers-train` → complete (not error)
- [ ] `~/ml-sync/results/train.log` non-empty, epochs visible
- [ ] `~/ml-sync/results/model.pt` present and loadable
- [ ] Cycle proven twice: change → push repo → re-push kernel → new artifacts
- [ ] If private repo: second run succeeds with secret only (token never printed)

## Risks / gotchas recap

| Risk | Mitigation |
|---|---|
| Kernel is single-file by design | All logic in GitHub repo; launcher only clones + calls `main()` |
| P100 GPU fails with cu128 image | `machine_shape: "NvidiaTeslaT4"` |
| Dataset still processing on first push | `kaggle datasets status` before first kernel push |
| No CLI log streaming | Tee progress to `/kaggle/working/train.log`, download it |
| Repo URL / dataset slug mismatch | Dataset mounts at `/kaggle/input/<slug>` — match `dataset_sources` and `config.DATA_DIR` |
| Long run hits quota/time cap | Checkpoint + resume pattern (Phase 6) |
| Private-repo token in clone URL | Fine in ephemeral run; never `print()` it |

## Open questions for the user

1. Dataset format on disk: flat files or nested folders (`-r zip`)?
2. Framework and pinned versions (`requirements.txt`)?
3. Public or private GitHub repo?
4. Expected run length vs. Kaggle quota — is checkpoint/resume needed from day one?
