# Training environment

Use Python 3.13 and CEBRA 0.6.0, matching local execution. Direct scientific
package versions are pinned in `requirements-training.txt`. This is a reproducible
starting recipe, not a claim that Linux/CUDA has been tested on the allocated node.
`local-macos-freeze.txt` records all installed local distributions for diagnosis;
do not install that macOS snapshot wholesale on Linux.

On the compute host, create an isolated environment. Install torch 2.10.0 from
the official wheel index appropriate to the approved CUDA/driver/GPU combination,
then install `environments/requirements-training.txt` and `pip install -e .`.
Consult [PyTorch's installation guidance](https://pytorch.org/get-started/locally/).
The actual wheel/driver choice must follow the provided hardware; no CUDA index
or Slurm partition is assumed here. Save `python -m pip freeze` from the resulting
Linux environment with the experiment output before any long run.

Run:

```sh
python -m unittest discover -s tests -v
python -m xcebra_ibl.experiments --preflight-only --device cuda --output outputs/preflight
```

Preflight records CUDA availability, package and data hashes, resolved model and
preprocessing choices. It does not execute a GPU kernel. The first actual
calibration must verify GPU execution and peak VRAM. Use `--device cuda` for
allocated GPU jobs so CEBRA cannot silently fall back to CPU.

The default `--device cuda_if_available` follows CUDA, then MPS, then CPU. The
runner now records both the requested and resolved device; explicit unavailable
devices fail during preflight. This order is covered by a regression test.

The original broad `requirements.txt` supports legacy downloading and analysis.
The controlled pilot does not need every optional legacy integration (captum,
cvxpy, ONE, wandb, Hydra). Its default Kaggle wrapper installs the pinned training
requirements. The legacy wrapper continues to install the original requirements.

Local NumPy 2.2/macOS BLAS may emit matrix-operation warnings with finite outputs.
Both locally tested NumPy 2.2.0 and 2.2.6 showed them on real data.
The runner preserves emitted warning details/counts in each session's
`warnings.json` without changing warning filters; training gradients, diagnostics, predictions and saved
numeric JSON must be finite. Linux/CUDA numerical validation remains a separate
acceptance step.
