# Kaggle GPU workflow

This project uses a private Kaggle dataset containing the local raw IBL
session exports and a two-file script kernel. The kernel clones the public
GitHub repository at HEAD, installs `requirements.txt`, and runs the real
`xcebra_ibl` pipeline on an `NvidiaTeslaT4`.

The current IDs are:

- Dataset: `helialiaia/ibl-xcebra-data`
- Kernel: `helialiaia/xcebra-ibl-gpu-train`
- GitHub repo: `https://github.com/Helialiaia01/NEUROAI.git`

## First upload

From the repository root:

```bash
python scripts/prepare_kaggle_dataset.py
kaggle datasets create -p data/downloaded
kaggle datasets status helialiaia/ibl-xcebra-data
```

The dataset directory is ignored by Git. The metadata helper only creates
`data/downloaded/dataset-metadata.json`; it does not copy the 72 GB dataset.

## Push code and run

Push code changes using the local GitHub token without printing it:

```bash
git -c credential.helper='!f() { printf "username=x-access-token\\npassword=%s\\n" "$(cat .secrets/gh)"; }; f' push origin main
kaggle kernels push -p kaggle/launcher
scripts/kaggle_wait_and_fetch.sh
```

The default run performs preprocessing, per-variable training, and analysis.
For a smaller validation run, set the pipeline arguments before pushing a new
kernel version, for example:

```bash
export KAGGLE_PIPELINE_ARGS="--preprocess --train --analyze --n-attribution-samples 100 --max-iterations 100"
```

Kaggle does not persist shell environment variables between runs. For a
repeatable non-default configuration, encode the value in the kernel's
Secrets/environment setup or update the launcher before pushing.

Outputs are written under `/kaggle/working`, including `train.log`, results,
preprocessed caches, final trained model files, and a solver checkpoint after
every CEBRA mini-batch under `trained_models/checkpoints/<eid>/`. Downloaded
artifacts go to `results/kaggle` by the helper script. Set
`checkpoint_frequency` in the model/training call if a less frequent local
checkpoint cadence is needed.
