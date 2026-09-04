"""Kaggle entry point: clone the GitHub repo and run its real pipeline."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO = "https://github.com/Helialiaia01/NEUROAI.git"
REPO_DIR = "/kaggle/working/repo"


def _find_dataset_dir():
    configured = os.environ.get("KAGGLE_INPUT_DIR")
    if configured:
        path = Path(configured)
        if path.exists():
            return path

    input_root = Path("/kaggle/input")
    for candidate in sorted(input_root.iterdir()):
        if candidate.is_dir() and next(candidate.rglob("data_*.npz"), None):
            return candidate
    raise FileNotFoundError(
        "No data_*.npz files found under /kaggle/input. "
        "Verify the dataset source in kernel-metadata.json."
    )


def _find_reference_dir():
    """Find the separately mounted Allen/RRR reference dataset."""
    configured = os.environ.get("XCEBRA_REFERENCE_DIR")
    if configured:
        path = Path(configured)
        if path.exists():
            return path

    input_root = Path("/kaggle/input")
    for candidate in sorted(input_root.iterdir()):
        if (
            candidate.is_dir()
            and (candidate / "area_list.csv").exists()
            and (candidate / "conn_cxcx.csv").exists()
            and (candidate / "RRRglobal_full.json").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "No Allen/RRR reference files found under /kaggle/input. "
        "Attach helialiaia/ibl-xcebra-reference to the kernel."
    )


def main():
    try:
        subprocess.run(["git", "clone", "--depth", "1", REPO, REPO_DIR], check=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", f"{REPO_DIR}/requirements.txt"],
            check=True,
        )

        os.environ["KAGGLE_INPUT_DIR"] = str(_find_dataset_dir())
        os.environ["XCEBRA_REFERENCE_DIR"] = str(_find_reference_dir())
        os.environ["KAGGLE_WORKING_DIR"] = "/kaggle/working"
        sys.path.insert(0, REPO_DIR)

        from xcebra_ibl.kaggle_train import main as train_main

        train_main()
    finally:
        # The cloned source tree is not an experiment artifact. Removing it
        # before Kaggle packages /kaggle/working preserves disk for results.
        shutil.rmtree(REPO_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
