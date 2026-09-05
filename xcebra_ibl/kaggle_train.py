"""Kaggle-safe entry point for the xCEBRA IBL training pipeline.

The Kaggle launcher imports this module after cloning the GitHub repository.
The pipeline itself remains the same as the local CLI, but its output is
captured in ``/kaggle/working/train.log`` for download after the run.
"""

from __future__ import annotations

import os
import shlex
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


class _Tee:
    """Write a stream to both Kaggle's live log and a persistent output file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _pipeline_args(argv=None):
    if argv is not None:
        return list(argv)
    configured = os.environ.get("KAGGLE_PIPELINE_ARGS")
    if configured:
        return shlex.split(configured)
    # Measure cost on one session before dispatching the full controlled grid.
    return ["--max-sessions", "1", "--seeds", "2025", "--dimensions", "4",
            "--iterations", "500"]



def main(argv=None):
    """Run the configured pipeline and fail the Kaggle job on any exception."""
    out_dir = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    old_argv = sys.argv
    try:
        sys.argv = ["xcebra_ibl.kaggle_train", *_pipeline_args(argv)]
        with log_path.open("w", encoding="utf-8") as log_file:
            tee = _Tee(sys.__stdout__, log_file)
            error_tee = _Tee(sys.__stderr__, log_file)
            with redirect_stdout(tee), redirect_stderr(error_tee):
                print("xCEBRA IBL Kaggle run")
                print(f"KAGGLE_INPUT_DIR={os.environ.get('KAGGLE_INPUT_DIR', '')}")
                print(f"KAGGLE_WORKING_DIR={out_dir}")
                print(f"Pipeline arguments: {' '.join(sys.argv[1:])}")

                if os.environ.get("KAGGLE_EXPERIMENT_MODE", "pilot") == "legacy":
                    from xcebra_ibl.run_pipeline import main as pipeline_main
                    pipeline_main()
                else:
                    from xcebra_ibl.experiments import main as pilot_main
                    pilot_main(sys.argv[1:])

                # A training run with no discoverable sessions would otherwise
                # exit successfully after printing a warning. Make that state
                # visible to Kaggle as an error.
                args = sys.argv[1:]
                if "--train" in args or "--all" in args or "--stream" in args:
                    results_path = out_dir / "results" / "xcebra_neuron_results.json"
                    if not results_path.exists():
                        raise RuntimeError(
                            f"Training produced no results at {results_path}; "
                            "check the mounted dataset and preprocessing log."
                        )
                print("Kaggle pipeline completed successfully.")
    except Exception:
        # Persist the traceback before the exception reaches the launcher.
        with log_path.open("a", encoding="utf-8") as log_file:
            traceback.print_exc(file=log_file)
        raise
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
