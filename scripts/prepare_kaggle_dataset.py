"""Prepare metadata for uploading the local IBL NPZ export to Kaggle.

The data stays in its existing location; this only creates the metadata file
required by ``kaggle datasets create``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/downloaded"))
    parser.add_argument("--dataset-id", default="helialiaia/ibl-xcebra-data")
    parser.add_argument("--title", default="IBL xCEBRA Raw Sessions")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No .npz session files found in {data_dir}")

    metadata = {
        "title": args.title,
        "id": args.dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            f"Raw IBL session exports for the xCEBRA pipeline ({len(npz_files)} NPZ files)."
        ),
    }
    metadata_path = data_dir / "dataset-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metadata_path}")
    print(f"Found {len(npz_files)} NPZ session files")
    print(f"Upload with: kaggle datasets create -p {data_dir}")


if __name__ == "__main__":
    main()
