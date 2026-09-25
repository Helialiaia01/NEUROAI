"""Load the published Wang et al. RRR artifact without holding its 1.27 GB JSON in memory."""

from pathlib import Path

import numpy as np
import pandas as pd


RRR_METADATA = ("eid", "uuids", "acronym", "RRR_r2", "null_r2")
STREAM_THRESHOLD_BYTES = 256 * 1024**2


def _reject_lfs_pointer(path: Path) -> None:
    with path.open("rb") as handle:
        prefix = handle.read(100)
    if prefix.startswith(b"version https://git-lfs.github.com/spec"):
        raise ValueError("RRR input is a Git-LFS pointer, not model results")


def _stream_column(path: Path, column: str) -> dict:
    try:
        import ijson
    except ImportError as exc:
        raise ImportError(
            "Reading the full published RRR artifact requires ijson; "
            "install the project requirements before running the comparison"
        ) from exc
    with path.open("rb") as handle:
        return dict(ijson.kvitems(handle, column, use_float=True))


def read_rrr_metadata(path) -> pd.DataFrame:
    """Return authoritative identity and performance columns plus source row IDs."""
    path = Path(path)
    _reject_lfs_pointer(path)
    if path.stat().st_size <= STREAM_THRESHOLD_BYTES:
        try:
            frame = pd.read_json(path)
        except ValueError as exc:
            raise ValueError("RRR input is not a valid results table") from exc
        missing = set(RRR_METADATA).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing published RRR columns: {sorted(missing)}")
        frame = frame.loc[:, RRR_METADATA].copy()
        frame["_rrr_row"] = frame.index.astype(str)
        return frame.reset_index(drop=True)

    columns = {name: _stream_column(path, name) for name in RRR_METADATA}
    keys = list(columns[RRR_METADATA[0]])
    if any(list(columns[name]) != keys for name in RRR_METADATA[1:]):
        raise ValueError("Published RRR columns have inconsistent row identifiers")
    frame = pd.DataFrame({name: [columns[name][key] for key in keys] for name in RRR_METADATA})
    frame["_rrr_row"] = keys
    return frame


def read_rrr_magnitudes(path, row_ids, n_variables: int) -> dict[str, np.ndarray]:
    """Return sum-over-time absolute beta for selected source rows.

    Jacobian attribution is unsigned, so this matches the paper's area-level
    absolute-selectivity definition. The paper's clustering analysis instead
    sums signed beta, which is not directly comparable to an unsigned Jacobian
    magnitude.
    """
    path = Path(path)
    wanted = {str(row) for row in row_ids}
    if not wanted:
        return {}
    if path.stat().st_size <= STREAM_THRESHOLD_BYTES:
        frame = pd.read_json(path)
        if "RRR_beta" not in frame:
            raise ValueError("Missing published RRR column: RRR_beta")
        source = ((str(i), value) for i, value in frame["RRR_beta"].items())
    else:
        try:
            import ijson
        except ImportError as exc:
            raise ImportError(
                "Reading the full published RRR artifact requires ijson; "
                "install the project requirements before running the comparison"
            ) from exc
        handle = path.open("rb")
        source = ijson.kvitems(handle, "RRR_beta", use_float=True)

    result = {}
    try:
        for key, value in source:
            key = str(key)
            if key not in wanted:
                continue
            beta = np.asarray(value, dtype=float)
            if beta.ndim != 2 or beta.shape[0] != n_variables + 1:
                raise ValueError(
                    f"Unexpected RRR_beta shape {beta.shape}; expected "
                    f"({n_variables + 1}, time), with the intercept last"
                )
            result[key] = np.abs(beta[:-1]).sum(axis=1)
    finally:
        if "handle" in locals():
            handle.close()
    missing = wanted.difference(result)
    if missing:
        raise ValueError(f"RRR_beta is missing {len(missing)} requested rows")
    return result
