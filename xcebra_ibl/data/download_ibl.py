"""
Step 1: Download IBL data via ONE API.

Mirrors the exact download logic from brainwide-RRR/example1/step1_IBL_downloaddata.py
but adapted for standalone use.

Each session is saved as a .npz file containing:
    - spike_count_matrix : (K, T, N) firing rates (spike counts / dt)
    - behavior           : (K,) string-encoded trial info
    - timeline           : (K, 4) event bin indices
    - clusters_g         : dict with neuron metadata
    - wheel_vel          : (K, T, 1) wheel velocity
    - whisker_motion     : (K, T, 2) left/right whisker motion energy
    - licks              : (K, T, 1) lick rate
"""

import sys
import os
import glob
import shutil
import numpy as np
from pathlib import Path

# Add brainwide-RRR repo to path so we can reuse their download functions
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "repos" / "brainwide-RRR"))

from xcebra_ibl.configs.config import (
    DATA_RAW_DIR, CORTICAL_AREAS, T_WINDOW, T_BF_STIMON,
    SPSDT, MAX_SESSIONS_PER_AREA, IBL_BASE_URL, IBL_USERNAME,
    IBL_PASSWORD, MIN_FIRING_RATE, MAX_FIRING_RATE
)


def download_ibl_sessions(
    cache_dir=None,
    data_dir=None,
    areas=None,
    max_sessions=None,
    verbose=True,
):
    """
    Download IBL sessions for each cortical area via the ONE API.

    This function uses the original brainwide-RRR download utilities
    to ensure identical preprocessing as the paper.

    Parameters
    ----------
    cache_dir : Path, optional
        Cache directory for ONE API. Default: DATA_RAW_DIR / "ibl_cache"
    data_dir : Path, optional
        Output directory for .npz files. Default: DATA_RAW_DIR
    areas : list of str, optional
        Cortical area acronyms to download. Default: CORTICAL_AREAS
    max_sessions : int, optional
        Max sessions per area. Default: MAX_SESSIONS_PER_AREA
    verbose : bool
        Print progress information.
    """
    try:
        from one.api import ONE
        from iblatlas.atlas import AllenAtlas
        from example1.utils.download_data import (
            load_data_from_pid,
            filter_trials_func,
            filter_neurons_func,
        )
    except ImportError as e:
        print(f"ERROR: Missing dependency for IBL download: {e}")
        print("Install with: pip install ONE-api iblatlas")
        print("\nAlternatively, you can manually place pre-downloaded .npz files in:")
        print(f"  {DATA_RAW_DIR}")
        return

    if cache_dir is None:
        cache_dir = DATA_RAW_DIR / "ibl_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if data_dir is None:
        data_dir = DATA_RAW_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if areas is None:
        areas = CORTICAL_AREAS
    if max_sessions is None:
        max_sessions = MAX_SESSIONS_PER_AREA

    # Connect to IBL public server (no interactive prompts)
    one = ONE(
        base_url=IBL_BASE_URL,
        username=IBL_USERNAME,
        password=IBL_PASSWORD,
        cache_dir=str(cache_dir),
        silent=True,
    )
    ba = AllenAtlas()

    spsdt = SPSDT
    only_label1 = False

    total_downloaded = 0
    for area_acronym in areas:
        pids_area = np.array(
            [p for p in one.search_insertions(atlas_acronym=area_acronym, query_type="remote")]
        )
        if verbose:
            print(f"Area {area_acronym}: {len(pids_area)} probe insertions found")

        n_valid = 0
        for pid_ in pids_area:
            if n_valid >= max_sessions:
                break

            eid, pname = one.pid2eid(pid_)
            fname = data_dir / f"data_wtonguepaw_{eid}_all_spsT{int(spsdt * 1e3)}_{only_label1}.npz"

            if fname.exists():
                n_valid += 1
                continue

            try:
                data_ret = load_data_from_pid(
                    eid, one, ba,
                    lambda t: filter_trials_func(
                        t, remove_timeextreme_event=(True, 0.8)
                    ),
                    lambda n: filter_neurons_func(
                        n,
                        remove_frextreme=(True, MIN_FIRING_RATE, MAX_FIRING_RATE),
                        only_goodneuron=only_label1,
                        only_area=False,
                    ),
                    spsdt=spsdt,
                    min_neurons=10,
                    Twindow=T_WINDOW,
                    t_bf_stimOn=T_BF_STIMON,
                    load_motion_energy=True,
                    load_wheel_velocity=True,
                    load_tongue=True,
                )
                
                # Note: we do not fabricate or save a 'paw' variable here.
                # The original brainwide-RRR pipeline did not rely on paw data,
                # so we skip extracting/saving any paw_motion proxy.

            except Exception as e:
                if verbose:
                    print(f"  Failed to load {eid}: {e}")
                continue

            if data_ret is None:
                continue

            np.savez(
                str(fname),
                spike_count_matrix=data_ret["spike_count_matrix"],
                behavior=data_ret["behavior"],
                timeline=data_ret["timeline"],
                clusters_g=data_ret["clusters_g"],
                pid=pid_,
                eid=eid,
                wheel_vel=data_ret["wheel_vel"],
                whisker_motion=data_ret["whisker_motion"],
                licks=data_ret["licks"],
            )
            n_valid += 1
            total_downloaded += 1

            # Clean cache to save disk space
            for f in glob.glob(os.path.join(str(cache_dir), "*")):
                if os.path.isdir(f):
                    shutil.rmtree(f)

        if verbose:
            print(f"  → {n_valid} valid sessions for {area_acronym}")

    if verbose:
        print(f"\nTotal sessions downloaded: {total_downloaded}")
        print(f"Data saved to: {data_dir}")


def list_available_sessions(data_dir=None):
    """List all downloaded session .npz files and return their EIDs."""
    if data_dir is None:
        data_dir = DATA_RAW_DIR
    data_dir = Path(data_dir)

    npz_files = sorted(data_dir.rglob("*.npz"))
    eids = []
    for f in npz_files:
        stem = f.stem
        if stem.startswith("data_wtonguepaw_"):
            eid = "_".join(stem.split("_")[2:-3])
        elif stem.startswith("data_"):
            eid = stem[len("data_"):]
        else:
            eid = stem
        eids.append(eid)
    return eids, npz_files


if __name__ == "__main__":
    print("IBL Data Downloader for xCEBRA-IBL Pipeline")
    print("=" * 50)
    print(f"Output directory: {DATA_RAW_DIR}")
    print(f"Areas to download: {len(CORTICAL_AREAS)} cortical regions")
    print()

    existing_eids, _ = list_available_sessions()
    if existing_eids:
        print(f"Found {len(existing_eids)} existing sessions.")
        resp = input("Continue downloading? [y/N] ")
        if resp.lower() != "y":
            sys.exit(0)

    download_ibl_sessions(verbose=True)
