"""
Step 2: Preprocess IBL data for xCEBRA training.

Mirrors the exact preprocessing from brainwide-RRR save_and_load_data.py::_read_Xy()
to produce:
    - Neural activity x: (K, T, N) → reshaped to (K*T, N) for CEBRA
    - Behavioral labels c: (K, T, 8) → reshaped to (K*T, 8) for CEBRA
    - Per-neuron metadata (area, uuid, firing rate, etc.)

The key difference from the RRR pipeline is that we format the data
for contrastive learning rather than regression.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from xcebra_ibl.configs.config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, SPSDT,
    MIN_FIRING_RATE, MAX_SILENT_PROB, MIN_NEURONS, UNIT_LABEL_MIN,
    MIN_TRIALS, REMOVE_BLOCK5, GAUSSIAN_SMOOTH_SIGMA,
    TRANSFORM_MFR, STANDARDIZE_Y, STANDARDIZE_X,
    AREAS_EXCLUDE, VARIABLE_NAMES, DISCRETE_VARIABLE_NAMES,
)


def _session_id(npz_path):
    """Return the session ID for either supported export filename format."""
    stem = Path(npz_path).stem
    if stem.startswith("data_wtonguepaw_"):
        # data_wtonguepaw_{eid}_all_spsT10_False.npz
        return "_".join(stem.split("_")[2:-3])
    if stem.startswith("data_"):
        # Current local export: data_{eid}.npz
        return stem[len("data_"):]
    return stem


# ──────────────────────────────────────────────────────
# Variable extraction functions (identical to brainwide-RRR)
# ──────────────────────────────────────────────────────

def _find_best_delay_by_cc(beh_signal, neural_data, max_delay=None):
    """
    Find optimal time delay between behavioral signal and neural activity
    using cross-correlation (mirrors brainwide-RRR utils.find_bestdelay_byCC).

    Parameters
    ----------
    beh_signal : (K, T_raw) behavioral signal
    neural_data : (K, T, N) neural activity
    max_delay : int or None, optional maximum delay to search (in bins).  The
        default ``None`` reproduces the reference RRR search range exactly.

    Returns
    -------
    best_delay : int, optimal delay in bins
    success : bool, whether a clear peak was found
    """
    K, T, N = neural_data.shape
    # This is the same statistic used by the reference RRR implementation:
    # correlate each neuron within each trial, average over trials, and select
    # the lag with the largest population-norm correlation.  Using the mean
    # neuron (as the previous implementation did) can cancel neurons with
    # opposite tuning and produces a different behavioral alignment.
    beh_signal = np.asarray(beh_signal, dtype=float)
    neural_data = np.asarray(neural_data, dtype=float)
    if beh_signal.shape[0] != K or beh_signal.shape[1] < T:
        raise ValueError(
            f"Behavior signal shape {beh_signal.shape} cannot cover neural shape "
            f"{neural_data.shape}."
        )

    beh_centered = beh_signal - np.mean(beh_signal, axis=1, keepdims=True)
    neural_centered = neural_data - np.mean(neural_data, axis=1, keepdims=True)
    lags = signal.correlation_lags(beh_signal.shape[1], T, mode="valid")
    cc_by_neuron = np.asarray([
        np.mean(
            [
                signal.correlate(
                    beh_centered[k], neural_centered[k, :, ni], mode="valid"
                )
                for k in range(K)
            ],
            axis=0,
        )
        for ni in range(N)
    ])
    cc_norm = np.linalg.norm(cc_by_neuron, axis=0)

    if max_delay is not None:
        valid = (lags >= 0) & (lags <= max_delay)
        if not np.any(valid):
            return 10, False
        candidate = np.flatnonzero(valid)
        best_idx = candidate[np.argmax(cc_norm[valid])]
    else:
        best_idx = int(np.argmax(cc_norm))

    best_delay = int(lags[best_idx])
    # Match the reference success criterion: an interior lag is considered a
    # measurable delay; edge maxima are treated as an unsuccessful search.
    success = bool(best_delay > lags.min() and best_delay < lags.max())
    return (best_delay if success else 10), success


def _extract_variable(var_name, temp_data, ks_include, neural_data, K, T, shift_beh=True):
    """
    Extract a single behavioral variable, matching the brainwide-RRR approach.

    Parameters
    ----------
    var_name : str, one of VARIABLE_NAMES
    temp_data : dict-like, loaded .npz file
    ks_include : bool array (K_total,), trial selection mask
    neural_data : (K, T, N), neural activity (after trial selection)
    K, T : int, number of included trials and time bins

    Returns
    -------
    var_3d : (K, T, 1) variable values
    delay_info : dict with delay information (for time-varying variables)
    """
    behavior = temp_data["behavior"][ks_include]
    delay_info = {}

    if var_name == "block":
        # Block prior: (probLeft - 0.5) / 0.3 → {-1, 0, 1}
        vals = np.round(np.array([(float(b.split("_")[0]) - 0.5) / 0.3 for b in behavior]))
        return np.repeat(vals, T).reshape((K, T, 1)), delay_info

    elif var_name == "side":
        # Stimulus side: 1.0 (left) or -1.0 (right)
        vals = np.round(np.array([float(b.split("_")[1]) for b in behavior]))
        return np.repeat(vals, T).reshape((K, T, 1)), delay_info

    elif var_name == "contrast_level":
        # Contrast level: 0 (zero), 1 (low ≤12.5%), 4 (high >12.5%)
        def _b2cont(b):
            c = float(b.split("_")[2])
            if c == 0:
                return 0.0
            elif c >= 0.25:
                return 4.0
            else:
                return 1.0
        vals = np.round(np.array([_b2cont(b) for b in behavior]))
        return np.repeat(vals, T).reshape((K, T, 1)), delay_info

    elif var_name == "choice":
        # Choice: 1 (CW/left) or -1 (CCW/right)
        vals = np.round(np.array([float(b.split("_")[3]) for b in behavior]))
        return np.repeat(vals, T).reshape((K, T, 1)), delay_info

    elif var_name == "outcome":
        # Outcome: 1 (correct) or -1 (error), derived from side == choice
        sides = np.array([float(b.split("_")[1]) for b in behavior])
        choices = np.array([float(b.split("_")[3]) for b in behavior])
        vals = np.round((sides == choices) * 2 - 1.0)
        return np.repeat(vals, T).reshape((K, T, 1)), delay_info

    elif var_name == "wheel":
        beh_raw = temp_data["wheel_vel"][ks_include, :-1]  # (K, T_raw, 1)
        if len(beh_raw.shape) == 2:
            beh_raw = beh_raw[:, :, np.newaxis]
        return _preprocess_movement(beh_raw, neural_data, shift_beh, delay_info, var_name)

    elif var_name == "lick":
        beh_raw = temp_data["licks"][ks_include, :-1]  # (K, T_raw, 1)
        if len(beh_raw.shape) == 2:
            beh_raw = beh_raw[:, :, np.newaxis]
        return _preprocess_movement(beh_raw, neural_data, shift_beh, delay_info, var_name)

    elif var_name == "whisker_max":
        beh_raw = temp_data["whisker_motion"][ks_include, :-1]  # (K, T_raw, 2)
        # Take max of left/right whisker motion energy
        beh_raw = np.max(beh_raw, axis=-1, keepdims=True)  # (K, T_raw, 1)
        return _preprocess_movement(beh_raw, neural_data, shift_beh, delay_info, var_name)

    else:
        raise ValueError(f"Unknown variable: {var_name}")


def _preprocess_movement(beh_raw, neural_data, shift_beh, delay_info, var_name):
    """
    Preprocess a time-varying behavioral variable:
    1. Find optimal time delay via cross-correlation
    2. Slice to match neural data length
    3. Z-score across trials and time
    """
    K, T, N = neural_data.shape
    default_delay = 10  # 10 bins = 100 ms

    if beh_raw.shape[1] < T + default_delay:
        raise ValueError(
            f"Movement signal has {beh_raw.shape[1]} bins, but {T} neural bins "
            f"plus the {default_delay}-bin fallback delay are required."
        )
    beh_processed = np.zeros((K, T, beh_raw.shape[2]))

    if shift_beh:
        for i in range(beh_raw.shape[2]):
            bd, success = _find_best_delay_by_cc(beh_raw[:, :, i], neural_data)
            delay_info[f"{var_name}_delay_{i}"] = bd
            delay_info[f"{var_name}_success_{i}"] = success
            beh_processed[:, :, i] = beh_raw[:, bd:bd + T, i]
    else:
        beh_processed = beh_raw[:, default_delay:default_delay + T, :]

    # Z-score across all trials and time
    mean_val = np.mean(beh_processed, axis=(0, 1))
    std_val = np.std(beh_processed, axis=(0, 1))
    std_val = np.clip(std_val, 1e-8, None)
    beh_processed = (beh_processed - mean_val) / std_val

    return beh_processed, delay_info


def preprocess_session(
    npz_path,
    var_list=None,
    smooth_w=None,
    min_mfr=None,
    max_sp=None,
    min_trials=None,
    min_neurons=None,
    unit_label_min=None,
    transform_mfr=None,
    standardize_y=None,
    standardize_X=None,
    remove_block5=None,
    exclude_areas=None,
    include_areas=None,
    spsdt=None,
    verbose=False,
):
    """
    Preprocess a single IBL session .npz file.

    This exactly mirrors _read_Xy() from brainwide-RRR to ensure
    identical preprocessing as the "Rarely Categorical" paper.

    Parameters
    ----------
    npz_path : str or Path
        Path to the raw .npz session file.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - 'X_3d': (K, T, 8) behavioral variables (z-scored)
        - 'y_3d': (K, T, N) neural activity (z-scored, smoothed)
        - 'X_2d': (K*T, 8) flattened for CEBRA
        - 'y_2d': (K*T, N) flattened for CEBRA
        - 'trial_ids': (K*T,) trial index for each time bin
        - 'time_ids': (K*T,) time bin index within trial
        - 'metadata': dict with neuron info, preprocessing params
        Returns None if session doesn't meet inclusion criteria.
    """
    # Default parameters from config
    if var_list is None:
        var_list = VARIABLE_NAMES
    if smooth_w is None:
        smooth_w = GAUSSIAN_SMOOTH_SIGMA
    if min_mfr is None:
        min_mfr = MIN_FIRING_RATE
    if max_sp is None:
        max_sp = MAX_SILENT_PROB
    if min_trials is None:
        min_trials = MIN_TRIALS
    if min_neurons is None:
        min_neurons = MIN_NEURONS
    if unit_label_min is None:
        unit_label_min = UNIT_LABEL_MIN
    if transform_mfr is None:
        transform_mfr = TRANSFORM_MFR
    if standardize_y is None:
        standardize_y = STANDARDIZE_Y
    if standardize_X is None:
        standardize_X = STANDARDIZE_X
    if remove_block5 is None:
        remove_block5 = REMOVE_BLOCK5
    if exclude_areas is None:
        exclude_areas = AREAS_EXCLUDE
    if spsdt is None:
        spsdt = SPSDT

    # Load raw data
    temp = np.load(str(npz_path), allow_pickle=True)

    # ── Extract spike count matrix ──
    # Trim edge bins: bins 10 to -11 (matching brainwide-RRR)
    data_allN = temp["spike_count_matrix"][:, 10:-11, :] * spsdt  # Convert to spike counts
    data_allN = np.clip(data_allN, 0, None)

    # Extract cluster/neuron metadata
    clusters_g_allN = {}
    for k in temp["clusters_g"].item():
        clusters_g_allN[k] = temp["clusters_g"].item()[k]

    K_total, T, N_total = data_allN.shape

    # ── Trial selection ──
    ks_include = np.ones(K_total, dtype=bool)
    if remove_block5:
        block = np.round(
            np.array([(float(b.split("_")[0]) - 0.5) / 0.3 for b in temp["behavior"]])
        )
        ks_include = ~(block == 0.0)

    data_allN = data_allN[ks_include]
    K = data_allN.shape[0]

    # Apply the trial filter before checking the threshold.  The old code
    # checked K_total, which could admit sessions with too few usable trials
    # after removing block-5 trials.
    if K < min_trials:
        if verbose:
            print(f"  Skipping: K={K} < min_trials={min_trials}")
        return None

    # ── Neuron selection ──
    # Silent probability < threshold
    cs = np.mean(np.all(data_allN == 0.0, axis=1), axis=0) < max_sp
    # Mean firing rate > threshold
    cs &= np.mean(data_allN, (0, 1)) / spsdt > min_mfr
    # Unit label >= threshold
    good_unit = clusters_g_allN["label"] >= unit_label_min
    cs &= good_unit

    # Area filtering
    if include_areas is not None:
        good_area_l = np.asarray(include_areas)
    else:
        good_area_l = np.unique(clusters_g_allN["acronym"])
    if exclude_areas is not None and len(exclude_areas) > 0:
        good_area_l = good_area_l[~np.isin(good_area_l, exclude_areas)]
    good_area = np.isin(clusters_g_allN["acronym"], good_area_l)
    cs &= good_area

    data = data_allN[:, :, cs]
    clusters_g = {k: clusters_g_allN[k][cs] for k in clusters_g_allN}
    K, T, N = data.shape

    if N < min_neurons:
        if verbose:
            print(f"  Skipping: N={N} < min_neurons={min_neurons}")
        return None

    if verbose:
        print(f"  Shape: K={K} trials, T={T} time bins, N={N} neurons")

    # ── Extract behavioral variables ──
    best_delays = {}
    var_arrays = []
    for var_name in var_list:
        try:
            var_3d, delay_info = _extract_variable(
                var_name, temp, ks_include, data, K, T, shift_beh=True
            )
            var_arrays.append(var_3d)
            best_delays.update(delay_info)
        except Exception as e:
            if verbose:
                print(f"  Error extracting {var_name}: {e}")
            return None

    X_3d_raw = np.concatenate(var_arrays, axis=-1)  # (K, T, 8)

    # ── Process neural activity (y) ──
    y_3d = data.copy()

    # Optional transform
    if transform_mfr == "sqrt":
        y_3d = np.sqrt(y_3d)
    elif transform_mfr == "log":
        y_3d = np.log(y_3d + 1e-3)

    # Gaussian smoothing
    if smooth_w > 0:
        y_3d = gaussian_filter1d(y_3d, smooth_w, axis=1)

    # Z-score per neuron per time bin
    if standardize_y:
        mean_y = np.mean(y_3d, axis=0)  # (T, N)
        std_y = np.std(y_3d, axis=0)    # (T, N)
        std_y = np.clip(std_y, 1e-8, None)
    else:
        mean_y = np.zeros(y_3d.shape[1:])
        std_y = np.ones(y_3d.shape[1:])
    y_3d = (y_3d - mean_y) / std_y

    # Z-score input variables per variable per time bin
    if standardize_X:
        mean_X = np.mean(X_3d_raw, axis=0)  # (T, 8)
        std_X = np.std(X_3d_raw, axis=0)    # (T, 8)
        std_X = np.clip(std_X, 1e-8, None)
    else:
        mean_X = np.zeros(X_3d_raw.shape[1:])
        std_X = np.ones(X_3d_raw.shape[1:])
    X_3d = (X_3d_raw - mean_X) / std_X

    # Keep a separate label representation.  Continuous movement variables
    # use the normalized representation, while categorical variables retain
    # their original integer-valued classes for CEBRA's discrete sampler.
    labels_3d = X_3d.copy()
    label_arrays = {}
    label_classes = {}
    for var_idx, var_name in enumerate(var_list):
        if var_name in DISCRETE_VARIABLE_NAMES:
            discrete = np.rint(X_3d_raw[:, :, var_idx]).astype(np.int64)
            labels_3d[:, :, var_idx] = discrete
            # CEBRA's discrete sampler uses np.bincount and therefore
            # requires non-negative integer class ids.  Preserve the original
            # classes in metadata while passing compact ids to the sampler.
            classes = np.unique(discrete)
            label_arrays[var_name] = np.searchsorted(
                classes, discrete.reshape(K * T)
            ).astype(np.int64)
            label_classes[var_name] = classes.tolist()
        else:
            label_arrays[var_name] = X_3d[:, :, var_idx].astype(np.float32).reshape(K * T)

    # ── Flatten for CEBRA (2D format) ──
    # Each row = one population vector at one time bin of one trial
    y_2d = y_3d.reshape(K * T, N)           # (K*T, N)
    X_2d = X_3d.reshape(K * T, len(var_list))  # (K*T, 8)

    # Trial and time indices for cross-validation
    trial_ids = np.repeat(np.arange(K), T)   # (K*T,)
    time_ids = np.tile(np.arange(T), K)      # (K*T,)

    # Extract eid from either the legacy or current export filename.
    eid = _session_id(npz_path)

    return {
        "X_3d": X_3d,                         # (K, T, 8)
        "labels_3d": labels_3d,               # unscaled categorical labels
        "y_3d": y_3d,                         # (K, T, N)
        "X_2d": X_2d,                         # (K*T, 8)
        "labels_2d": labels_3d.reshape(K * T, len(var_list)),
        "label_arrays": label_arrays,
        "label_classes": label_classes,
        "y_2d": y_2d,                         # (K*T, N)
        "trial_ids": trial_ids,               # (K*T,)
        "time_ids": time_ids,                 # (K*T,)
        "K": K,
        "T": T,
        "N": N,
        "eid": eid,
        "metadata": {
            "uuids": clusters_g.get("uuids", np.array([])),
            "acronym": clusters_g.get("acronym", np.array([])),
            "firing_rate": clusters_g.get("firing_rate", np.array([])),
            "depths": clusters_g.get("depths", np.array([])),
            "Cosmos": clusters_g.get("Cosmos", np.array([])),
            "mean_y_TN": mean_y,
            "std_y_TN": std_y,
            "mean_X_Tv": mean_X,
            "std_X_Tv": std_X,
            "best_delays": best_delays,
            "var_list": var_list,
        },
    }


def preprocess_all_sessions(
    data_dir=None,
    output_dir=None,
    verbose=True,
):
    """
    Preprocess all downloaded IBL sessions.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing raw .npz files.
    output_dir : Path, optional
        Directory to save preprocessed .pkl files.
    verbose : bool

    Returns
    -------
    dict : eid → preprocessed session data
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR
    if output_dir is None:
        output_dir = DATA_PROCESSED_DIR

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support legacy names and the current data/downloaded/data_{eid}.npz
    # export. rglob also handles an enclosing directory in the Kaggle dataset.
    npz_files = sorted(data_dir.rglob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {data_dir}")
        print("Run download_ibl.py first, or place files manually.")
        return {}

    if verbose:
        print(f"Found {len(npz_files)} session files to preprocess")

    all_sessions = {}
    for npz_path in tqdm(npz_files, desc="Preprocessing sessions"):
        # Check if already preprocessed
        eid = _session_id(npz_path)
        cache_path = output_dir / f"session_{eid}.pkl"

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                session_data = pickle.load(f)
            all_sessions[eid] = session_data
            continue

        session_data = preprocess_session(npz_path, verbose=verbose)
        if session_data is not None:
            all_sessions[eid] = session_data
            with open(cache_path, "wb") as f:
                pickle.dump(session_data, f)

    if verbose:
        total_neurons = sum(s["N"] for s in all_sessions.values())
        total_trials = sum(s["K"] for s in all_sessions.values())
        print(f"\nPreprocessed {len(all_sessions)} sessions")
        print(f"Total neurons: {total_neurons}")
        print(f"Total trials: {total_trials}")

        # Count per area
        area_counts = {}
        for s in all_sessions.values():
            for area in s["metadata"]["acronym"]:
                area_counts[area] = area_counts.get(area, 0) + 1
        print(f"Unique brain areas: {len(area_counts)}")

    return all_sessions


def load_preprocessed_sessions(output_dir=None):
    """Load previously preprocessed sessions from cache."""
    if output_dir is None:
        output_dir = DATA_PROCESSED_DIR
    output_dir = Path(output_dir)

    all_sessions = {}
    for pkl_path in sorted(output_dir.glob("session_*.pkl")):
        eid = pkl_path.stem.replace("session_", "")
        with open(pkl_path, "rb") as f:
            all_sessions[eid] = pickle.load(f)

    return all_sessions


def build_neuron_dataframe(all_sessions):
    """
    Build a pandas DataFrame with one row per neuron across all sessions.
    This mirrors the structure from brainwide-RRR step2_save_trainedRRR_2_df.py.

    Returns
    -------
    pd.DataFrame with columns: eid, ni, uuids, acronym, mfr_task
    """
    import pandas as pd

    rows = []
    for eid, session in sorted(all_sessions.items()):
        meta = session["metadata"]
        N = session["N"]
        for ni in range(N):
            rows.append({
                "eid": eid,
                "ni": ni,
                "uuids": meta["uuids"][ni] if len(meta["uuids"]) > ni else "",
                "acronym": meta["acronym"][ni] if len(meta["acronym"]) > ni else "",
                "mfr_task": float(meta["firing_rate"][ni]) if len(meta["firing_rate"]) > ni else 0.0,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("IBL Data Preprocessor for xCEBRA-IBL Pipeline")
    print("=" * 50)
    all_sessions = preprocess_all_sessions(verbose=True)
