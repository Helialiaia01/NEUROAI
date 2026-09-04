"""
Step 4: Selectivity analysis and clustering with xCEBRA attribution maps.

This module mirrors step3_analyze_selectivity.py from brainwide-RRR but uses
xCEBRA Jacobian-based attributions instead of RRR β coefficients.

The analysis:
1. Compute per-neuron selectivity: α_{n,v} = A_{v,n} (Jacobian attribution for
   variable v at neuron n) — replaces Σ_t |β_{n,v,t}| from RRR
2. Compute per-area selectivity profile: mean α across neurons in each area
3. Z-score across areas
4. Correlate selectivity similarity with anatomical connectivity
5. Cluster areas by their 8D selectivity fingerprint
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from xcebra_ibl.configs.config import (
    VARIABLE_NAMES, VARIABLE_DISPLAY_NAMES, N_VARIABLES,
    MIN_NEURONS_PER_AREA, MIN_NEURONS_PER_AREA_CORR,
    MIN_DELTA_R2,
    RESULTS_DIR, CORTICAL_AREAS, DATA_PROCESSED_DIR,
    ALLEN_AREA_LIST_CSV, ALLEN_CONN_MATRIX_CSV,
    RRR_RESULTS_DEFAULT,
)


def _load_allen_anatomy_from_csv():
    """Load Allen cortical area order and connectivity from local CSV files."""
    area_list_path = Path(ALLEN_AREA_LIST_CSV)
    conn_mat_path = Path(ALLEN_CONN_MATRIX_CSV)

    if not area_list_path.exists() or not conn_mat_path.exists():
        missing = [
            str(p) for p in [area_list_path, conn_mat_path] if not p.exists()
        ]
        raise FileNotFoundError(f"Missing Allen anatomy files: {missing}")

    conn_area_list = pd.read_csv(area_list_path, header=None).values[:, 0]
    conn_i2a = {i: a for i, a in enumerate(conn_area_list)}
    conn_a2i = {a: i for i, a in enumerate(conn_area_list)}
    conn_mat = pd.read_csv(conn_mat_path, header=None).values
    if conn_mat.shape != (len(conn_area_list), len(conn_area_list)):
        raise ValueError(
            "Allen connectivity matrix shape does not match area list: "
            f"{conn_mat.shape} vs {(len(conn_area_list), len(conn_area_list))}"
        )

    mod2area = {
        "prefrontal": ["FRP", "ACAd", "ACAv", "PL", "ILA", "ORBl", "ORBm", "ORBvl"],
        "lateral": ["AId", "AIv", "AIp", "GU", "VISC", "TEa", "PERI", "ECT"],
        "somatomotor": ["SSs", "SSp-bfd", "SSp-tr", "SSp-ll", "SSp-ul", "SSp-un", "SSp-n", "SSp-m", "MOp", "MOs"],
        "visual": ["VISal", "VISl", "VISp", "VISpl", "VISli", "VISpor", "VISrl"],
        "medial": ["VISa", "VISam", "VISpm", "RSPagl", "RSPd", "RSPv"],
        "auditory": ["AUDd", "AUDp", "AUDpo", "AUDv"],
    }
    area2mod = {area: mod for mod, areas in mod2area.items() for area in areas}

    return {
        "cortical_area_list": conn_area_list,
        "conn_mat": conn_mat,
        "area2H": conn_a2i,
        "H2area": conn_i2a,
        "mod2area": mod2area,
        "area2mod": area2mod,
    }


def _compute_local_linear_baseline(min_neurons=50):
    """
    Compute a local linear selectivity baseline from preprocessed sessions.

    This serves as a practical fallback when the original RRRglobal_full.json
    artifact is not available (e.g., Git-LFS pointer only).
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from xcebra_ibl.data.preprocess import load_preprocessed_sessions

    sessions = load_preprocessed_sessions(DATA_PROCESSED_DIR)
    if not sessions:
        return None

    area_to_sel = {}

    for eid, session in sorted(sessions.items()):
        X = np.asarray(session["X_2d"], dtype=np.float64)  # (samples, 8)
        Y = np.asarray(session["y_2d"], dtype=np.float64)  # (samples, N)
        acronyms = np.asarray(session["metadata"]["acronym"])

        if X.shape[0] < 100 or Y.shape[1] == 0:
            continue

        # Basic finite sanitization + robust clipping for numerical stability.
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -1e3, 1e3)
        Y = np.clip(Y, -1e3, 1e3)
        x_clip = np.nanpercentile(np.abs(X), 99.5)
        y_clip = np.nanpercentile(np.abs(Y), 99.5)
        if np.isfinite(x_clip) and x_clip > 0:
            X = np.clip(X, -x_clip, x_clip)
        if np.isfinite(y_clip) and y_clip > 0:
            Y = np.clip(Y, -y_clip, y_clip)

        x_scaler = StandardScaler(with_mean=True, with_std=True)
        y_scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = x_scaler.fit_transform(X)
        Ys = y_scaler.fit_transform(Y)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        Ys = np.nan_to_num(Ys, nan=0.0, posinf=0.0, neginf=0.0)

        # Multi-target ridge: fit all neurons at once.
        model = Ridge(alpha=10.0, solver="svd")
        model.fit(Xs, Ys)
        coef = np.abs(model.coef_)  # (N, 8)

        n_neurons = min(coef.shape[0], len(acronyms))
        for ni in range(n_neurons):
            area = acronyms[ni]
            if area not in area_to_sel:
                area_to_sel[area] = []
            area_to_sel[area].append(coef[ni])

    # Aggregate per area
    area_list = []
    sel_list = []
    for area in sorted(area_to_sel.keys()):
        arr = np.asarray(area_to_sel[area])
        if arr.shape[0] < min_neurons:
            continue
        area_list.append(area)
        sel_list.append(arr.mean(axis=0))

    if not area_list:
        return None

    sel = np.asarray(sel_list)
    sel_z = (sel - sel.mean(0)) / (sel.std(0) + 1e-8)

    return {
        "areas": area_list,
        "sel_raw": sel,
        "sel_z": sel_z,
        "source": "local_linear_fallback",
    }


def compute_selectivity_from_attributions(
    neuron_df: pd.DataFrame,
    min_neurons_per_area: int = None,
    area_list: list = None,
) -> dict:
    """
    Compute per-area selectivity profiles from per-neuron xCEBRA attributions.

    This is the xCEBRA analog of the RRR selectivity analysis:
    - RRR:    α_{n,v} = Σ_t |β_{n,v,t}|
    - xCEBRA: α_{n,v} = E_x[‖∂f_v/∂x_n‖²]  (Jacobian attribution)

    Parameters
    ----------
    neuron_df : pd.DataFrame
        Must contain columns: 'acronym' and 'xcebra_attr_{var_name}' for each variable.
    min_neurons_per_area : int
        Minimum neurons required per area. Default: MIN_NEURONS_PER_AREA
    area_list : list, optional
        Restrict to these areas. Default: all areas in data.

    Returns
    -------
    dict with keys:
        - 'sel_areas': (n_areas, 8) z-scored selectivity per area
        - 'raw_sel_areas': (n_areas, 8) raw mean selectivity per area
        - 'area_order': list of area names
        - 'sel_per_neuron': (n_neurons, 8) per-neuron selectivity
        - 'n_neurons_per_area': dict {area: count}
    """
    if min_neurons_per_area is None:
        min_neurons_per_area = MIN_NEURONS_PER_AREA

    if neuron_df.empty:
        raise ValueError("Cannot compute selectivity profiles from an empty DataFrame")

    # Extract per-neuron selectivity matrix
    attr_cols = [f"xcebra_attr_{v}" for v in VARIABLE_NAMES]
    missing = [c for c in attr_cols if c not in neuron_df.columns]
    if missing or "acronym" not in neuron_df.columns:
        raise ValueError(f"Missing selectivity columns: {missing or ['acronym']}")
    sel_per_neuron = neuron_df[attr_cols].values  # (n_neurons, 8)

    # Get areas
    acronyms = neuron_df["acronym"].values
    unique_areas = np.unique(acronyms)

    if area_list is not None:
        unique_areas = np.array([a for a in area_list if a in unique_areas])

    # Filter by minimum neuron count
    area_order = []
    raw_sel_areas = []
    n_neurons_per_area = {}

    for area in unique_areas:
        mask = acronyms == area
        n_neurons = mask.sum()
        if n_neurons >= min_neurons_per_area:
            area_order.append(area)
            raw_sel_areas.append(sel_per_neuron[mask].mean(axis=0))
            n_neurons_per_area[area] = n_neurons

    raw_sel_areas = np.array(raw_sel_areas)  # (n_areas, 8)
    if raw_sel_areas.ndim != 2 or raw_sel_areas.shape[0] < 2:
        raise ValueError(
            f"Need at least two sufficiently sampled areas; got {len(area_order)}"
        )

    # Z-score per variable across areas
    sel_areas = (raw_sel_areas - raw_sel_areas.mean(axis=0)) / (
        raw_sel_areas.std(axis=0) + 1e-8
    )

    return {
        "sel_areas": sel_areas,
        "raw_sel_areas": raw_sel_areas,
        "area_order": area_order,
        "sel_per_neuron": sel_per_neuron,
        "n_neurons_per_area": n_neurons_per_area,
    }


def compute_selectivity_similarity(sel_areas, area_order):
    """
    Compute pairwise cosine similarity between area selectivity profiles.

    Parameters
    ----------
    sel_areas : (n_areas, 8) z-scored selectivity profiles
    area_order : list of area names

    Returns
    -------
    sim_matrix : (n_areas, n_areas) cosine similarity matrix
    """
    sim_matrix = cosine_similarity(sel_areas)
    return sim_matrix


def correlate_with_anatomy(
    sel_areas,
    area_order,
    anatomical_connectivity=None,
    area2hierarchy=None,
    min_pairs=10,
):
    """
    Correlate selectivity similarity with anatomical connectivity.

    This mirrors the analysis in step3_analyze_selectivity.py
    (Fig. 3d in the paper).

    Parameters
    ----------
    sel_areas : (n_areas, 8) z-scored selectivity profiles
    area_order : list of area names
    anatomical_connectivity : (n_areas_atlas, n_areas_atlas) connectivity matrix
        If None, will attempt to load from brainwide-RRR utils.
    area2hierarchy : dict {area: hierarchy_index}
        If None, will attempt to load.

    Returns
    -------
    dict with keys:
        - 'spearman_r': Spearman correlation coefficient
        - 'spearman_p': p-value
        - 'conn_values': list of connectivity values
        - 'sim_values': list of similarity values
        - 'pairs': list of (area_i, area_j) tuples
    """
    n_areas = len(area_order)

    conn_values = []
    sim_values = []
    pairs = []

    for i in range(n_areas):
        for j in range(i + 1, n_areas):
            # Cosine similarity between selectivity profiles
            a = sel_areas[i]
            b = sel_areas[j]
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            sim_values.append(sim)
            pairs.append((area_order[i], area_order[j]))

            # Anatomical connectivity
            if anatomical_connectivity is not None and area2hierarchy is not None:
                hi = area2hierarchy.get(area_order[i])
                hj = area2hierarchy.get(area_order[j])
                if hi is not None and hj is not None:
                    conn = (
                        anatomical_connectivity[hi, hj]
                        + anatomical_connectivity[hj, hi]
                    ) / 2
                    conn_values.append(conn)
                else:
                    conn_values.append(np.nan)
            else:
                conn_values.append(np.nan)

    # Compute Spearman correlation
    valid = ~np.isnan(conn_values)
    if valid.sum() >= min_pairs:
        rs, ps = spearmanr(
            np.array(conn_values)[valid], np.array(sim_values)[valid]
        )
    else:
        rs, ps = np.nan, np.nan

    return {
        "spearman_r": rs,
        "spearman_p": ps,
        "conn_values": conn_values,
        "sim_values": sim_values,
        "pairs": pairs,
        "n_valid_pairs": int(valid.sum()),
    }


def cluster_areas(sel_areas, area_order, n_clusters=6, method="ward"):
    """
    Hierarchical clustering of brain areas by their selectivity profiles.

    Parameters
    ----------
    sel_areas : (n_areas, 8) z-scored selectivity profiles
    area_order : list of area names
    n_clusters : int, number of clusters
    method : str, linkage method

    Returns
    -------
    dict with keys:
        - 'cluster_labels': (n_areas,) cluster assignment
        - 'linkage_matrix': scipy linkage matrix
        - 'area_clusters': dict {cluster_id: [area_names]}
        - 'cluster_profiles': dict {cluster_id: (8,) mean profile}
    """
    sel_areas = np.asarray(sel_areas, dtype=float)
    if sel_areas.ndim != 2 or sel_areas.shape[0] != len(area_order):
        raise ValueError("sel_areas and area_order have incompatible shapes")
    if len(area_order) < 2:
        raise ValueError("At least two areas are required for clustering")
    if not np.isfinite(sel_areas).all():
        raise ValueError("Selectivity profiles contain non-finite values")
    n_clusters = min(int(n_clusters), len(area_order))

    # Compute linkage
    dist = pdist(sel_areas, metric="cosine")
    Z = linkage(dist, method=method)

    # Cut tree
    cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")

    # Organize results
    area_clusters = {}
    cluster_profiles = {}
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        areas_in_cluster = [area_order[i] for i in range(len(area_order)) if mask[i]]
        area_clusters[int(c)] = areas_in_cluster
        cluster_profiles[int(c)] = sel_areas[mask].mean(axis=0)

    return {
        "cluster_labels": cluster_labels,
        "linkage_matrix": Z,
        "area_clusters": area_clusters,
        "cluster_profiles": cluster_profiles,
    }


def compare_with_rrr(
    xcebra_sel_areas,
    xcebra_area_order,
    rrr_df_path=None,
    min_neurons=50,
    allow_local_fallback=True,
):
    """
    Compare xCEBRA selectivity profiles with the original RRR results.

    Parameters
    ----------
    xcebra_sel_areas : (n_areas, 8)
    xcebra_area_order : list
    rrr_df_path : str, path to RRRglobal_full.json (optional)
    min_neurons : int

    Returns
    -------
    dict with comparison metrics
    """
    if rrr_df_path is None:
        rrr_df_path = RRR_RESULTS_DEFAULT

    rrr_df_path = Path(rrr_df_path)

    if not rrr_df_path.exists():
        # Fallback: search under brainwide-RRR for a matching result file
        fallback_candidates = list(
            rrr_df_path.parent.parent.parent.rglob("RRRglobal_full.json")
        )
        if fallback_candidates:
            rrr_df_path = fallback_candidates[0]

    if not rrr_df_path.exists():
        print(f"RRR results not found at {rrr_df_path}")
        if allow_local_fallback:
            print("Using local linear baseline fallback for comparison.")
            baseline = _compute_local_linear_baseline(min_neurons=min_neurons)
            if baseline is None:
                print("Local linear baseline unavailable. Comparison skipped.")
                return None
            return _compare_sel_matrices(
                xcebra_sel_areas,
                xcebra_area_order,
                baseline["sel_z"],
                baseline["areas"],
                baseline_label="local_linear",
            )
        print("Comparison with RRR will be skipped.")
        return None

    # Detect Git LFS pointer placeholders and skip cleanly.
    try:
        with open(rrr_df_path, "r", encoding="utf-8") as f:
            head = "".join([f.readline() for _ in range(3)])
        if "git-lfs.github.com/spec/v1" in head:
            print(f"RRR results at {rrr_df_path} are Git LFS pointers (content not downloaded).")
            if allow_local_fallback:
                print("Using local linear baseline fallback for comparison.")
                baseline = _compute_local_linear_baseline(min_neurons=min_neurons)
                if baseline is None:
                    print("Local linear baseline unavailable. Comparison skipped.")
                    return None
                return _compare_sel_matrices(
                    xcebra_sel_areas,
                    xcebra_area_order,
                    baseline["sel_z"],
                    baseline["areas"],
                    baseline_label="local_linear",
                )
            print("Comparison with RRR will be skipped.")
            return None
    except Exception:
        pass

    try:
        rrr_df = pd.read_json(str(rrr_df_path))
    except Exception as e:
        print(f"Could not parse RRR results at {rrr_df_path}: {e}")
        if allow_local_fallback:
            print("Using local linear baseline fallback for comparison.")
            baseline = _compute_local_linear_baseline(min_neurons=min_neurons)
            if baseline is None:
                print("Local linear baseline unavailable. Comparison skipped.")
                return None
            return _compare_sel_matrices(
                xcebra_sel_areas,
                xcebra_area_order,
                baseline["sel_z"],
                baseline["areas"],
                baseline_label="local_linear",
            )
        print("Comparison with RRR will be skipped.")
        return None

    # Compute RRR selectivity (same as step3_analyze_selectivity.py)
    required_rrr = {"RRRglobal_r2", "meanact_r2", "RRRglobal_beta", "acronym"}
    missing_rrr = required_rrr.difference(rrr_df.columns)
    if missing_rrr:
        raise ValueError(f"RRR artifact is missing columns: {sorted(missing_rrr)}")
    rrr_df["RRRglobal_deltaR2"] = rrr_df["RRRglobal_r2"] - rrr_df["meanact_r2"]
    nis_incmask = (
        (rrr_df["RRRglobal_deltaR2"] > MIN_DELTA_R2)
        & rrr_df["acronym"].isin(CORTICAL_AREAS)
    )

    beta_values = rrr_df.loc[nis_incmask, "RRRglobal_beta"].tolist()
    if not beta_values:
        return {"message": "No RRR neurons passed the delta-R2 threshold"}
    coef_vs = np.asarray([np.asarray(b)[:-1] for b in beta_values], dtype=float)
    if coef_vs.ndim != 3 or coef_vs.shape[1] != N_VARIABLES:
        raise ValueError(
            "Unexpected RRRglobal_beta shape after removing the intercept: "
            f"{coef_vs.shape}; expected (*, {N_VARIABLES}, time)"
        )
    rrr_sel_per_neuron = np.abs(coef_vs).sum(2)  # (n_neurons, 8)

    # Build per-area selectivity for RRR (same filtering)
    rrr_acronyms = rrr_df.loc[nis_incmask, "acronym"].values
    rrr_areas = np.unique(rrr_acronyms)

    # Build z-scored RRR area selectivity first
    rrr_area_list = []
    rrr_area_sel_list = []
    for area in sorted(rrr_areas):
        rrr_mask = rrr_acronyms == area
        if rrr_mask.sum() < min_neurons:
            continue
        rrr_area_list.append(area)
        rrr_area_sel_list.append(rrr_sel_per_neuron[rrr_mask].mean(0))

    if not rrr_area_list:
        return {"message": "No RRR areas with sufficient neurons"}

    rrr_area_sel = np.asarray(rrr_area_sel_list)
    rrr_sel_z_full = (rrr_area_sel - rrr_area_sel.mean(0)) / (rrr_area_sel.std(0) + 1e-8)

    return _compare_sel_matrices(
        xcebra_sel_areas,
        xcebra_area_order,
        rrr_sel_z_full,
        rrr_area_list,
        baseline_label="rrr",
    )


def _compare_sel_matrices(
    xcebra_sel_areas,
    xcebra_area_order,
    baseline_sel_z,
    baseline_area_order,
    baseline_label="rrr",
):
    """Shared comparison logic between xCEBRA and a baseline selectivity matrix."""
    common_areas = set(xcebra_area_order) & set(baseline_area_order)

    baseline_sel_list = []
    xcebra_sel_list = []
    common_area_list = []

    for area in sorted(common_areas):
        try:
            xcebra_idx = xcebra_area_order.index(area)
        except ValueError:
            continue
        try:
            baseline_idx = baseline_area_order.index(area)
        except ValueError:
            continue

        xcebra_area_sel = xcebra_sel_areas[xcebra_idx]
        baseline_area_sel = baseline_sel_z[baseline_idx]

        baseline_sel_list.append(baseline_area_sel)
        xcebra_sel_list.append(xcebra_area_sel)
        common_area_list.append(area)

    if not common_area_list:
        return {"message": "No common areas with sufficient neurons"}

    baseline_sel = np.array(baseline_sel_list)
    xcebra_sel = np.array(xcebra_sel_list)

    # Z-score xCEBRA to match baseline scale convention
    xcebra_sel_z = (xcebra_sel - xcebra_sel.mean(0)) / (xcebra_sel.std(0) + 1e-8)
    baseline_sel_z = (baseline_sel - baseline_sel.mean(0)) / (baseline_sel.std(0) + 1e-8)

    # Compare: per-area cosine similarity between baseline and xCEBRA profiles
    per_area_sim = []
    for i in range(len(common_area_list)):
        sim = np.dot(baseline_sel_z[i], xcebra_sel_z[i]) / (
            np.linalg.norm(baseline_sel_z[i]) * np.linalg.norm(xcebra_sel_z[i]) + 1e-8
        )
        per_area_sim.append(sim)

    # Correlation of flattened selectivity matrices
    rs, ps = spearmanr(baseline_sel_z.ravel(), xcebra_sel_z.ravel())

    # Per-variable correlation
    per_var_corr = {}
    for v in range(N_VARIABLES):
        rv, pv = spearmanr(baseline_sel_z[:, v], xcebra_sel_z[:, v])
        per_var_corr[VARIABLE_NAMES[v]] = {"spearman_r": rv, "p_value": pv}

    return {
        "baseline_label": baseline_label,
        "common_areas": common_area_list,
        "per_area_similarity": per_area_sim,
        "mean_area_similarity": np.mean(per_area_sim),
        "overall_spearman_r": rs,
        "overall_spearman_p": ps,
        "per_variable_correlation": per_var_corr,
        "rrr_sel_z": baseline_sel_z,
        "xcebra_sel_z": xcebra_sel_z,
    }


def run_full_selectivity_analysis(neuron_df, save_dir=None, comparison_mode="fallback"):
    """
    Run the complete selectivity analysis pipeline.

    Parameters
    ----------
    neuron_df : pd.DataFrame with xcebra attribution columns
    save_dir : Path
    comparison_mode : str
        "fallback" (default): use local linear baseline when RRR artifact unavailable.
        "strict": require true RRR artifact; otherwise raise an error.

    Returns
    -------
    results : dict with all analysis outputs
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("xCEBRA Selectivity Analysis")
    print("=" * 60)

    # 1. Compute per-area selectivity
    print("\n[1] Computing per-area selectivity profiles...")
    sel_result = compute_selectivity_from_attributions(
        neuron_df,
        min_neurons_per_area=MIN_NEURONS_PER_AREA,
        area_list=CORTICAL_AREAS,
    )
    print(f"    Areas included: {len(sel_result['area_order'])}")
    print(f"    Neuron counts: {list(sel_result['n_neurons_per_area'].values())[:10]}...")

    # 2. Compute pairwise similarity
    print("\n[2] Computing selectivity similarity matrix...")
    sim_matrix = compute_selectivity_similarity(
        sel_result["sel_areas"], sel_result["area_order"]
    )

    # 3. Attempt anatomical connectivity correlation
    print("\n[3] Correlating with anatomical connectivity...")
    try:
        atm = _load_allen_anatomy_from_csv()
        # Match the reference analysis, which uses log connectivity.  The
        # matrix has zero diagonal; off-diagonal entries are positive.
        conn_mat = np.log(np.clip(atm["conn_mat"], np.finfo(float).tiny, None))
        area2H = atm["area2H"]
        area2mod = atm["area2mod"]

        corr_sel_result = compute_selectivity_from_attributions(
            neuron_df,
            min_neurons_per_area=MIN_NEURONS_PER_AREA_CORR,
            area_list=CORTICAL_AREAS,
        )

        corr_result = correlate_with_anatomy(
            corr_sel_result["sel_areas"],
            corr_sel_result["area_order"],
            conn_mat,
            area2H,
        )
        print(f"    Spearman ρ = {corr_result['spearman_r']:.3f}")
        print(f"    p-value = {corr_result['spearman_p']:.4f}")
    except Exception as e:
        print(f"    Could not load anatomical info: {e}")
        corr_result = None
        area2mod = None

    # 4. Cluster areas
    print("\n[4] Clustering brain areas...")
    cluster_result = cluster_areas(
        sel_result["sel_areas"], sel_result["area_order"], n_clusters=6
    )
    for c_id, areas in cluster_result["area_clusters"].items():
        print(f"    Cluster {c_id}: {areas}")

    # 5. Compare with RRR (if available)
    print("\n[5] Comparing with RRR selectivity...")
    allow_local_fallback = comparison_mode != "strict"
    comparison = compare_with_rrr(
        corr_sel_result["sel_areas"] if "corr_sel_result" in locals() else sel_result["sel_areas"],
        corr_sel_result["area_order"] if "corr_sel_result" in locals() else sel_result["area_order"],
        allow_local_fallback=allow_local_fallback,
    )

    if comparison_mode == "strict":
        if (comparison is None) or ("overall_spearman_r" not in comparison) or (
            comparison.get("baseline_label") != "rrr"
        ):
            raise RuntimeError(
                "Strict comparison mode requires a real RRR artifact, but it was not available. "
                "Provide non-pointer RRRglobal_full.json or use comparison_mode='fallback'."
            )

    if comparison and "overall_spearman_r" in comparison:
        baseline_label = comparison.get("baseline_label", "rrr")
        baseline_name = "RRR" if baseline_label == "rrr" else "Linear baseline"
        print(f"    {baseline_name} vs xCEBRA Spearman ρ = {comparison['overall_spearman_r']:.3f}")
        print(f"    Mean per-area similarity = {comparison['mean_area_similarity']:.3f}")

    # Save all results
    results = {
        "selectivity": sel_result,
        "similarity_matrix": sim_matrix,
        "anatomical_correlation": corr_result,
        "clustering": cluster_result,
        "rrr_comparison": comparison,
    }

    # Save selectivity profiles
    sel_df = pd.DataFrame(
        sel_result["sel_areas"],
        index=sel_result["area_order"],
        columns=VARIABLE_DISPLAY_NAMES,
    )
    sel_df.to_csv(save_dir / "xcebra_selectivity_profiles.csv")

    # Save clustering
    cluster_df = pd.DataFrame({
        "area": sel_result["area_order"],
        "cluster": cluster_result["cluster_labels"],
    })
    cluster_df.to_csv(save_dir / "xcebra_area_clusters.csv", index=False)

    print(f"\n  Results saved to {save_dir}")
    return results
