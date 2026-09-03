"""
Step 3: Train xCEBRA models on IBL sessions.

For each session, trains per-variable xCEBRA models and extracts
per-neuron attribution maps (Jacobian-based selectivity).

This replaces the RRR encoding model (step2_train_RRR.py from brainwide-RRR).
"""

import numpy as np
import pickle
import json
from pathlib import Path
from tqdm import tqdm

from xcebra_ibl.configs.config import (
    DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    VARIABLE_NAMES, N_VARIABLES, N_CV_FOLDS, TEST_FRACTION,
    MAX_ITERATIONS, BATCH_SIZE, EMBEDDING_DIM_PER_GROUP,
)
from xcebra_ibl.data.preprocess import load_preprocessed_sessions, build_neuron_dataframe
from xcebra_ibl.models.xcebra_model import XCEBRAModel


def _log_session_to_wandb(wandb_run, eid, result, interval=50):
    """Log session-level training metrics to an active W&B run."""
    losses = result.get("losses", {})
    n = int(result.get("N", 0))
    k = int(result.get("K", 0))
    t = int(result.get("T", 0))

    wandb_run.log(
        {
            "session/neuron_count": n,
            "session/trial_count": k,
            "session/time_bins": t,
            "session/sample_count": k * t,
        }
    )

    for var_name, loss_curve in losses.items():
        curve = np.asarray(loss_curve, dtype=float).reshape(-1)
        if curve.size == 0:
            continue

        step_idx = list(range(0, curve.size, max(1, interval)))
        if step_idx[-1] != curve.size - 1:
            step_idx.append(curve.size - 1)

        for it in step_idx:
            wandb_run.log(
                {
                    f"train/loss/{eid}/{var_name}": float(curve[it]),
                    "train/iteration": int(it),
                }
            )

        wandb_run.summary[f"train/{eid}/{var_name}/final_loss"] = float(curve[-1])
        wandb_run.summary[f"train/{eid}/{var_name}/min_loss"] = float(np.min(curve))

    attribution_maps = result.get("attribution_maps", {})
    for var_name, attr in attribution_maps.items():
        arr = np.asarray(attr, dtype=float).reshape(-1)
        if arr.size == 0:
            continue
        wandb_run.summary[f"attr/{eid}/{var_name}/mean"] = float(np.mean(arr))
        wandb_run.summary[f"attr/{eid}/{var_name}/max"] = float(np.max(arr))


def prepare_labels_from_session(session_data):
    """
    Convert session data into a dict of label arrays for CEBRA training.

    For the per-variable approach, each variable is a 1D array of length K*T.
    Discrete variables: integer-valued
    Continuous variables: float-valued

    Parameters
    ----------
    session_data : dict from preprocess_session()

    Returns
    -------
    labels : dict {var_name: (K*T,) array}
    """
    X_2d = session_data["X_2d"]  # (K*T, 8) z-scored behavioral variables
    labels = {}
    for v, var_name in enumerate(VARIABLE_NAMES):
        labels[var_name] = X_2d[:, v]
    return labels


def train_session_xcebra(
    session_data,
    eid,
    method="per_variable",
    max_iterations=None,
    batch_size=None,
    n_attribution_samples=2000,
    model_kwargs=None,
    wandb_run=None,
    wandb_log_interval=50,
    save_models=True,
    checkpoint_frequency=1,
    verbose=True,
):
    """
    Train xCEBRA model(s) for a single IBL session and extract attribution maps.

    Parameters
    ----------
    session_data : dict from preprocess_session()
    eid : str, session ID
    method : "per_variable" or "joint"
    max_iterations : int, override config
    batch_size : int, override config
    n_attribution_samples : int, samples for Jacobian computation
    save_models : bool
    verbose : bool

    Returns
    -------
    result : dict with keys:
        - 'eid': session ID
        - 'attribution_maps': {var_name: (N,) scores per neuron}
        - 'embeddings': {var_name: (n_samples, dim) embeddings}
        - 'model': the XCEBRAModel instance
        - 'losses': training losses per variable
    """
    if max_iterations is None:
        max_iterations = MAX_ITERATIONS
    if batch_size is None:
        batch_size = BATCH_SIZE
    if model_kwargs is None:
        model_kwargs = {}

    neural_data = session_data["y_2d"]  # (K*T, N)
    labels = prepare_labels_from_session(session_data)
    N = session_data["N"]
    K = session_data["K"]
    T = session_data["T"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Session {eid}: K={K} trials, T={T} time bins, N={N} neurons")
        print(f"Total samples: {K*T}")
        print(f"Training method: {method}")
        print(f"{'='*60}")

    # Check minimum data requirements
    if K * T < batch_size:
        if verbose:
            print(f"  Not enough data (K*T={K*T} < batch_size={batch_size}). Skipping.")
        return None

    # Create and train model
    model = XCEBRAModel(
        max_iterations=max_iterations,
        batch_size=min(batch_size, K * T),
        checkpoint_dir=MODELS_DIR / "checkpoints" / eid,
        checkpoint_frequency=checkpoint_frequency,
        **model_kwargs,
    )

    if method == "per_variable":
        model.fit_per_variable(neural_data, labels, verbose=verbose)
    elif method == "joint":
        model.fit_joint(neural_data, labels, verbose=verbose)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract embeddings
    if verbose:
        print("\n  Extracting embeddings...")
    if method == "per_variable":
        embeddings = model.transform_per_variable(neural_data)
    else:
        embeddings = model.transform_joint(neural_data)

    # Compute attribution maps (Jacobian-based)
    if verbose:
        print("\n  Computing attribution maps (Jacobian)...")
    attribution_maps = model.compute_attribution_maps(
        neural_data,
        method=method,
        n_samples=min(n_attribution_samples, K * T),
    )

    # Save models
    if save_models:
        session_model_dir = MODELS_DIR / eid
        session_model_dir.mkdir(parents=True, exist_ok=True)
        model.save(save_dir=session_model_dir)
        if verbose:
            print(f"\n  Models saved to {session_model_dir}")

    result = {
        "eid": eid,
        "attribution_maps": attribution_maps,
        "embeddings": embeddings,
        "model": model,
        "losses": model.training_losses_,
        "N": N, "K": K, "T": T,
    }

    if wandb_run is not None:
        _log_session_to_wandb(
            wandb_run,
            eid=eid,
            result=result,
            interval=wandb_log_interval,
        )

    return result


def cross_validate_session(
    session_data,
    eid,
    n_folds=None,
    test_fraction=None,
    method="per_variable",
    verbose=True,
):
    """
    Cross-validate xCEBRA model on a single session.

    Evaluates the quality of embeddings using a decoding metric:
    train on embedding, predict behavioral variable.

    Parameters
    ----------
    session_data : dict
    eid : str
    n_folds : int
    test_fraction : float
    method : str

    Returns
    -------
    cv_results : dict {var_name: {'r2_scores': [...], 'mean_r2': float}}
    """
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    if n_folds is None:
        n_folds = N_CV_FOLDS
    if test_fraction is None:
        test_fraction = TEST_FRACTION

    neural_data = session_data["y_2d"]
    labels = prepare_labels_from_session(session_data)
    K = session_data["K"]
    T = session_data["T"]

    # Use trial-based cross-validation (same trial never in both train/test)
    trial_ids = session_data["trial_ids"]
    unique_trials = np.unique(trial_ids)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_results = {var_name: {"r2_scores": []} for var_name in VARIABLE_NAMES}

    for fold_idx, (train_trial_idx, test_trial_idx) in enumerate(kf.split(unique_trials)):
        if verbose:
            print(f"\n  Fold {fold_idx + 1}/{n_folds}")

        train_trials = unique_trials[train_trial_idx]
        test_trials = unique_trials[test_trial_idx]

        train_mask = np.isin(trial_ids, train_trials)
        test_mask = np.isin(trial_ids, test_trials)

        train_neural = neural_data[train_mask]
        test_neural = neural_data[test_mask]
        train_labels = {v: labels[v][train_mask] for v in VARIABLE_NAMES}
        test_labels = {v: labels[v][test_mask] for v in VARIABLE_NAMES}

        # Train CEBRA on training data
        model = XCEBRAModel(
            max_iterations=MAX_ITERATIONS // 2,  # Fewer iters for CV
            batch_size=min(BATCH_SIZE, train_neural.shape[0]),
        )

        if method == "per_variable":
            model.fit_per_variable(train_neural, train_labels, verbose=False)
            train_emb = model.transform_per_variable(train_neural)
            test_emb = model.transform_per_variable(test_neural)
        else:
            model.fit_joint(train_neural, train_labels, verbose=False)
            train_emb = model.transform_joint(train_neural)
            test_emb = model.transform_joint(test_neural)

        # Decode each variable from embeddings using Ridge regression
        for var_name in VARIABLE_NAMES:
            if var_name not in train_emb:
                continue

            decoder = Ridge(alpha=1.0)
            decoder.fit(train_emb[var_name], train_labels[var_name])
            pred = decoder.predict(test_emb[var_name])
            r2 = r2_score(test_labels[var_name], pred)
            cv_results[var_name]["r2_scores"].append(r2)

    # Compute means
    for var_name in VARIABLE_NAMES:
        scores = cv_results[var_name]["r2_scores"]
        cv_results[var_name]["mean_r2"] = np.mean(scores) if scores else 0.0
        cv_results[var_name]["std_r2"] = np.std(scores) if scores else 0.0

    return cv_results


def train_all_sessions(
    all_sessions=None,
    method="per_variable",
    max_iterations=None,
    batch_size=None,
    n_attribution_samples=2000,
    model_kwargs=None,
    wandb_run=None,
    wandb_log_interval=50,
    verbose=True,
):
    """
    Train xCEBRA models on all preprocessed sessions and collect results.

    Parameters
    ----------
    all_sessions : dict, {eid: session_data}
    method : str, "per_variable" or "joint"
    max_iterations : int
    verbose : bool

    Returns
    -------
    all_results : dict, {eid: result_dict}
    neuron_df : DataFrame with per-neuron attributions
    """
    import pandas as pd

    if all_sessions is None:
        all_sessions = load_preprocessed_sessions()

    if not all_sessions:
        print("No preprocessed sessions found. Run preprocessing first.")
        return {}, pd.DataFrame()

    all_results = {}
    neuron_rows = []

    for eid in tqdm(sorted(all_sessions.keys()), desc="Training xCEBRA"):
        session_data = all_sessions[eid]

        result = train_session_xcebra(
            session_data, eid,
            method=method,
            max_iterations=max_iterations,
            batch_size=batch_size,
            n_attribution_samples=n_attribution_samples,
            model_kwargs=model_kwargs,
            wandb_run=wandb_run,
            wandb_log_interval=wandb_log_interval,
            verbose=verbose,
        )

        if result is None:
            continue

        all_results[eid] = result

        # Collect per-neuron data
        meta = session_data["metadata"]
        attr = result["attribution_maps"]
        N = result["N"]

        for ni in range(N):
            row = {
                "eid": eid,
                "ni": ni,
                "uuids": meta["uuids"][ni] if len(meta["uuids"]) > ni else "",
                "acronym": meta["acronym"][ni] if len(meta["acronym"]) > ni else "",
                "mfr_task": float(meta["firing_rate"][ni]) if len(meta["firing_rate"]) > ni else 0.0,
            }
            # Add per-variable attribution scores
            for var_name in VARIABLE_NAMES:
                if var_name in attr:
                    row[f"xcebra_attr_{var_name}"] = float(attr[var_name][ni])
                else:
                    row[f"xcebra_attr_{var_name}"] = 0.0
            neuron_rows.append(row)

    neuron_df = pd.DataFrame(neuron_rows)

    # Save results
    results_path = RESULTS_DIR / "xcebra_neuron_results.json"
    neuron_df.to_json(str(results_path), orient="records", indent=2)
    if verbose:
        print(f"\nResults saved to {results_path}")
        print(f"Total neurons analyzed: {len(neuron_df)}")
        print(f"Total sessions analyzed: {len(all_results)}")

    return all_results, neuron_df


if __name__ == "__main__":
    print("xCEBRA Training Pipeline for IBL Data")
    print("=" * 50)

    # Load preprocessed data
    all_sessions = load_preprocessed_sessions()
    if not all_sessions:
        print("No preprocessed sessions found.")
        print(f"Please run preprocessing first:")
        print(f"  python -m xcebra_ibl.data.preprocess")
    else:
        print(f"Found {len(all_sessions)} preprocessed sessions")
        all_results, neuron_df = train_all_sessions(
            all_sessions, method="per_variable", verbose=True
        )
