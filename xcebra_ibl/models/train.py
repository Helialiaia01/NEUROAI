"""
Step 3: Train xCEBRA models on IBL sessions.

For each session, trains per-variable xCEBRA models and extracts
per-neuron attribution maps (Jacobian-based selectivity).

This replaces the RRR encoding model (step2_train_RRR.py from brainwide-RRR).
"""

import numpy as np
import pickle
import json
import gc
import traceback
import hashlib
import random
from pathlib import Path
from tqdm import tqdm

from xcebra_ibl.configs.config import (
    DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    VARIABLE_NAMES, N_VARIABLES, N_CV_FOLDS, TEST_FRACTION,
    MAX_ITERATIONS, BATCH_SIZE, EMBEDDING_DIM_PER_GROUP, RANDOM_SEED,
)
from xcebra_ibl.data.preprocess import (
    load_preprocessed_sessions,
    build_neuron_dataframe,
    preprocess_session,
    _session_id,
)
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
    if "label_arrays" in session_data:
        return {
            var_name: np.asarray(values)
            for var_name, values in session_data["label_arrays"].items()
        }

    # Preserve integer classes for categorical variables.  Falling back to
    # X_2d keeps old caches readable, but new preprocessing writes labels_2d.
    X_2d = session_data.get("labels_2d", session_data["X_2d"])
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
    checkpoint_retention=None,
    seed=None,
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

    if seed is None:
        seed = RANDOM_SEED
    # Give every session a stable but distinct random stream.
    session_seed = int.from_bytes(
        hashlib.sha256(str(eid).encode("utf-8")).digest()[:4], "little"
    ) ^ int(seed)
    random.seed(session_seed)
    np.random.seed(session_seed)
    try:
        import torch
        torch.manual_seed(session_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(session_seed)
    except ImportError:
        pass

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
        checkpoint_retention=checkpoint_retention,
        random_seed=session_seed,
        **model_kwargs,
    )

    if method == "per_variable":
        model.fit_per_variable(
            neural_data,
            labels,
            trial_ids=session_data.get("trial_ids"),
            time_ids=session_data.get("time_ids"),
            trial_length=T,
            verbose=verbose,
        )
    elif method == "joint":
        model.fit_joint(
            neural_data,
            labels,
            trial_ids=session_data.get("trial_ids"),
            time_ids=session_data.get("time_ids"),
            trial_length=T,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract embeddings
    if verbose:
        print("\n  Extracting embeddings...")
    if method == "per_variable":
        embeddings = model.transform_per_variable(
            neural_data,
            trial_ids=session_data.get("trial_ids"),
            time_ids=session_data.get("time_ids"),
            trial_length=T,
        )
    else:
        embeddings = model.transform_joint(
            neural_data,
            trial_ids=session_data.get("trial_ids"),
            time_ids=session_data.get("time_ids"),
            trial_length=T,
        )

    # Compute attribution maps (Jacobian-based)
    if verbose:
        print("\n  Computing attribution maps (Jacobian)...")
    attribution_maps = model.compute_attribution_maps(
        neural_data,
        method=method,
        n_samples=min(n_attribution_samples, K * T),
        trial_ids=session_data.get("trial_ids"),
        time_ids=session_data.get("time_ids"),
        trial_length=T,
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


def _neuron_rows(session_data, result):
    """Convert one session result into JSON-serializable neuron rows."""
    meta = session_data["metadata"]
    attr = result["attribution_maps"]
    rows = []
    for ni in range(result["N"]):
        row = {
            "eid": result["eid"],
            "ni": ni,
            "uuids": meta["uuids"][ni] if len(meta["uuids"]) > ni else "",
            "acronym": meta["acronym"][ni] if len(meta["acronym"]) > ni else "",
            "mfr_task": float(meta["firing_rate"][ni])
            if len(meta["firing_rate"]) > ni
            else 0.0,
        }
        for var_name in VARIABLE_NAMES:
            row[f"xcebra_attr_{var_name}"] = (
                float(attr[var_name][ni]) if var_name in attr else 0.0
            )
        rows.append(row)
    return rows


def _write_json(path, payload):
    """Atomically write a small JSON state file."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def train_streaming_sessions(
    data_dir=None,
    results_dir=None,
    method="per_variable",
    max_iterations=None,
    batch_size=None,
    n_attribution_samples=2000,
    checkpoint_frequency=1,
    checkpoint_retention=2,
    max_sessions=None,
    wandb_run=None,
    wandb_log_interval=50,
    save_models=True,
    seed=None,
    verbose=True,
):
    """Preprocess and train sessions one at a time with resumable state.

    This is the production path for constrained runtimes such as Kaggle. It
    never stores the full preprocessed dataset or all trained model objects in
    memory. Results are appended as JSONL and a small state file records which
    sessions completed or were skipped.
    """
    from xcebra_ibl.configs.config import DATA_RAW_DIR, RESULTS_DIR

    data_dir = Path(data_dir) if data_dir is not None else DATA_RAW_DIR
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    result_jsonl = results_dir / "xcebra_neuron_results.jsonl"
    result_json = results_dir / "xcebra_neuron_results.json"
    state_path = results_dir / "streaming_state.json"

    state = {"version": 1, "sessions": {}}
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        state.setdefault("sessions", {})

    # If a prior run was interrupted after appending rows but before updating
    # state, use the JSONL file as an additional resume signal.
    completed = {
        eid
        for eid, info in state["sessions"].items()
        if info.get("status") in {"complete", "skipped"}
    }
    if result_jsonl.exists():
        with result_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completed.add(json.loads(line)["eid"])

    npz_files = sorted(data_dir.rglob("*.npz"))
    if max_sessions is not None:
        npz_files = npz_files[:max_sessions]
    if not npz_files:
        raise FileNotFoundError(f"No .npz session files found under {data_dir}")

    print(
        f"Streaming {len(npz_files)} sessions; "
        f"already complete/skipped: {len(completed)}"
    )

    for npz_path in tqdm(npz_files, desc="Streaming xCEBRA sessions"):
        eid = _session_id(npz_path)
        if eid in completed:
            continue

        state["sessions"][eid] = {
            "status": "processing",
            "source": str(npz_path.name),
        }
        _write_json(state_path, state)

        session_data = None
        try:
            session_data = preprocess_session(npz_path, verbose=verbose)
            if session_data is None:
                state["sessions"][eid] = {
                    "status": "skipped",
                    "source": str(npz_path.name),
                    "reason": "preprocessing_filters",
                }
                _write_json(state_path, state)
                continue

            result = train_session_xcebra(
                session_data,
                eid,
                method=method,
                max_iterations=max_iterations,
                batch_size=batch_size,
                n_attribution_samples=n_attribution_samples,
                wandb_run=wandb_run,
                wandb_log_interval=wandb_log_interval,
                save_models=save_models,
                checkpoint_frequency=checkpoint_frequency,
                checkpoint_retention=checkpoint_retention,
                seed=seed,
                verbose=verbose,
            )
            if result is None:
                state["sessions"][eid] = {
                    "status": "skipped",
                    "source": str(npz_path.name),
                    "reason": "training_filters",
                }
                _write_json(state_path, state)
                continue

            with result_jsonl.open("a", encoding="utf-8") as handle:
                for row in _neuron_rows(session_data, result):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

            state["sessions"][eid] = {
                "status": "complete",
                "source": str(npz_path.name),
                "neurons": int(result["N"]),
                "trials": int(result["K"]),
            }
            _write_json(state_path, state)
            completed.add(eid)

        except Exception as exc:
            state["sessions"][eid] = {
                "status": "failed",
                "source": str(npz_path.name),
                "error": repr(exc),
            }
            _write_json(state_path, state)
            traceback.print_exc()
            print(f"Session {eid} failed; continuing with the next session.")
        finally:
            # Release the large arrays and model graphs before the next raw
            # session is loaded. This is important on both disk- and RAM-
            # constrained workers.
            session_data = None
            result = None
            gc.collect()

    import pandas as pd

    if result_jsonl.exists():
        neuron_df = pd.read_json(str(result_jsonl), lines=True)
    else:
        neuron_df = pd.DataFrame()
    neuron_df.to_json(str(result_json), orient="records", indent=2)

    summary = {
        "total_sources": len(npz_files),
        "complete": sum(
            info.get("status") == "complete"
            for info in state["sessions"].values()
        ),
        "skipped": sum(
            info.get("status") == "skipped"
            for info in state["sessions"].values()
        ),
        "failed": sum(
            info.get("status") == "failed"
            for info in state["sessions"].values()
        ),
        "neurons": len(neuron_df),
    }
    state["summary"] = summary
    _write_json(state_path, state)
    print(f"Streaming summary: {summary}")
    print(f"Results saved to {result_json}")
    return neuron_df


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
    cv_results : dict
        Per-variable held-out scores. Continuous variables report R²;
        categorical variables report balanced accuracy. Splits are by trial.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, r2_score
    from sklearn.model_selection import KFold

    if n_folds is None:
        n_folds = N_CV_FOLDS
    if test_fraction is None:
        test_fraction = TEST_FRACTION

    if "source_path" not in session_data:
        raise ValueError("Leakage-safe CV requires source_path; regenerate cached preprocessing or use xcebra_ibl.experiments")
    neural_data = session_data["y_2d"]
    labels = prepare_labels_from_session(session_data)
    K = session_data["K"]
    T = session_data["T"]

    # Use trial-based cross-validation (same trial never in both train/test)
    trial_ids = session_data["trial_ids"]
    unique_trials = np.unique(trial_ids)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_results = {}
    for var_name in VARIABLE_NAMES:
        is_discrete = np.issubdtype(labels[var_name].dtype, np.integer)
        cv_results[var_name] = {
            "scores": [],
            "metric": "balanced_accuracy" if is_discrete else "r2",
        }

    for fold_idx, (train_trial_idx, test_trial_idx) in enumerate(kf.split(unique_trials)):
        if verbose:
            print(f"\n  Fold {fold_idx + 1}/{n_folds}")

        train_trials = unique_trials[train_trial_idx]
        test_trials = unique_trials[test_trial_idx]

        fold_session = preprocess_session(session_data["source_path"], fit_trials=train_trials)
        if fold_session is None:
            raise ValueError("Fold does not pass training-only preprocessing filters")
        neural_data = fold_session["y_2d"]
        labels = prepare_labels_from_session(fold_session)

        train_mask = np.isin(trial_ids, train_trials)
        test_mask = np.isin(trial_ids, test_trials)

        train_neural = neural_data[train_mask]
        test_neural = neural_data[test_mask]
        train_labels = {v: labels[v][train_mask] for v in VARIABLE_NAMES}
        test_labels = {v: labels[v][test_mask] for v in VARIABLE_NAMES}
        train_trial_ids = trial_ids[train_mask]
        test_trial_ids = trial_ids[test_mask]
        train_time_ids = session_data["time_ids"][train_mask]
        test_time_ids = session_data["time_ids"][test_mask]

        # Train CEBRA on training data
        model = XCEBRAModel(
            max_iterations=MAX_ITERATIONS // 2,  # Fewer iters for CV
            batch_size=min(BATCH_SIZE, train_neural.shape[0]),
        )

        if method == "per_variable":
            model.fit_per_variable(
                train_neural,
                train_labels,
                trial_ids=train_trial_ids,
                time_ids=train_time_ids,
                trial_length=T,
                verbose=False,
            )
            train_emb = model.transform_per_variable(
                train_neural, train_trial_ids, train_time_ids, T
            )
            test_emb = model.transform_per_variable(
                test_neural, test_trial_ids, test_time_ids, T
            )
        else:
            model.fit_joint(
                train_neural,
                train_labels,
                trial_ids=train_trial_ids,
                time_ids=train_time_ids,
                trial_length=T,
                verbose=False,
            )
            train_emb = model.transform_joint(
                train_neural, train_trial_ids, train_time_ids, T
            )
            test_emb = model.transform_joint(
                test_neural, test_trial_ids, test_time_ids, T
            )

        # Decode each variable from embeddings using Ridge regression
        for var_name in VARIABLE_NAMES:
            if var_name not in train_emb:
                continue

            y_train = train_labels[var_name]
            y_test = test_labels[var_name]
            if np.issubdtype(y_train.dtype, np.integer):
                # R² is not meaningful for category IDs. Use balanced
                # accuracy, which remains interpretable under class imbalance.
                if np.unique(y_train).size < 2:
                    decoder = DummyClassifier(strategy="most_frequent")
                else:
                    decoder = LogisticRegression(max_iter=1000, random_state=42)
                decoder.fit(train_emb[var_name], y_train)
                pred = decoder.predict(test_emb[var_name])
                score = balanced_accuracy_score(y_test, pred)
            else:
                decoder = Ridge(alpha=1.0)
                decoder.fit(train_emb[var_name], y_train)
                pred = decoder.predict(test_emb[var_name])
                score = r2_score(y_test, pred)
            cv_results[var_name]["scores"].append(float(score))

    # Compute means
    for var_name in VARIABLE_NAMES:
        scores = cv_results[var_name]["scores"]
        cv_results[var_name]["mean_score"] = np.mean(scores) if scores else np.nan
        cv_results[var_name]["std_score"] = np.std(scores) if scores else np.nan
        if cv_results[var_name]["metric"] == "r2":
            cv_results[var_name]["mean_r2"] = cv_results[var_name]["mean_score"]
            cv_results[var_name]["std_r2"] = cv_results[var_name]["std_score"]
        else:
            cv_results[var_name]["mean_balanced_accuracy"] = cv_results[var_name]["mean_score"]
            cv_results[var_name]["std_balanced_accuracy"] = cv_results[var_name]["std_score"]

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
    seed=None,
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
            seed=seed,
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
