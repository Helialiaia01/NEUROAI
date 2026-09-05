"""Auditable trial-held-out pilot: python -m xcebra_ibl.experiments --help."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from xcebra_ibl.configs.config import DATA_RAW_DIR, VARIABLE_NAMES, REMOVE_BLOCK5
from xcebra_ibl.data.preprocess import preprocess_session, _session_id


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def split_trials(count, seed):
    if count < 10:
        raise ValueError('At least 10 retained trials are required')
    order = np.random.default_rng(seed).permutation(count)
    n = max(2, int(count * .2))
    return dict(train=np.sort(order[2*n:]), validation=np.sort(order[n:2*n]), test=np.sort(order[:n]))


def shuffle_trials(labels, trial_length, seed):
    """One joint trial permutation preserves trajectories and label dependence."""
    count = len(next(iter(labels.values()))) // trial_length
    permutation = np.random.default_rng(seed).permutation(count)
    return {v: y.reshape(count, trial_length)[permutation].reshape(-1) for v, y in labels.items()}


def score(y, pred, discrete):
    if not discrete and np.var(y) < 1e-12:
        return None
    if discrete and np.unique(y).size < 2:
        return None
    value = balanced_accuracy_score(y, pred) if discrete else r2_score(y, pred)
    return float(value) if np.isfinite(value) else None


def interval(y, pred, trials, discrete, draws, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(trials == t) for t in np.unique(trials)]
    scores = []
    for _ in range(draws):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        value = score(y[idx], pred[idx], discrete)
        if value is not None:
            scores.append(value)
    return dict(score=score(y, pred, discrete), ci95=np.quantile(scores, [.025, .975]).tolist() if scores else None,
                valid_bootstraps=len(scores), unit='held_out_trial', metric='balanced_accuracy' if discrete else 'r2')


def decode(features, targets, discrete, include_knn=True):
    """Select decoder hyperparameters on validation trials only."""
    choices = []
    for alpha in (.1, 1., 10., 100.):
        estimator = (LogisticRegression(C=1/alpha, max_iter=2000, random_state=0)
                     if discrete else Ridge(alpha=alpha, solver="lsqr"))
        choices.append((f'linear_{alpha}', estimator))
    for k in ((5, 25, 100) if include_knn else ()):
        if k <= len(features['train']):
            choices.append((f'knn_{k}', KNeighborsClassifier(k) if discrete else KNeighborsRegressor(k)))
    if discrete and np.unique(targets['train']).size < 2:
        choices = [('constant', DummyClassifier(strategy='most_frequent'))]
    best = {}
    for name, estimator in choices:
        family = name.split('_')[0]
        model = make_pipeline(StandardScaler(), estimator)
        model.fit(features['train'], targets['train'])
        validation = score(targets['validation'], model.predict(features['validation']), discrete)
        if family not in best or (validation is not None and (best[family][0] is None or validation > best[family][0])):
            best[family] = validation, name, model
    return {family: dict(validation_score=value, decoder=name,
                        prediction=model.predict(features['test']))
            for family, (value, name, model) in best.items()}


def encoding_baselines(session, splits, out):
    """Local time-resolved Ridge/RRR encoding; not the paper's global RRR fit."""
    X, Y = session['X_3d'], session['y_3d']
    output, coefficients, test_predictions = {}, {}, {}
    for rank in (None, 2, 4, 8):
        best = None
        for alpha in (.1, 1., 10., 100.):
            preds = {part: np.empty_like(Y[idx]) for part, idx in splits.items() if part != 'train'}
            fitted_coefficients = []
            for t in range(session['T']):
                model = Ridge(alpha=alpha).fit(X[splits['train'], t], Y[splits['train'], t])
                center = model.intercept_
                projection = None
                if rank is not None:
                    fitted = model.predict(X[splits['train'], t]) - center
                    q, _ = np.linalg.qr(X[splits["train"], t] - X[splits["train"], t].mean(axis=0))
                    _, _, vt = np.linalg.svd(q.T @ fitted, full_matrices=False)
                    basis = vt[:min(rank, len(vt))].T
                    projection = basis
                coef = model.coef_.T
                fitted_coefficients.append(coef if projection is None else (coef @ projection) @ projection.T)
                for part in preds:
                    prediction = model.predict(X[splits[part], t])
                    preds[part][:, t] = prediction if projection is None else ((prediction-center) @ projection) @ projection.T + center
            value = r2_score(Y[splits['validation']].reshape(-1, session['N']), preds['validation'].reshape(-1, session['N']))
            if best is None or value > best[0]:
                best = value, alpha, preds['test'], np.asarray(fitted_coefficients)
        name = 'ridge' if rank is None else f'rrr_rank{rank}'
        coefficients[name] = best[3]
        test_predictions[name] = best[2]
        output[name] = dict(
            validation_r2=float(best[0]), alpha=best[1],
            test_r2_per_neuron=r2_score(Y[splits['test']].reshape(-1, session['N']), best[2].reshape(-1, session['N']), multioutput='raw_values').tolist())
    np.savez_compressed(out / 'encoding_coefficients.npz', **coefficients)
    np.savez_compressed(out / 'encoding_predictions.npz', truth=Y[splits['test']], **test_predictions)
    return output


def run_session(path, out, args):
    if (out / 'scores.json').exists():
        return
    from xcebra_ibl.models.xcebra_model import XCEBRAModel
    import cebra
    import torch
    with np.load(path, allow_pickle=True) as raw:
        behavior = raw['behavior']
        keep = np.array([float(b.split('_')[0]) != .5 for b in behavior]) if REMOVE_BLOCK5 else np.ones(len(behavior), bool)
    splits = split_trials(int(keep.sum()), args.split_seed)
    session = preprocess_session(path, fit_trials=splits['train'], verbose=True)
    if session is None:
        write_json(out / 'skipped.json', {'reason': 'preprocessing_filters_or_missing_variables'})
        return
    T, N = session['T'], session['N']
    session['y_2d'] = session['y_2d'].astype(np.float32)
    if not np.isfinite(session['y_2d']).all() or any(not np.isfinite(a).all() for a in session['label_arrays'].values()):
        raise ValueError('Non-finite preprocessed data')
    np.savez_compressed(out / 'preprocessing.npz', **{k: v for k, v in session['metadata'].items() if k != 'best_delays'}, **splits)
    write_json(out / 'alignment.json', session['metadata']['best_delays'])
    write_json(out / 'encoding.json', encoding_baselines(session, splits, out))
    offset = cebra.models.init('offset10-model', num_neurons=N, num_units=128, num_output=4).get_offset()
    left, right = int(offset.left), int(offset.right)
    interior = np.tile((np.arange(T) >= left) & (np.arange(T) < T-right+1), session['K'])
    masks = {p: np.isin(session['trial_ids'], idx) for p, idx in splits.items()}
    evaluation = {p: masks[p] & interior for p in splits}
    # Baselines receive the same neural channels and temporal receptive field.
    centers = np.flatnonzero(interior)
    contexts = np.stack([session['y_2d'][i-left:i+right].ravel() for i in centers])
    raw_features = {p: contexts[masks[p][centers]] for p in splits}
    truth = {v: {p: session['label_arrays'][v][evaluation[p]] for p in splits} for v in args.variables}
    trial_test = session['trial_ids'][evaluation['test']]
    records, baseline_predictions = [], {}
    for v in args.variables:
        discrete = np.issubdtype(truth[v]['train'].dtype, np.integer)
        for family, result in decode(raw_features, truth[v], discrete, include_knn=False).items():
            prediction = result.pop('prediction')
            baseline_predictions[f'{v}_{family}'] = prediction
            baseline_predictions[f'{v}_truth'] = truth[v]['test']
            records.append(dict(variable=v, representation='neural_context', family=family, **result,
                                **interval(truth[v]['test'], prediction, trial_test, discrete, args.bootstrap, args.split_seed)))
    np.savez_compressed(out / "decoding_baselines.npz", trial_ids=trial_test, **baseline_predictions)
    candidates = {}
    for seed in args.seeds:
        for dim in args.dimensions:
            for control in range(args.nulls + 1):
                key = f'seed{seed}_dim{dim}_' + ('observed' if control == 0 else f'null{control}')
                destination = out / key
                destination.mkdir(exist_ok=True)
                if (destination / 'complete.json').exists():
                    candidates[key] = json.loads((destination / 'complete.json').read_text())
                    continue
                started = time.monotonic()
                train_labels = {v: session['label_arrays'][v][masks['train']] for v in args.variables}
                if control:
                    train_labels = shuffle_trials(train_labels, T, args.split_seed + control)
                model = XCEBRAModel(embedding_dim_per_group=dim, max_iterations=args.iterations,
                    batch_size=args.batch_size, device=args.device, random_seed=seed,
                    checkpoint_dir=destination / 'checkpoints', checkpoint_frequency=args.checkpoint_frequency,
                    checkpoint_retention=1)
                print(f'{path.name}: {key}', flush=True)
                model.fit_per_variable(session['y_2d'][masks['train']], train_labels,
                    session['trial_ids'][masks['train']], session['time_ids'][masks['train']], T, verbose=False)
                model.save(destination)
                embeddings = {p: model.transform_per_variable(session['y_2d'][mask], session['trial_ids'][mask],
                    session['time_ids'][mask], T) for p, mask in masks.items()}
                np.savez_compressed(destination / 'embeddings.npz', **{f'{p}_{v}': e for p, emb in embeddings.items() for v, e in emb.items()})
                predictions, metrics = {}, {}
                for v in args.variables:
                    feats = {p: embeddings[p][v][interior[masks[p]]] for p in splits}
                    targets = dict(truth[v])
                    targets['train'] = train_labels[v][interior[masks['train']]]
                    discrete = np.issubdtype(targets['train'].dtype, np.integer)
                    metrics[v] = {}
                    for family, result in decode(feats, targets, discrete).items():
                        predictions[f'{v}_{family}'] = result.pop('prediction')
                        metrics[v][family] = result
                np.savez_compressed(destination / 'predictions.npz', **predictions)
                attr = model.compute_attribution_maps(session['y_2d'][masks['test']], n_samples=args.attribution_samples,
                    trial_ids=session['trial_ids'][masks['test']], time_ids=session['time_ids'][masks['test']], trial_length=T)
                np.savez_compressed(destination / 'attributions.npz', **attr)
                losses = {v: np.asarray(loss).reshape(-1).tolist() for v, loss in model.training_losses_.items()}
                if any(not np.isfinite(loss).all() for loss in losses.values()):
                    raise ValueError('Non-finite training loss')
                write_json(destination / 'losses.json', losses)
                candidates[key] = dict(metrics=metrics, seconds=time.monotonic()-started, seed=seed, dimension=dim, control=control)
                write_json(destination / 'complete.json', candidates[key])
                del model, embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    # Select dimensions by mean validation performance across observed seeds.
    selected = {}
    for v in args.variables:
        families = next(iter(candidates.values()))['metrics'][v]
        for family in families:
            values = {d: [c['metrics'][v][family]['validation_score'] for c in candidates.values()
                           if c['dimension'] == d and c['control'] == 0] for d in args.dimensions}
            valid = {d: np.mean([x for x in vals if x is not None]) for d, vals in values.items() if any(x is not None for x in vals)}
            if not valid:
                continue
            chosen = max(valid, key=valid.get)
            selected[f'{v}_{family}'] = chosen
            for key, c in candidates.items():
                if c['dimension'] != chosen:
                    continue
                with np.load(out / key / 'predictions.npz') as saved:
                    pred = saved[f'{v}_{family}']
                discrete = np.issubdtype(truth[v]['train'].dtype, np.integer)
                records.append(dict(variable=v, representation='regularized_cebra_adaptation', family=family,
                    seed=c['seed'], dimension=chosen, control=c['control'],
                    **c['metrics'][v][family], **interval(truth[v]['test'], pred, trial_test, discrete, args.bootstrap, args.split_seed)))
    write_json(out / 'stability.json', stability(out, candidates, args.variables, T, left, right))
    write_json(out / 'scores.json', dict(selected_dimensions=selected, decoding=records,
        interpretation='Within-session generalization to held-out trials. Nulls are trial-exchangeability diagnostics, not calibrated significance tests.'))


def stability(out, candidates, variables, T, left, right):
    """Align validation embeddings; measure agreement on untouched test trials."""
    from itertools import combinations
    from scipy.stats import spearmanr
    result = []
    observed = [(k, c) for k, c in candidates.items() if c['control'] == 0]
    for (ka, a), (kb, b) in combinations(observed, 2):
        if a['dimension'] != b['dimension']:
            continue
        with np.load(out / ka / 'embeddings.npz') as ea, np.load(out / kb / 'embeddings.npz') as eb, np.load(out / ka / 'attributions.npz') as aa, np.load(out / kb / 'attributions.npz') as ab:
            for v in variables:
                av, bv = ea[f'validation_{v}'], eb[f'validation_{v}']
                at, bt = ea[f'test_{v}'], eb[f'test_{v}']
                iv = np.tile((np.arange(T)>=left)&(np.arange(T)<T-right+1), len(av)//T)
                it = np.tile((np.arange(T)>=left)&(np.arange(T)<T-right+1), len(at)//T)
                align = Ridge(alpha=1.).fit(av[iv], bv[iv])
                rho = spearmanr(aa[v], ab[v]).statistic
                result.append(dict(variable=v, dimension=a['dimension'], seeds=[a['seed'], b['seed']],
                    aligned_test_r2=float(r2_score(bt[it], align.predict(at[it]))),
                    attribution_spearman=float(rho) if np.isfinite(rho) else None))
    return result


def summarize(output, draws, seed):
    """Average seeds within session before bootstrapping sessions."""
    grouped = {}
    for path in sorted(output.glob('*/scores.json')):
        session = json.loads(path.read_text())
        for row in session['decoding']:
            if row['score'] is None:
                continue
            key = (row['variable'], row['representation'], row['family'], row.get('control', 0))
            grouped.setdefault(key, {}).setdefault(path.parent.name, []).append(row['score'])
    rng = np.random.default_rng(seed)
    result = []
    for (v, representation, family, control), sessions in grouped.items():
        means = np.array([np.mean(values) for values in sessions.values()])
        bootstrap = rng.choice(means, (draws, len(means)), replace=True).mean(axis=1)
        result.append(dict(variable=v, representation=representation, family=family, control=control,
            sessions=len(means), mean_score=float(means.mean()),
            session_bootstrap_ci95=np.quantile(bootstrap, [.025,.975]).tolist() if len(means)>1 else None,
            session_scores={eid:float(np.mean(scores)) for eid,scores in sessions.items()}))
    write_json(output / 'summary.json', result)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=DATA_RAW_DIR)
    p.add_argument('--output', type=Path, default=Path(os.environ.get('KAGGLE_WORKING_DIR', 'outputs')) / 'pilot')
    p.add_argument('--max-sessions', type=int, default=3)
    p.add_argument('--seeds', type=int, nargs='+', default=[2025, 2026, 2027])
    p.add_argument('--dimensions', type=int, nargs='+', default=[2, 4, 8])
    p.add_argument('--variables', nargs='+', choices=VARIABLE_NAMES, default=VARIABLE_NAMES)
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--nulls', type=int, default=1)
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--attribution-samples', type=int, default=256)
    p.add_argument('--checkpoint-frequency', type=int, default=100)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--device', default='cuda_if_available')
    args = p.parse_args(argv)
    for name in ('max_sessions', 'iterations', 'bootstrap', 'batch_size', 'attribution_samples', 'checkpoint_frequency'):
        if getattr(args, name) < 1:
            p.error(f'{name} must be positive')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.dimensions)) != len(args.dimensions):
        p.error('Seeds and dimensions must be unique')
    if args.nulls < 1 or min(args.dimensions) < 2:
        p.error('At least one null and dimensions >=2 required')
    sources = sorted(args.data_dir.rglob('*.npz'))[:args.max_sessions]
    if not sources:
        raise FileNotFoundError(f'No session NPZ files under {args.data_dir}')
    if len({_session_id(f) for f in sources}) != len(sources):
        raise ValueError('Duplicate session IDs in input dataset')
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    import torch
    fingerprints = []
    for path in sources:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                digest.update(chunk)
        fingerprints.append(dict(name=path.name, sha256=digest.hexdigest()))
    code_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob('*.py')):
        code_hash.update(str(path.relative_to(Path(__file__).parent)).encode())
        code_hash.update(path.read_bytes())
    manifest = dict(runtime=dict(cuda=torch.version.cuda, device=args.device,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), config=config, sources=fingerprints, code_sha256=code_hash.hexdigest(),
                    packages={name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'cebra', 'scikit-learn', 'scipy')})
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Output contains a different experiment; use a new --output directory')
    write_json(manifest_path, manifest)
    print(f'Pilot upper bound: {len(sources)*len(args.seeds)*len(args.dimensions)*(args.nulls+1)*len(args.variables)} encoder fits', flush=True)
    completed = 0
    for path in sources:
        destination = args.output / _session_id(path)
        destination.mkdir(exist_ok=True)
        run_session(path, destination, args)
        completed += (destination / 'scores.json').exists()
    if not completed:
        raise RuntimeError('No sessions passed preprocessing')
    summarize(args.output, args.bootstrap, args.split_seed)
    write_json(args.output / 'complete.json', dict(completed_sessions=completed, requested_sessions=len(sources)))


if __name__ == '__main__':
    main()
