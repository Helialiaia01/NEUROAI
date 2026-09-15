"""Auditable trial-held-out pilot: python -m xcebra_ibl.experiments --help."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import resource
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from xcebra_ibl.configs.config import DATA_RAW_DIR, VARIABLE_NAMES, REMOVE_BLOCK5, CORTICAL_AREAS
from xcebra_ibl.data.preprocess import preprocess_session, _session_id
from xcebra_ibl.experiment.evaluation import split_trials, shuffle_trials, score, interval, decode, encoding_baselines
from xcebra_ibl.experiment.design import select_sources, block_ids, context_features, session_qc
from xcebra_ibl.experiment.artifacts import write_json, complete, verified, sha256, interruption_status, warning_report


def run_session(path, out, args):
    if verified(out, 'session_complete.json'):
        return
    started_session = time.monotonic()
    from xcebra_ibl.models.xcebra_model import XCEBRAModel
    import cebra
    import torch
    with np.load(path, allow_pickle=True) as raw:
        behavior = raw['behavior']
        keep = np.array([float(b.split('_')[0]) != .5 for b in behavior]) if REMOVE_BLOCK5 else np.ones(len(behavior), bool)
    blocks = block_ids(behavior)[keep]
    if keep.sum() < 10:
        write_json(out / 'skipped.json', {'reason': 'fewer_than_10_retained_trials'})
        return
    splits = split_trials(int(keep.sum()), args.split_seed)
    session = preprocess_session(path, fit_trials=splits['train'], verbose=True,
                                 include_areas=CORTICAL_AREAS if args.areas == 'cortical' else None)
    if session is None:
        write_json(out / 'skipped.json', {'reason': 'preprocessing_filters_or_missing_variables'})
        return
    write_json(out / "qc.json", session_qc(session, splits, blocks, args.subjects.get(session["eid"]), args.areas))
    T, N = session['T'], session['N']
    session['y_2d'] = session['y_2d'].astype(np.float32)
    if not np.isfinite(session['y_2d']).all() or any(not np.isfinite(a).all() for a in session['label_arrays'].values()):
        raise ValueError('Non-finite preprocessed data')
    np.savez_compressed(out / 'preprocessing.npz', block_ids=blocks, **{k: v for k, v in session['metadata'].items() if k != 'best_delays'}, **splits)
    write_json(out / 'alignment.json', session['metadata']['best_delays'])
    write_json(out / 'encoding.json', encoding_baselines(session, splits, out))
    offset = cebra.models.init('offset10-model', num_neurons=N, num_units=128, num_output=4).get_offset()
    left, right = int(offset.left), int(offset.right)
    interior = np.tile((np.arange(T) >= left) & (np.arange(T) < T-right+1), session['K'])
    masks = {p: np.isin(session['trial_ids'], idx) for p, idx in splits.items()}
    evaluation = {p: masks[p] & interior for p in splits}
    # Baselines receive the same neural channels and temporal receptive field.
    context_dir = out / '.context'
    raw_features = context_features(session['y_2d'], masks, interior, left, right, context_dir)
    truth = {v: {p: session['label_arrays'][v][evaluation[p]] for p in splits} for v in args.variables}
    trial_test = session['trial_ids'][evaluation['test']]
    bootstrap_ids = blocks[trial_test] if args.bootstrap_unit == 'block' else trial_test
    bootstrap_unit = 'held_out_block' if args.bootstrap_unit == 'block' else 'held_out_trial'
    records, baseline_predictions = [], {}
    for v in args.variables:
        discrete = np.issubdtype(truth[v]['train'].dtype, np.integer)
        for family, result in decode(raw_features, truth[v], discrete, include_knn=False, save_dir=out / "decoders", prefix=v).items():
            prediction = result.pop('prediction')
            baseline_predictions[f'{v}_{family}'] = prediction
            baseline_predictions[f'{v}_truth'] = truth[v]['test']
            records.append(dict(variable=v, representation='neural_context', family=family, **result,
                                **interval(truth[v]['test'], prediction, bootstrap_ids, discrete, args.bootstrap, args.split_seed, unit=bootstrap_unit)))
    np.savez_compressed(out / "decoding_baselines.npz", trial_ids=trial_test, bootstrap_ids=bootstrap_ids, **baseline_predictions)
    del raw_features
    for path_to_remove in context_dir.glob("*.npy"):
        path_to_remove.unlink()
    context_dir.rmdir()
    candidates = {}
    for seed in args.seeds:
        for dim in args.dimensions:
            for control in range(args.nulls + 1):
                key = f'seed{seed}_dim{dim}_' + ('observed' if control == 0 else f'null{control}')
                destination = out / key
                destination.mkdir(exist_ok=True)
                recovered = verified(destination)
                if recovered:
                    candidates[key] = recovered
                    continue
                started = time.monotonic()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                train_labels = {v: session['label_arrays'][v][masks['train']] for v in args.variables}
                if control:
                    train_labels = shuffle_trials(train_labels, T, args.split_seed + control,
                        groups=blocks[splits['train']] if args.shuffle == 'within_block' else None)
                model = XCEBRAModel(embedding_dim_per_group=dim, max_iterations=args.iterations,
                    batch_size=args.batch_size, device=args.device, random_seed=seed,
                    checkpoint_dir=destination / 'checkpoints', checkpoint_frequency=args.checkpoint_frequency,
                    checkpoint_retention=1, recovery_dir=destination / "recovery",
                    learning_rate=args.learning_rate, temperature=args.temperature,
                    num_hidden_units=args.hidden_units, time_offsets=args.time_offset,
                    jacobian_reg_weight=args.regularization)
                print(f'{path.name}: {key}', flush=True)
                model.fit_per_variable(session['y_2d'][masks['train']], train_labels,
                    session['trial_ids'][masks['train']], session['time_ids'][masks['train']], T, verbose=False)
                model.save(destination)
                write_json(destination / 'diagnostics.json', model.training_diagnostics_)
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
                    for family, result in decode(feats, targets, discrete, save_dir=destination / "decoders", prefix=v).items():
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
                candidates[key] = dict(metrics=metrics, seconds=time.monotonic()-started, seed=seed, dimension=dim, control=control,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                    changed_supervision={v:bool(np.any(train_labels[v] != session['label_arrays'][v][masks['train']])) for v in args.variables})
                complete(destination, candidates[key])
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
                    **c['metrics'][v][family], **interval(truth[v]['test'], pred, bootstrap_ids, discrete, args.bootstrap, args.split_seed,
                        baseline=baseline_predictions.get(f'{v}_linear', baseline_predictions.get(f'{v}_constant')),
                        unit=bootstrap_unit)))
    write_json(out / 'stability.json', stability(out, candidates, args.variables, T, left, right))
    write_json(out / 'scores.json', dict(selected_dimensions=selected, decoding=records,
        interpretation='Within-session generalization to held-out trials. Nulls are trial-exchangeability diagnostics, not calibrated significance tests.'))
    write_json(out / 'profile.json', dict(seconds=time.monotonic()-started_session,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024),
        note='RSS is the process high-water mark, not isolated session memory'))
    complete(out, {'status': 'complete'}, marker='session_complete.json')


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
    grouped, subjects, improvements = {}, {}, {}
    for path in sorted(output.glob('*/scores.json')):
        session = json.loads(path.read_text())
        qc = json.loads((path.parent / 'qc.json').read_text())
        subjects[path.parent.name] = qc.get('subject')
        for row in session['decoding']:
            if row['score'] is None:
                continue
            key = (row['variable'], row['representation'], row['family'], row.get('control', 0))
            grouped.setdefault(key, {}).setdefault(path.parent.name, []).append(row['score'])
            if row.get('improvement') is not None:
                improvements.setdefault(key, {}).setdefault(path.parent.name, []).append(row['improvement'])
    rng = np.random.default_rng(seed)
    result = []
    for (v, representation, family, control), sessions in grouped.items():
        def aggregate(groups):
            units = {}
            all_subjects = all(subjects[eid] for eid in groups)
            for eid, scores in groups.items():
                unit = subjects[eid] if all_subjects else eid
                units.setdefault(unit, []).append(float(np.mean(scores)))
            means = np.array([np.mean(values) for values in units.values()])
            bootstrap = rng.choice(means, (draws, len(means)), replace=True).mean(axis=1)
            return dict(mean=float(means.mean()), ci95=np.quantile(bootstrap,[.025,.975]).tolist() if len(means)>1 else None,
                        unit='subject' if all_subjects else 'session_subjects_unknown', n_units=len(means))
        key = (v, representation, family, control)
        result.append(dict(variable=v, representation=representation, family=family, control=control,
            scores=aggregate(sessions), paired_improvement=aggregate(improvements[key]) if key in improvements else None,
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
    p.add_argument('--device', default='cuda_if_available',
                   help='Automatic preference: CUDA, then MPS, then CPU; explicit cuda requires an available GPU')
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--hidden-units', type=int, default=128)
    p.add_argument('--time-offset', type=int, default=10)
    p.add_argument('--regularization', type=float, default=.01)
    p.add_argument('--areas', choices=['cortical', 'all'], default='cortical')
    p.add_argument('--bootstrap-unit', choices=['trial', 'block'], default='trial')
    p.add_argument('--shuffle', choices=['trial', 'within_block'], default='trial')
    p.add_argument('--cohort', type=Path)
    p.add_argument('--phase', choices=['exploratory', 'confirmatory'], default='exploratory')
    p.add_argument('--session-ids', nargs='+')
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args(argv)
    from cebra.integrations.sklearn.utils import check_device
    requested_device = args.device
    args.device = check_device(args.device)
    for name in ('max_sessions', 'iterations', 'bootstrap', 'batch_size', 'attribution_samples', 'checkpoint_frequency'):
        if getattr(args, name) < 1:
            p.error(f'{name} must be positive')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.dimensions)) != len(args.dimensions):
        p.error('Seeds and dimensions must be unique')
    if min(args.seeds)<0 or args.split_seed<0:
        p.error('Seeds must be nonnegative')
    if args.nulls < 1 or min(args.dimensions) < 2:
        p.error('At least one null and dimensions >=2 required')
    if min(args.learning_rate, args.temperature, args.hidden_units, args.time_offset) <= 0 or args.regularization < 0:
        p.error('Model parameters must be positive; regularization may be zero')
    sources, subjects = select_sources(args)
    if not sources:
        raise FileNotFoundError(f'No session NPZ files under {args.data_dir}')
    if len({_session_id(f) for f in sources}) != len(sources):
        raise ValueError('Duplicate session IDs in input dataset')
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args.subjects = subjects
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
        if any(part in {'trained_models','data_processed','results'} for part in path.relative_to(Path(__file__).parent).parts):
            continue
        code_hash.update(str(path.relative_to(Path(__file__).parent)).encode())
        code_hash.update(path.read_bytes())
    from xcebra_ibl.configs import config as defaults
    config['preprocessing'] = {key:getattr(defaults,key) for key in (
        'MIN_TRIALS','MIN_NEURONS','MIN_FIRING_RATE','MAX_SILENT_PROB','UNIT_LABEL_MIN',
        'GAUSSIAN_SMOOTH_SIGMA','STANDARDIZE_Y','STANDARDIZE_X','TRANSFORM_MFR','SPSDT','REMOVE_BLOCK5')}
    config['method'] = 'regularized_cebra_adaptation'
    config['dimension_selection'] = 'mean_validation_decoding; stability reported separately'
    config['subject_ids'] = subjects
    config['cohort_sha256'] = sha256(args.cohort) if args.cohort else None
    config['input_units'] = 'firing_rate_hz_reference_export'
    config['maximum_firing_rate_filter'] = None
    config['architecture'] = 'offset10-model'
    config['attribution'] = dict(method='mean_abs_pinv_of_time_averaged_jacobian', rcond=1e-5, n_proj=-1, batch_size=256)
    config['area_inclusion'] = CORTICAL_AREAS if args.areas=='cortical' else None
    manifest = dict(runtime=dict(cuda=torch.version.cuda, device=args.device, requested_device=requested_device,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), config=config, sources=fingerprints, code_sha256=code_hash.hexdigest(),
                    packages={name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'cebra', 'scikit-learn', 'scipy')})
    if args.preflight_only:
        write_json(args.output / 'preflight.json', manifest)
        print(f'Preflight: {len(sources)} sessions; CUDA available: {torch.cuda.is_available()}')
        return
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Output contains a different experiment; use a new --output directory')
    write_json(manifest_path, manifest)
    print(f'Pilot upper bound: {len(sources)*len(args.seeds)*len(args.dimensions)*(args.nulls+1)*len(args.variables)} encoder fits', flush=True)
    completed, skipped = 0, []
    for path in sources:
        destination = args.output / _session_id(path)
        destination.mkdir(exist_ok=True)
        if not verified(destination, "session_complete.json"):
            with interruption_status(args.output), warning_report(destination):
                run_session(path, destination, args)
            if (destination / "scores.json").exists():
                complete(destination, {"status": "complete"}, marker="session_complete.json")
            elif (destination / "skipped.json").exists():
                complete(destination, {"status": "skipped"}, marker="session_complete.json")
        completed += (destination / 'scores.json').exists()
        if (destination / 'skipped.json').exists():
            skipped.append(dict(eid=destination.name, **json.loads((destination/'skipped.json').read_text())))
    summarize(args.output, args.bootstrap, args.split_seed)
    write_json(args.output / 'complete.json', dict(completed_sessions=completed, requested_sessions=len(sources),
        skipped_sessions=skipped, status='complete' if completed else 'all_sessions_excluded'))
    if not completed:
        print('All requested sessions were explicitly excluded by preprocessing; no scientific scores produced.', flush=True)


if __name__ == '__main__':
    main()
