"""Re-evaluate saved pilot predictions without training or changing source artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np

from xcebra_ibl.experiment.artifacts import sha256, write_json
from xcebra_ibl.experiment.evaluation import interval


def review(source, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Review output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    manifests = json.loads((source/'worker_manifests.json').read_text())
    provenance, diagnostics, changes, support, stability, profiles = {}, [], [], [], [], []
    skipped = []

    def read(path):
        provenance[str(path.relative_to(source))] = sha256(path)
        return json.loads(path.read_text())

    read(source/'worker_manifests.json')
    for eid, manifest in sorted(manifests.items()):
        directory = source/eid
        marker = read(directory/'session_complete.json')
        if marker['status'] == 'skipped':
            skipped.append(eid)
            continue
        if marker['status'] != 'complete':
            raise ValueError(f'Incomplete session: {eid}')
        # Validate every copied input against the original session manifest.
        # Large unused artifacts need not be downloaded for this review.
        for name, expected in marker['artifacts'].items():
            path = directory/name
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError('Artifact path escapes session directory')
            if path.is_file() and sha256(path) != expected:
                raise ValueError(f'Corrupted review input: {path}')
        scores = read(directory/'scores.json')
        config = manifest['config']
        baseline_file = directory/'decoding_baselines.npz'
        provenance[str(baseline_file.relative_to(source))] = sha256(baseline_file)
        baseline = dict(np.load(baseline_file))
        candidates = {}
        for path in sorted(directory.glob('seed*/complete.json')):
            candidate = read(path)
            key = (candidate['seed'], candidate['dimension'], candidate['control'])
            candidates[key] = path.parent
            prediction_file = path.parent/'predictions.npz'
            if sha256(prediction_file) != candidate['artifacts']['predictions.npz']:
                raise ValueError(f'Prediction checksum mismatch: {prediction_file}')
            provenance[str(prediction_file.relative_to(source))] = sha256(prediction_file)
            for variable, series in read(path.parent/'diagnostics.json').items():
                for name, values in series.items():
                    x = np.asarray(values, dtype=float)
                    if not len(x) or not np.isfinite(x).all():
                        raise ValueError(f'Invalid diagnostics: {path.parent}/{variable}/{name}')
                    width = max(1, len(x)//5)
                    diagnostics.append(dict(eid=eid, candidate=path.parent.name,
                        variable=variable, metric=name, steps=len(x),
                        first_mean=float(x[:width].mean()), last_mean=float(x[-width:].mean()),
                        previous_mean=float(x[-2*width:-width].mean()) if len(x)>=2*width else None))
        cached = {}
        print(f'Re-evaluating {eid}: {len(scores["decoding"])} rows', flush=True)
        for row in scores['decoding']:
            v, family = row['variable'], row['family']
            y = baseline[f'{v}_truth']
            discrete = row['metric'] == 'balanced_accuracy'
            base = baseline.get(f'{v}_linear', baseline.get(f'{v}_constant'))
            if row['representation'] == 'neural_context':
                pred = baseline[f'{v}_{family}']
                paired = None
            else:
                key = (row['seed'], row['dimension'], row['control'])
                if key not in cached:
                    with np.load(candidates[key]/'predictions.npz') as saved:
                        cached[key] = dict(saved)
                pred = cached[key][f'{v}_{family}']
                paired = base
            new = interval(y, pred, baseline['bootstrap_ids'], discrete,
                           config['bootstrap'], config['split_seed'],
                           baseline=paired, unit=row['unit'])
            if not np.isclose(new['score'], row['score'], rtol=1e-7, atol=1e-9):
                raise ValueError(f'Point estimate changed: {eid}/{v}/{family}')
            changes.append(dict(eid=eid, variable=v, family=family,
                seed=row.get('seed'), control=row.get('control', 0),
                representation=row['representation'], old_ci95=row['ci95'],
                old_valid_bootstraps=row['valid_bootstraps'], revised=new))
        for v in sorted({r['variable'] for r in scores['decoding']}):
            y = baseline[f'{v}_truth']
            if np.issubdtype(y.dtype, np.integer):
                labels, counts = np.unique(y, return_counts=True)
                support.append(dict(eid=eid, variable=v, classes=labels.tolist(), counts=counts.tolist()))
        stability.extend(dict(eid=eid, **r) for r in read(directory/'stability.json'))
        profiles.append(dict(eid=eid, **read(directory/'profile.json')))
    write_json(output/'revised_intervals.json', changes)
    write_json(output/'diagnostics_summary.json', diagnostics)
    write_json(output/'stability.json', stability)
    write_json(output/'class_support.json', support)
    write_json(output/'runtime.json', profiles)
    write_json(output/'review_complete.json', dict(status='complete', source=str(source.resolve()),
        source_sha256=provenance, sessions=len(profiles), skipped_sessions=skipped, rows=len(changes),
        review_code_sha256=sha256(Path(__file__)),
        evaluation_code_sha256=sha256(Path(__file__).resolve().parents[1]/'xcebra_ibl/experiment/evaluation.py'),
        interpretation='Evaluation-only revision; point scores and dimension selections preserved. '
                       'Intervals condition on all reference target classes being present. '
                       'Training loss summaries do not establish convergence.'))
    print(f'Review complete: {len(changes)} unchanged point estimates; {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review(args.input, args.output)
