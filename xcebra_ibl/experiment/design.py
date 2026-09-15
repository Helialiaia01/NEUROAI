"""Explicit cohort and offline within-session inferential design."""
import json
from pathlib import Path
import numpy as np
from xcebra_ibl.configs.config import CORTICAL_AREAS
from xcebra_ibl.data.preprocess import _session_id


def block_ids(behavior):
    priors = np.array([float(str(b).split('_')[0]) for b in behavior])
    return np.cumsum(np.r_[0, priors[1:] != priors[:-1]]).astype(int)


def select_sources(args):
    files = {_session_id(p): p for p in sorted(args.data_dir.rglob('*.npz'))}
    if len(files) != len(list(args.data_dir.rglob('*.npz'))):
        raise ValueError('Duplicate raw session IDs')
    if args.cohort:
        cohort = json.loads(args.cohort.read_text())
        rows = cohort['sessions']
        if any(row.get('phase') not in {'exploratory','confirmatory'} for row in rows):
            raise ValueError('Cohort phases must be exploratory or confirmatory')
        if any(row.get('subject') is not None and not isinstance(row['subject'],str) for row in rows):
            raise ValueError('Subject IDs must be strings or null')
        if len({r['eid'] for r in rows}) != len(rows):
            raise ValueError('Cohort session IDs must be unique across phases')
        selected = [r['eid'] for r in rows if r['phase'] == args.phase]
        if args.session_ids:
            if not set(args.session_ids).issubset(selected):
                raise ValueError('Requested sessions do not belong to the selected cohort phase')
            selected = args.session_ids
        if args.phase == 'confirmatory' and len(args.dimensions) != 1:
            raise ValueError('Freeze a single dimension before confirmatory evaluation')
    else:
        if args.phase == 'confirmatory':
            raise ValueError('Confirmatory evaluation requires an explicit reserved cohort')
        selected = args.session_ids or sorted(files)[:args.max_sessions]
        rows = []
    missing = set(selected) - files.keys()
    if missing:
        raise FileNotFoundError(f'Missing selected sessions: {sorted(missing)}')
    if not selected:
        raise ValueError('No sessions selected')
    return [files[eid] for eid in selected], {row['eid']: row.get('subject') for row in rows}


def context_features(neural, masks, interior, left, right, directory):
    """Write each split in bounded chunks; never materialize the whole context tensor."""
    result = {}
    directory.mkdir(parents=True, exist_ok=True)
    offsets = np.arange(-left, right)
    for part, mask in masks.items():
        centers = np.flatnonzero(mask & interior)
        array = np.lib.format.open_memmap(directory / f'{part}.npy', mode='w+', dtype='float32',
                                         shape=(len(centers), neural.shape[1]*(left+right)))
        for start in range(0, len(centers), 256):
            chunk = centers[start:start+256]
            array[start:start+len(chunk)] = neural[chunk[:,None]+offsets].reshape(len(chunk), -1)
        array.flush()
        result[part] = array
    return result


def session_qc(session, splits, blocks, subject, cohort):
    counts = {}
    for var, label in session['label_arrays'].items():
        values = label.reshape(session['K'], session['T'])
        if np.issubdtype(values.dtype, np.integer):
            counts[var] = {part: {str(int(v)): int(n) for v,n in zip(*np.unique(values[idx,0], return_counts=True))}
                           for part,idx in splits.items()}
    return dict(eid=session['eid'], subject=subject, cohort=cohort,
                neurons=session['N'], retained_trials=session['K'], time_bins=session['T'],
                split_trials={p:len(idx) for p,idx in splits.items()}, categorical_trial_counts=counts,
                blocks=int(len(np.unique(blocks))),
                outcome_semantics='stimulus-side/choice agreement; not measured reward delivery',
                neural_units='reference export firing rate multiplied by 0.01 s to obtain counts',
                neural_scaling='training-trial mean/std separately per neuron and time bin',
                observation='offline; symmetric smoothing and temporal context',
                cortex_only=cohort=='cortical', cortical_areas=CORTICAL_AREAS if cohort=='cortical' else None)
