"""Audit saved synthetic models against CEBRA's independent Jacobian backend.

No training or IBL test-based method selection. Requires the deterministic models
written by scripts.validate_synthetic_recovery, including their report.json.
"""
import argparse
import importlib.util
from pathlib import Path
import json

import cebra
import numpy as np
import scipy.linalg
from scipy.signal import lfilter
from sklearn.metrics import roc_auc_score
import torch

from xcebra_ibl.models.xcebra_model import XCEBRAModel
from xcebra_ibl.experiment.evaluation import split_trials
from xcebra_ibl.experiment.artifacts import write_json, sha256


def audit(source, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Audit destination must be empty')
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source/'report.json').read_text())
    if metadata['shape'] != [40, 40, 12]:
        raise ValueError('Unsupported synthetic generator shape')
    weights = np.asarray(metadata['weights'])
    rng = np.random.default_rng(872)
    z = lfilter([1], [1, -.8], rng.normal(size=(40, 40, 2)), axis=1)
    z /= z.std(axis=(0, 1))
    neural = np.tanh(z @ weights) + rng.normal(scale=.05, size=(40, 40, 12))
    split = split_trials(40, 42)
    train = neural[split['train']]
    neural = ((neural-train.mean(axis=(0, 1)))/train.std(axis=(0, 1))).astype('float32').reshape(-1, 12)
    ids = np.repeat(np.arange(40), 40)
    times = np.tile(np.arange(40), 40)
    mask = np.isin(ids, split['test'])
    # Load only the official Jacobian backend. The attribution package's optional
    # cvxpy/Captum dependencies are not needed by this pure torch/numpy module.
    backend_path = Path(cebra.__file__).parent/'attribution/_jacobian.py'
    spec = importlib.util.spec_from_file_location('cebra_reference_jacobian', backend_path)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    rows, model_hashes = [], {}
    for directory in sorted(source.glob('seed*_reg*')):
        model = XCEBRAModel(device='cpu')
        model.load(directory)
        model._set_trial_structure(ids[mask], times[mask], 40)
        data = model._attribution_windows(neural[mask], 128)
        for vi, variable in enumerate(('wheel', 'whisker_max')):
            fitted = model.models_[variable]
            model_hashes[str(directory.name+'/'+f'xcebra_{variable}.pt')] = sha256(directory/f'xcebra_{variable}.pt')
            fitted.solver_.model.to('cpu').eval()
            x = torch.tensor(data, requires_grad=True)
            full = backend.compute_jacobian(fitted.solver_.model, [x], cuda_device='cpu')
            # Match float32 temporal reduction, then invert in float64 as in the runner.
            jac = full.mean(-1).astype('float64')
            inverse = np.stack([scipy.linalg.pinv(j, rtol=model.jacobian_pinv_rcond) for j in jac])
            reference = np.abs(inverse).mean(axis=(0, 2))
            actual = model._jacobian_attribution(fitted, data, batch_size=31)
            np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-5)
            singular = np.linalg.svd(jac, compute_uv=False)
            variants = {'inverse_l1_current': actual,
                        'inverse_l2_rotation_invariant': np.linalg.norm(inverse, axis=2).mean(0),
                        'forward_l2': np.linalg.norm(jac, axis=1).mean(0)}
            for cutoff in (1e-3, 1e-5, 1e-7):
                inv = np.linalg.pinv(jac, rcond=cutoff)
                variants[f'inverse_l1_rcond_{cutoff}'] = np.abs(inv).mean(axis=(0, 2))
            row = dict(model=directory.name, variable=variable,
                reference_max_absolute_error=float(np.max(np.abs(actual-reference))),
                median_effective_rank=float(np.median((singular > model.jacobian_pinv_rcond*singular[:, :1]).sum(1))),
                scores={name:dict(auroc=float(roc_auc_score(weights[vi] != 0, attr)),
                                  attribution=attr.tolist()) for name, attr in variants.items()})
            rows.append(row)
            print(f'{directory.name}/{variable}: reference agreement passed; rank {row["median_effective_rank"]}', flush=True)
    if len(rows) != len(metadata['results']):
        raise ValueError('Missing synthetic models')
    write_json(output/'audit.json', dict(status='complete', results=rows,
        cebra_version=cebra.__version__, reference_source=str(backend_path),
        reference_sha256=sha256(backend_path), model_sha256=model_hashes,
        synthetic_report_sha256=sha256(source/'report.json'),
        interpretation='Exploratory sensitivity audit on the same synthetic generator. '
                       'Variants are not independently validated replacements. '
                       'Numerical agreement does not establish biological identifiability.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.input, args.output)
