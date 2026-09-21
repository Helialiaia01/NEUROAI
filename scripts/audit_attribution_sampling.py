"""Separate model-seed variability from attribution sampling, without training."""
import argparse
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

import cebra
import numpy as np
from scipy.stats import spearmanr
import torch
from xcebra_ibl.models.xcebra_model import XCEBRAModel
from xcebra_ibl.experiment.artifacts import sha256, write_json


def audit(models, inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((models/'session_complete.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Source session incomplete')
    with np.load(inputs) as data:
        neural, ids, times, length = (data[k] for k in ('neural','trial_ids','time_ids','trial_length'))
    seeds = (2025, 2026, 2027)
    values, centers, model_hashes, diagnostics = {}, {}, {}, {}
    schedules = ((256,42),(256,43),(256,44),(1024,42),(1024,43))
    variables = None
    for seed in seeds:
        directory = models/f'seed{seed}_dim4_observed'
        files = sorted(directory.glob('xcebra_*.pt'))
        names = [f.stem[len('xcebra_'):] for f in files]
        if variables is not None and names != variables:
            raise ValueError('Model variables differ between seeds')
        variables = names
        if not names:
            raise ValueError('No saved encoders found')
        for path, variable in zip(files, names):
            digest = sha256(path)
            relative = str(path.relative_to(models))
            if digest != manifest['artifacts'][relative]:
                raise ValueError(f'Model checksum mismatch: {path}')
            model_hashes[relative] = digest
            # These are user-generated trusted checkpoints. Load encoder weights
            # onto CPU directly; CEBRA 0.6 estimator loading restores CUDA device
            # metadata even when torch.load(map_location='cpu') is requested.
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            args, state = checkpoint['args'], checkpoint['state']
            net = cebra.models.init(args['model_architecture'], num_neurons=state['n_features_in_'],
                                    num_units=args['num_hidden_units'],num_output=args['output_dimension'])
            net.load_state_dict(checkpoint['state_dict']['model'], strict=True)
            net.eval()
            model = XCEBRAModel(model_architecture=args['model_architecture'],random_seed=seed)
            fitted = SimpleNamespace(solver_=SimpleNamespace(model=net))
            model.models_ = {variable:fitted}
            model._set_trial_structure(ids, times, int(length))
            for count, sample_seed in (*schedules, (256,seed)):
                model.attribution_seed = sample_seed
                windows = model._attribution_windows(neural, count)
                key = f'{seed}_{variable}_{count}_{sample_seed}'
                values[key] = model._jacobian_attribution(fitted,windows,batch_size=64)
                diagnostics[key] = dict(model.last_attribution_diagnostics_)
                centers[key] = model.attribution_centers_.copy()
            print(f'Completed seed {seed}, {variable}', flush=True)
    def rho(a,b):
        value = float(spearmanr(a,b).statistic)
        return value if np.isfinite(value) else None
    rows = []
    for variable in variables:
        row = dict(variable=variable,between_models={},within_model_sampling={})
        for count, sample_seed in (*schedules,(256,'legacy')):
            pairs = []
            for a,b in combinations(seeds,2):
                sa,sb = (a,b) if sample_seed=='legacy' else (sample_seed,sample_seed)
                ka,kb = f'{a}_{variable}_{count}_{sa}',f'{b}_{variable}_{count}_{sb}'
                if sample_seed!='legacy':
                    np.testing.assert_array_equal(centers[ka],centers[kb])
                pairs.append(rho(values[ka],values[kb]))
            row['between_models'][f'{count}_{sample_seed}'] = pairs
        for count, sample_seeds in ((256,(42,43,44)),(1024,(42,43))):
            row['within_model_sampling'][str(count)] = [rho(values[f'{seed}_{variable}_{count}_{a}'],
                values[f'{seed}_{variable}_{count}_{b}']) for seed in seeds for a,b in combinations(sample_seeds,2)]
        rows.append(row)
    np.savez_compressed(output/'attributions.npz',**values)
    np.savez_compressed(output/'sample_centers.npz',**centers)
    write_json(output/'sampling_audit.json',dict(status='complete',rows=rows,model_sha256=model_hashes,
        diagnostics=diagnostics,
        inputs_sha256=sha256(inputs),model_code_sha256=sha256(Path(__file__).resolve().parents[1]/'xcebra_ibl/models/xcebra_model.py'),
        interpretation='Paired sample-set diagnostic on saved encoders. No fitting or model selection. '
                       'Same-model repeatability measures sampling precision, not biological validity.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    audit(args.models,args.inputs,args.output)
