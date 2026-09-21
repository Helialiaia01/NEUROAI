"""Bounded synthetic comparison; not an IBL runner or identifiability proof."""
import argparse
import copy
import random
import time
from pathlib import Path

import cebra
import numpy as np
import torch
import joblib
from scipy.signal import lfilter
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from cebra.data.datasets import DatasetxCEBRA
from cebra.data.multiobjective import ContrastiveMultiObjectiveLoader
from cebra.solver.multiobjective import MultiObjectiveConfig
from cebra.solver.schedulers import LinearRampUp
from cebra.models.jacobian_regularizer import JacobianReg

from xcebra_ibl.experiment.artifacts import write_json, sha256
from xcebra_ibl.experiment.evaluation import decode
from xcebra_ibl.models.trials import install_trial_safe_expander
from xcebra_ibl.models.xcebra_model import XCEBRAModel
from xcebra_ibl.models.randomness import seed_loader_generators


def generate(graph, data_seed=872):
    rng = np.random.default_rng(data_seed)
    weights = np.zeros((2, 12))
    if graph == 'anchor':
        weights[0, 0] = weights[1, 6] = 1
    elif graph == 'redundant':
        weights[0, :6] = weights[1, 6:] = np.linspace(.6, 1.4, 6)
    else:
        raise ValueError(graph)
    latent, neural = {}, {}
    for part, length in [('train', 1200), ('validation', 400), ('test', 400)]:
        # Separate continuous sequences, including independent burn-in.
        latent[part] = lfilter([1], [1, -.8], rng.normal(size=(length+100, 2)), axis=0)[100:]
    scale = latent['train'].std(0)
    for part, z in latent.items():
        latent[part] = z / scale
        neural[part] = np.tanh(latent[part] @ weights) + rng.normal(scale=.05, size=(len(z), 12))
    mean, std = neural['train'].mean(0), neural['train'].std(0)
    return {p: ((x-mean)/std).astype('float32') for p, x in neural.items()}, latent, weights


def joint_fit(x, z, steps):
    data = DatasetxCEBRA(torch.tensor(x), device='cpu',
                        latent0=torch.tensor(z[:, :1], dtype=torch.float32),
                        latent1=torch.tensor(z[:, 1:], dtype=torch.float32))
    loader = ContrastiveMultiObjectiveLoader(dataset=data, num_steps=steps, batch_size=64).to('cpu')
    config = MultiObjectiveConfig(loader)
    for i in range(3):
        config.set_slice(2*i if i < 2 else 0, 2*i+2 if i < 2 else 8)
        config.set_loss('FixedCosineInfoNCE', temperature=1.)
        if i < 2:
            config.set_distribution('time_delta', time_delta=1, label_name=f'latent{i}')
        else:
            config.set_distribution('time', time_offset=10)
        config.push()
    config.finalize()
    seed_loader_generators(loader, torch.initial_seed())
    net = cebra.models.init('offset10-model', num_neurons=12, num_units=32, num_output=8)
    data.configure_for(net)
    install_trial_safe_expander(data, np.zeros(len(x), dtype=int), len(x))
    # Official time positives can extend beyond the recording. Clamp those
    # centres at its boundary, while preserving valid behaviour-label centres.
    expand = data.expand_index
    data.expand_index = lambda index: expand(torch.as_tensor(index).clamp(0, len(x)-1))
    optimizer = torch.optim.Adam(list(net.parameters())+list(config.criterion.parameters()), lr=3e-4)
    solver = cebra.solver.init('multiobjective-solver', model=net,
        feature_ranges=config.feature_ranges, regularizer=JacobianReg(), renormalize=True,
        use_sam=False, criterion=config.criterion, optimizer=optimizer, tqdm_on=False).to('cpu')
    scheduler = LinearRampUp(n_splits=3, step_to_switch_on_reg=steps//4,
        step_to_switch_off_reg=steps//2, start_weight=0., end_weight=.01)
    solver.fit(loader=loader, scheduler_regularizer=scheduler)
    solver.model.set_split_outputs(False)
    return solver.model, {str(k): np.asarray(v).tolist() for k, v in solver.log.items()}


def windows(x, net):
    offset = net.get_offset()
    centers = np.arange(offset.left, len(x)-offset.right+1)
    indices = centers[:, None]+np.arange(-offset.left, offset.right)
    return torch.tensor(x[indices].transpose(0, 2, 1)), centers


def attribution(net, x):
    # Double precision avoids null radial directions from block normalizations
    # becoming artificial nonzero singular values. Invert full embedding first.
    net = copy.deepcopy(net).double().eval()
    batches = []
    for chunk in x[:256].split(32):
        chunk = chunk.double().requires_grad_(True)
        y = net(chunk).reshape(len(chunk), -1)
        jac = torch.stack([torch.autograd.grad(y[:, i].sum(), chunk, retain_graph=True)[0]
                           for i in range(y.shape[1])], dim=1).mean(-1).detach().numpy()
        batches.append(np.linalg.pinv(jac, rcond=1e-5))
    inverse = np.concatenate(batches)
    if not np.isfinite(inverse).all():
        raise ValueError('Nonfinite inverse')
    return inverse


def perturbation_check(net, group, prepared, targets, values, support, decoder):
    """Rank on validation only; freeze encoder/decoder for test mean-masking.

    These are model sensitivity checks, not biological interventions. Zero is
    the training neural mean. No decoder is refit after a group is masked.
    """
    k = int(np.count_nonzero(support))
    order = np.argsort(values, kind='stable')
    groups = dict(top=order[-k:], bottom=order[:k], connected=np.flatnonzero(support),
                  disconnected=np.flatnonzero(~support))
    rng = np.random.default_rng(4102)
    random_groups = [rng.choice(len(values), k, replace=False) for _ in range(99)]
    x = prepared['test'][0]

    def score(indices):
        masked = x.clone()
        masked[:, indices, :] = 0
        with torch.no_grad():
            embedding = torch.cat([net(b).reshape(len(b), -1)[:, group] for b in masked.split(128)]).numpy()
        return float(r2_score(targets['test'], decoder.predict(embedding)))

    baseline = score([])
    drops = {name: baseline-score(indices) for name, indices in groups.items()}
    random_drops = [baseline-score(indices) for indices in random_groups]
    return dict(baseline_r2=baseline, group_size=k,
        selected_neurons={name: indices.tolist() for name, indices in groups.items()},
        r2_drop=drops, random_r2_drops=random_drops,
        top_exceeds_random_median=bool(drops['top'] > np.median(random_drops)),
        top_exceeds_bottom=bool(drops['top'] > drops['bottom']),
        note='Validation-ranked, frozen-model test sensitivity to training-mean masking; not a causal or significance test.')


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started, rows = time.monotonic(), []
    write_json(args.output/'design.json', dict(iterations=args.iterations, seeds=args.seeds,
        graphs=args.graphs, data_seed=args.data_seed, perturbations=args.perturbations,
        cebra=cebra.__version__, torch=torch.__version__, source_sha256=sha256(Path(__file__)),
        caveat='Family comparison: two separate 2D encoders with constant regularization versus one 8D joint encoder with ramped regularization. Unequal capacity/objectives; no isolated causal claim. One fixed data seed, no IBL data.'))
    for graph in args.graphs:
        neural, latent, weights = generate(graph, args.data_seed)
        np.savez_compressed(args.output/f'{graph}_data.npz', weights=weights,
            **{f'x_{p}': x for p, x in neural.items()}, **{f'z_{p}': z for p, z in latent.items()})
        for seed in args.seeds:
            for family in ('adaptation', 'multiobjective'):
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                destination = args.output/f'{graph}_{seed}_{family}'
                destination.mkdir()
                print(f'Start {destination.name}', flush=True)
                if family == 'adaptation':
                    model = XCEBRAModel(max_iterations=args.iterations, batch_size=64,
                        num_hidden_units=32, embedding_dim_per_group=2, device='cpu',
                        random_seed=seed, jacobian_reg_weight=.01)
                    n = len(neural['train'])
                    model.fit_per_variable(neural['train'],
                        {v: latent['train'][:, i].astype('float32') for i, v in enumerate(('wheel', 'whisker_max'))},
                        np.zeros(n, dtype=int), np.arange(n), n, verbose=False)
                    model.save(destination)
                    nets = [model.models_[v].solver_.model for v in ('wheel', 'whisker_max')]
                    slices = [slice(0, 2)]*2
                    write_json(destination/'loss.json', model.training_diagnostics_)
                else:
                    net, log = joint_fit(neural['train'], latent['train'], args.iterations)
                    torch.save(net.state_dict(), destination/'model.pt')
                    write_json(destination/'loss.json', log)
                    nets, slices = [net]*2, [slice(0, 2), slice(2, 4)]
                for i, (net, group) in enumerate(zip(nets, slices)):
                    net.eval()
                    prepared = {p: windows(x, net) for p, x in neural.items()}
                    with torch.no_grad():
                        features = {p: torch.cat([net(b).reshape(len(b), -1)[:, group]
                            for b in x.split(128)]).numpy() for p, (x, _) in prepared.items()}
                    targets = {p: latent[p][c, i] for p, (_, c) in prepared.items()}
                    result = decode(features, targets, False, include_knn=False,
                                    save_dir=destination, prefix=f'decoder_{i}')['linear']
                    values = np.abs(attribution(net, prepared['test'][0])[:, :, group]).mean(axis=(0, 2))
                    np.save(destination/f'attribution_{i}.npy', values)
                    row = dict(graph=graph, seed=seed, family=family, latent=i,
                        auroc=float(roc_auc_score(weights[i] != 0, values)),
                        average_precision=float(average_precision_score(weights[i] != 0, values)),
                        validation_r2=result['validation_score'],
                        test_r2=float(r2_score(targets['test'], result['prediction'])))
                    if args.perturbations:
                        validation_values = np.abs(attribution(net, prepared['validation'][0])[:, :, group]).mean(axis=(0, 2))
                        np.save(destination/f'validation_attribution_{i}.npy', validation_values)
                        row['perturbation'] = perturbation_check(net, group, prepared, targets,
                            validation_values, weights[i] != 0,
                            joblib.load(destination/f'decoder_{i}_linear.joblib'))
                    rows.append(row)
                write_json(args.output/'progress.json', dict(results=rows, seconds=time.monotonic()-started))
                print(f'Completed {destination.name}: {time.monotonic()-started:.1f}s total', flush=True)
    write_json(args.output/'report.json', dict(status='complete', results=rows, seconds=time.monotonic()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--data-seed', type=int, default=872)
    parser.add_argument('--perturbations', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2025, 2026])
    parser.add_argument('--graphs', choices=['anchor', 'redundant'], nargs='+', default=['anchor', 'redundant'])
    args = parser.parse_args()
    if args.iterations < 4:
        parser.error('At least four iterations required for the regularizer ramp')
    run(args)
