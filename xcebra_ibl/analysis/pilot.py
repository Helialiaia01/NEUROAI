"""Offline, session-aware analysis of controlled pilot outputs.

This is an explicit analysis protocol, not an exact reproduction of published
RRR clustering. Gaussian-null rejection is distributional evidence, not causality.
"""
import argparse
from dataclasses import dataclass, asdict
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from xcebra_ibl.experiment.artifacts import verified, write_json, sha256


@dataclass
class ClusterConfig:
    min_neurons: int = 20
    max_clusters: int = 6
    repeats: int = 5
    min_ari: float = .8
    null_draws: int = 99
    seed: int = 2025


def standardize(profiles):
    profiles = np.asarray(profiles, dtype=float)
    if not np.isfinite(profiles).all():
        raise ValueError('Non-finite neuronal profile')
    return (profiles-profiles.mean(axis=0))/np.maximum(profiles.std(axis=0), 1e-8)


def best_clustering(profiles, config, seed):
    X = standardize(profiles)
    n = len(X)
    best = dict(silhouette=0., k=None, resampling_ari=None)
    for k in range(2, min(config.max_clusters, n//3)+1):
        if len(np.unique(X, axis=0)) < k:
            continue
        fitted = KMeans(n_clusters=k, n_init=10, random_state=seed+k).fit(X)
        if len(np.unique(fitted.labels_)) != k:
            continue
        rng = np.random.default_rng(seed+k)
        aris = []
        for repeat in range(config.repeats):
            subset = rng.choice(n, size=max(k+1, int(.8*n)), replace=False)
            alternate = KMeans(n_clusters=k, n_init=5, random_state=seed+repeat+k).fit(X[subset])
            aris.append(adjusted_rand_score(fitted.labels_, alternate.predict(X)))
        stability = float(np.median(aris))
        value = float(silhouette_score(X, fitted.labels_))
        if stability >= config.min_ari and value > best['silhouette']:
            best = dict(silhouette=value, k=k, resampling_ari=stability)
    return best


def gaussian_test(profiles, config):
    X = standardize(profiles)
    observed = best_clustering(X, config, config.seed)
    rng = np.random.default_rng(config.seed)
    # The full search/reproducibility filter is repeated on every Gaussian draw.
    covariance = np.atleast_2d(np.cov(X, rowvar=False))
    null = []
    for i in range(config.null_draws):
        null.append(best_clustering(rng.multivariate_normal(X.mean(axis=0), covariance, len(X)),
                                   config, config.seed+i+1)['silhouette'])
        if (i+1) % 10 == 0 or i+1 == config.null_draws:
            print(f'  Gaussian null draws: {i+1}/{config.null_draws}', flush=True)
    p = (1+sum(value >= observed['silhouette'] for value in null))/(1+len(null))
    return dict(observed, p_value=float(p), null_silhouettes=null, minimum_p=1/(config.null_draws+1))


def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    if not len(p):
        return []
    order = np.argsort(p)
    corrected = np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    result = np.empty_like(corrected)
    result[order] = np.minimum(corrected, 1.)
    return result.tolist()


def collect_session(directory):
    verified(directory, 'session_complete.json')
    scores = json.loads((directory/'scores.json').read_text())
    candidates = [(p.parent, json.loads(p.read_text())) for p in sorted(directory.glob('seed*/complete.json'))]
    variables = sorted({key.rsplit('_',1)[0] for key in scores['selected_dimensions'] if key.endswith('_linear')})
    if len(variables) < 2:
        raise ValueError('At least two non-degenerate variables with linear dimension selection are required')
    seeds = sorted({c['seed'] for _,c in candidates if c['control']==0})
    with np.load(directory/'preprocessing.npz', allow_pickle=True) as meta:
        areas = meta['acronym'].astype(str)
        raw_neurons = meta['retained_raw_neurons']
        uuids = meta['uuids'].astype(str)
        if len(uuids) != len(areas):
            uuids = np.array(['']*len(areas))
        order = meta['var_list'].astype(str).tolist()
    seed_profiles, raw_profiles = [], []
    for seed in seeds:
        columns = []
        for var in variables:
            dim = scores['selected_dimensions'][f'{var}_linear']
            matches = [d for d,c in candidates if c['seed']==seed and c['dimension']==dim and c['control']==0]
            if len(matches) != 1:
                raise ValueError(f'Ambiguous attribution source for {var}, seed {seed}')
            with np.load(matches[0]/'attributions.npz') as attrs:
                columns.append(attrs[var])
        # Normalize within session/seed to avoid arbitrary embedding-scale dominance.
        raw_profiles.append(np.column_stack(columns))
        seed_profiles.append(standardize(raw_profiles[-1]))
    seed_profiles = np.asarray(seed_profiles)
    profiles = seed_profiles.mean(axis=0)
    with np.load(directory/'encoding_coefficients.npz') as encoding:
        ridge_profiles = np.mean(np.abs(encoding['ridge'][:,[order.index(v) for v in variables],:]),axis=0).T
    table = pd.DataFrame(dict(eid=directory.name, ni=raw_neurons, uuids=uuids, acronym=areas))
    for i,var in enumerate(variables):
        table[f'xcebra_attr_{var}'] = np.mean(raw_profiles,axis=0)[:,i]
        table[f'profile_{var}'] = profiles[:,i]
        table[f'local_ridge_{var}'] = ridge_profiles[:,i]
    return table, variables, profiles, seed_profiles, ridge_profiles


def compare_published(table, path, variables):
    """Require genuine published columns and join by session + neuron UUID."""
    with path.open() as handle:
        prefix = handle.read(100)
    if prefix.startswith('version https://git-lfs.github.com/spec'):
        raise ValueError('RRR input is a Git-LFS pointer, not model results')
    try:
        reference = pd.read_json(path)
    except ValueError as exc:
        raise ValueError('RRR input is not a valid results table') from exc
    required = {'eid','uuids','acronym','RRRglobal_beta','RRRglobal_r2','meanact_r2'}
    if not required.issubset(reference):
        raise ValueError(f'Missing published RRR columns: {sorted(required-set(reference))}')
    if reference.duplicated(['eid','uuids']).any() or table.duplicated(['eid','uuids']).any():
        raise ValueError('Ambiguous neuron identities in RRR comparison')
    matched = table[table.uuids.ne('')].merge(reference, on=['eid','uuids'], suffixes=('', '_rrr'), validate='one_to_one')
    if matched.empty or not np.all(matched.acronym == matched.acronym_rrr):
        raise ValueError('No identity-matched neurons, or brain-area identity mismatch')
    # Keep the published performance filter explicit; no local fallback.
    matched = matched[(matched.RRRglobal_r2-matched.meanact_r2) > .015]
    from xcebra_ibl.configs.config import VARIABLE_NAMES
    beta = np.array(matched.RRRglobal_beta.tolist())
    # Reference schema is neuron x (variables + intercept) x time.
    if beta.ndim != 3 or beta.shape[1] != len(VARIABLE_NAMES)+1:
        raise ValueError('Expected published beta shape: neurons x 9 x time')
    coefficients = np.abs(beta[:,:-1,:]).mean(axis=2)
    def correlation(a,b):
        value = float(spearmanr(a,b).statistic)
        return value if np.isfinite(value) else None
    return dict(source='published_rrr', source_sha256=sha256(path), matched_neurons=len(matched),
        variables={v:correlation(matched[f'xcebra_attr_{v}'], coefficients[:,VARIABLE_NAMES.index(v)])
                   for v in variables})


def analyze(input_dir, output, config, rrr=None):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Analysis destination must be empty; preserve previous results')
    output.mkdir(parents=True, exist_ok=True)
    tables, tests = [], []
    for directory in sorted(input_dir.iterdir()):
        if not (directory/'session_complete.json').exists():
            continue
        if verified(directory, 'session_complete.json').get('status') == 'skipped':
            continue
        table, variables, profiles, seeds, ridge = collect_session(directory)
        print(f'Analyzing session {directory.name}: {len(table)} neurons', flush=True)
        tables.append(table)
        for area in sorted(table.acronym.unique()):
            mask = table.acronym.to_numpy()==area
            if mask.sum() < config.min_neurons:
                continue
            for method, values in [('regularized_cebra_adaptation', profiles), ('local_ridge',ridge)]:
                print(f'  Area {area}, {method}, {int(mask.sum())} neurons', flush=True)
                result = gaussian_test(values[mask], config)
                result.update(eid=directory.name, area=area, method=method, neurons=int(mask.sum()), variables=variables)
                if method=='regularized_cebra_adaptation' and result['k'] and len(seeds)>1:
                    assignments = [KMeans(n_clusters=result['k'], n_init=10, random_state=config.seed).fit_predict(standardize(s[mask])) for s in seeds]
                    result['cross_seed_ari'] = float(np.median([adjusted_rand_score(a,b) for a,b in combinations(assignments,2)]))
                else:
                    result['cross_seed_ari'] = None
                tests.append(result)
    if not tables:
        raise ValueError('No complete pilot sessions found')
    combined = pd.concat(tables,ignore_index=True)
    if combined.duplicated(['eid','ni']).any():
        raise ValueError('Duplicate session/neuron rows')
    combined.to_json(output/'neuron_profiles.json',orient='records',indent=2)
    numeric = [c for c in combined if c.startswith(('xcebra_attr_','profile_','local_ridge_'))]
    area = combined.groupby(['eid','acronym'])[numeric].mean().reset_index()
    area.to_json(output/'session_area_profiles.json',orient='records',indent=2)
    for result,q in zip(tests,bh_adjust([r['p_value'] for r in tests])):
        result['q_bh'] = q
        result['passes_distributional_test'] = q <= .05 and result['k'] is not None
        result['cross_seed_reproduced'] = result['cross_seed_ari'] is not None and result['cross_seed_ari'] >= config.min_ari
    write_json(output/'clustering.json',dict(config=asdict(config), tests=tests,
        correction_family='all session-area-method tests in this invocation',
        interpretation='Exploratory Gaussian-null tests, conditional on selected variables/dimensions. Gaussian rejection does not prove biological categories.'))
    if rrr:
        variables = [c[len('xcebra_attr_'):] for c in numeric if c.startswith('xcebra_attr_')]
        write_json(output/'published_rrr_comparison.json',compare_published(combined,rrr,variables))
    write_json(output/'analysis_complete.json', dict(status='complete', sessions=len(tables),
               tests=len(tests), config=asdict(config), input=str(input_dir.resolve())))
    print(f'Analysis complete: {len(tables)} sessions, {len(tests)} tests; {output}', flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--null-draws',type=int,default=99)
    p.add_argument('--min-neurons',type=int,default=20)
    p.add_argument('--rrr',type=Path)
    args=p.parse_args()
    if args.null_draws<1 or args.min_neurons<6:
        p.error('Positive null draws and at least six neurons required')
    analyze(args.input,args.output,ClusterConfig(null_draws=args.null_draws,min_neurons=args.min_neurons),args.rrr)


if __name__=='__main__':
    main()
