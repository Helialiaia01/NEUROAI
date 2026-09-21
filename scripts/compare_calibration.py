"""Compare fixed-dimension pilot candidates using validation and saved stability."""
import argparse
import json
from pathlib import Path
import numpy as np
from xcebra_ibl.experiment.artifacts import sha256, write_json


def compare(old, new, output):
    output.mkdir(parents=True, exist_ok=False)
    def read(path):
        return json.loads(path.read_text())
    manifests = [read(p/'session_complete.json') for p in (old, new)]
    for p, manifest in zip((old, new), manifests):
        if manifest['status'] != 'complete':
            raise ValueError('Incomplete session')
        for name, digest in manifest['artifacts'].items():
            path = p/name
            if not path.resolve().is_relative_to(p.resolve()):
                raise ValueError('Invalid manifest path')
            if path.is_file() and sha256(path) != digest:
                raise ValueError(f'Corrupted input: {path}')
    if manifests[0]['artifacts']['preprocessing.npz'] != manifests[1]['artifacts']['preprocessing.npz']:
        raise ValueError('Preprocessing/splits differ')
    rows, curves, prefix_errors = [], [], []
    variables = list(read(old/'seed2025_dim4_observed/complete.json')['metrics'])
    for v in variables:
        row = dict(variable=v)
        for steps, p in ((500, old), (1000, new)):
            observed, null = [], []
            for seed in (2025, 2026, 2027):
                for control, scores in (('observed', observed), ('null1', null)):
                    d=p/f'seed{seed}_dim4_{control}'
                    scores.append(read(d/'complete.json')['metrics'][v]['linear']['validation_score'])
                    diag=read(d/'diagnostics.json')[v]
                    x=np.asarray(diag['loss_reg'])
                    if len(x)!=steps or not all(np.isfinite(a).all() for a in diag.values()):
                        raise ValueError('Invalid optimization diagnostics')
                    curves.append(dict(variable=v, seed=seed, control=control, steps=steps,
                        loss=x.tolist(), tail_mean=float(x[-100:].mean()),
                        previous_mean=float(x[-200:-100].mean()),
                        gradient_tail_mean=float(np.mean(diag['gradient_norm'][-100:]))))
                    if steps==1000:
                        before=np.asarray(read(old/f'seed{seed}_dim4_{control}'/'diagnostics.json')[v]['loss_reg'])
                        prefix_errors.append(float(np.max(np.abs(before-x[:500]))))
            stability=[r['attribution_spearman'] for r in read(p/'stability.json')
                       if r['dimension']==4 and r['variable']==v and r['attribution_spearman'] is not None]
            row[str(steps)]=dict(validation_mean=float(np.mean(observed)), validation_seeds=observed,
                null_validation_mean=float(np.mean(null)), null_validation_seeds=null,
                attribution_rho_median=float(np.median(stability)), attribution_rho_pairs=stability)
        rows.append(row)
    write_json(output/'comparison.json', dict(rows=rows, curves=curves,
        preprocessing_identical=True, maximum_loss_prefix_difference=max(prefix_errors),
        source_markers={str(p):sha256(p/'session_complete.json') for p in (old,new)},
        interpretation='One exploratory session, fixed dimension 4, matched seeds. Validation scores only; no test-based model selection. Loss tails are descriptive, not a formal convergence test.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,4,figsize=(13,6),layout='constrained')
    for ax,v in zip(axes.flat,variables):
        for seed,color in zip((2025,2026,2027),('C0','C1','C2')):
            x=np.asarray(next(r['loss'] for r in curves if r['variable']==v and r['seed']==seed and r['steps']==1000 and r['control']=='observed'))
            ax.plot(np.arange(25,1001,25),x.reshape(-1,25).mean(1),color=color,label=str(seed))
        ax.axvline(500,color='gray',linestyle='--'); ax.set_title(v); ax.set_xlabel('Step'); ax.set_ylabel('Total regularized loss')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('1,000-step training trajectories; 25-step means; observed labels')
    fig.savefig(output/'convergence.png',dpi=150); plt.close(fig)
    print(json.dumps(dict(rows=rows,maximum_loss_prefix_difference=max(prefix_errors)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    compare(a.old,a.new,a.output)
