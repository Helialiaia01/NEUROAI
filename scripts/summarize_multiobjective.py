"""Validate and summarize a completed synthetic multiobjective benchmark."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from xcebra_ibl.experiment.artifacts import write_json, sha256


def summarize(root):
    design = json.loads((root/'design.json').read_text())
    report = json.loads((root/'report.json').read_text())
    assert report['status'] == 'complete'
    rows = report['results']
    expected = {(graph, seed, family, latent) for graph in design['graphs']
                for seed in design['seeds'] for family in ('adaptation', 'multiobjective')
                for latent in range(2)}
    assert len(rows) == len(expected)
    assert {(r['graph'], r['seed'], r['family'], r['latent']) for r in rows} == expected
    summaries = []
    fig, axes = plt.subplots(1, len(design['graphs']), figsize=(9, 4), squeeze=False)
    for ax, graph in zip(axes[0], design['graphs']):
        for index, family in enumerate(('adaptation', 'multiobjective')):
            group = [r for r in rows if r['graph'] == graph and r['family'] == family]
            metrics = {}
            for metric in ('auroc', 'average_precision', 'validation_r2', 'test_r2'):
                values = np.array([r[metric] for r in group])
                assert np.isfinite(values).all()
                metrics[metric] = dict(mean=float(values.mean()), minimum=float(values.min()), maximum=float(values.max()))
            ranks = []
            for latent in range(2):
                values = [np.load(root/f'{graph}_{seed}_{family}'/f'attribution_{latent}.npy') for seed in design['seeds']]
                assert all(v.shape == (12,) and np.isfinite(v).all() for v in values)
                for i, left in enumerate(values):
                    for right in values[i+1:]:
                        correlation = float(spearmanr(left, right).statistic)
                        ranks.append(correlation if np.isfinite(correlation) else None)
            summaries.append(dict(graph=graph, family=family, metrics=metrics, cross_seed_rank_rho=ranks))
            ax.scatter(index+np.linspace(-.12, .12, len(group)), [r['auroc'] for r in group], s=45)
        ax.axhline(.5, color='gray', linestyle='--', label='chance AUROC')
        ax.set(xticks=[0, 1], xticklabels=['Adaptation', 'Multiobjective'], ylim=(0, 1.05), title=graph, ylabel='Known-support AUROC')
    fig.suptitle('Synthetic diagnostic: two latents × two training seeds\nOne data realization; unequal model families')
    fig.tight_layout()
    fig.savefig(root/'support_recovery.png', dpi=180)
    plt.close(fig)
    write_json(root/'summary.json', dict(status='verified', summaries=summaries,
        report_sha256=sha256(root/'report.json'), seconds=report['seconds'],
        artifact_sha256={str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
                         if p.is_file() and p.name != 'summary.json'}))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    summarize(parser.parse_args().input)
