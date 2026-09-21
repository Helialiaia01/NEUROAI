"""Summarize the fixed fresh-data and frozen-decoder perturbation check."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from xcebra_ibl.experiment.artifacts import sha256, write_json
from scripts.summarize_multiobjective import summarize


def review(root):
    rows = []
    for seed in (873, 874):
        source = root/f'data{seed}'
        design = json.loads((source/'design.json').read_text())
        assert design['data_seed'] == seed and design['iterations'] == 1000
        assert design['seeds'] == [2025, 2026] and design['graphs'] == ['redundant']
        assert design['perturbations']
        assert design['source_sha256'] == sha256(root/'source.py')
        for path in source.glob('*/loss.json'):
            losses = json.loads(path.read_text())
            groups = losses.values() if 'adaptation' in path.parent.name else [losses]
            for group in groups:
                for values in group.values():
                    assert np.isfinite(np.asarray(values, dtype=float)).all(), path
        summarize(source)
        report = json.loads((source/'report.json').read_text())
        for row in report['results']:
            p = row['perturbation']
            np.testing.assert_allclose(p['baseline_r2'], row['test_r2'], rtol=1e-6, atol=1e-7)
            assert p['group_size'] == 6 and len(p['random_r2_drops']) == 99
            assert all(len(indices) == 6 for indices in p['selected_neurons'].values())
            assert np.isfinite(list(p['r2_drop'].values())+p['random_r2_drops']).all()
            row['data_seed'] = seed
            row['top_minus_random'] = p['r2_drop']['top']-float(np.median(p['random_r2_drops']))
            row['top_minus_bottom'] = p['r2_drop']['top']-p['r2_drop']['bottom']
            rows.append(row)
    summaries = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for index, family in enumerate(('adaptation', 'multiobjective')):
        group = [r for r in rows if r['family'] == family]
        assert len(group) == 8
        top_random = sum(r['top_minus_random'] > 0 for r in group)
        top_bottom = sum(r['top_minus_bottom'] > 0 for r in group)
        positive_control = sum(r['perturbation']['r2_drop']['connected'] >
                               r['perturbation']['r2_drop']['disconnected'] for r in group)
        summaries.append(dict(family=family, cases=8,
            support_auroc_mean=float(np.mean([r['auroc'] for r in group])),
            support_auroc_range=[min(r['auroc'] for r in group), max(r['auroc'] for r in group)],
            test_r2_mean=float(np.mean([r['test_r2'] for r in group])),
            top_beats_random_cases=top_random, top_beats_bottom_cases=top_bottom,
            connected_beats_disconnected_cases=positive_control,
            consistent_direction_check=top_random == top_bottom == 8))
        for ax, key in zip(axes, ('top_minus_random', 'top_minus_bottom')):
            ax.scatter(index+np.linspace(-.12, .12, 8), [r[key] for r in group])
    for ax, label in zip(axes, ('Top − random-median R² drop', 'Top − bottom R² drop')):
        ax.axhline(0, color='gray', linestyle='--')
        ax.set(xticks=[0, 1], xticklabels=['Adaptation', 'Multiobjective'], ylabel=label)
    fig.suptitle('Fresh synthetic data: frozen-model neuron masking\nPositive values support the attribution ranking; eight cases per family')
    fig.tight_layout()
    fig.savefig(root/'perturbation_checks.png', dpi=180)
    plt.close(fig)
    write_json(root/'review.json', dict(status='complete', summaries=summaries, results=rows,
        sources={f'data{s}': sha256(root/f'data{s}'/'report.json') for s in (873, 874)},
        limitations='Two new data realizations of one fixed graph; masking sensitivity is not biological causality or formal significance.'))
    print('DECISION SUMMARY\n'+json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    review(parser.parse_args().input)
