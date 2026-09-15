"""Repeatable tiny CPU integration: workers, resume, provenance and merged analysis."""
import argparse
import json
from pathlib import Path
import runpy
import numpy as np
from xcebra_ibl.jobs import prepare, run, merge
from xcebra_ibl.analysis.pilot import analyze, ClusterConfig
from xcebra_ibl.experiment.artifacts import verified, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=args.output
    if root.exists() and any(root.iterdir()):
        raise ValueError('Verification output must be empty')
    (root/'data').mkdir(parents=True,exist_ok=True)
    fixture=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tests/test_experiments.py'))['fixture']
    for eid in ('fixture_a','fixture_b'):
        fixture(root/'data'/f'data_{eid}.npz')
    np.savez(root/'data'/'data_fixture_excluded.npz', behavior=np.array(['0.2_0_0']*6))
    cohort=root/'cohort.json';study=root/'study.json'
    write_json(cohort,{'sessions':[{'eid':eid,'phase':'exploratory','subject':'synthetic_mouse'} for eid in ('fixture_a','fixture_b','fixture_excluded')]})
    write_json(study,{'arguments':['--variables','wheel','choice','--seeds','2025','2026',
        '--dimensions','2','4','--iterations','3','--bootstrap','20','--batch-size','32',
        '--attribution-samples','11','--checkpoint-frequency','2','--device','cpu']})
    jobs=prepare(cohort,study,root/'plan','exploratory')
    for index in range(len(jobs['jobs'])):
        run(root/'plan'/'jobs.json',index,root/'data',root/'workers')
    run(root/'plan'/'jobs.json',0,root/'data',root/'workers')
    merge(root/'plan'/'jobs.json',root/'workers',root/'merged')
    rows=json.loads((root/'merged'/'summary.json').read_text())
    assert all(row['scores']['unit']=='subject' and row['scores']['n_units']==1 for row in rows)
    assert all(row['scores']['ci95'] is None for row in rows)
    for eid in ('fixture_a','fixture_b'):
        assert verified(root/'merged'/eid,'session_complete.json')
    assert verified(root/'merged'/'fixture_excluded','session_complete.json')['status']=='skipped'
    assert len(json.loads((root/'merged'/'merge_manifest.json').read_text())['skipped_sessions'])==1
    analyze(root/'merged',root/'analysis',ClusterConfig(min_neurons=6,null_draws=9))
    write_json(root/'verification.json',dict(status='passed',sessions=2,explicitly_excluded_sessions=1,encoder_fits=32,iterations_per_fit=3,
        checks=['workers','same-config resume','artifact integrity','manifest provenance','subject aggregation','explicit exclusions','offline analysis'],
        scientific_result=False))
    print('CPU integration passed.')


if __name__=='__main__':
    main()
