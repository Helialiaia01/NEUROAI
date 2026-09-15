"""Prepare portable per-session jobs and merge verified results; never submits jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from xcebra_ibl.experiment.artifacts import write_json, sha256, verified


def prepare(cohort_path, study_path, output, phase):
    cohort = json.loads(cohort_path.read_text())
    study = json.loads(study_path.read_text())
    forbidden = {'--output','--session-ids','--cohort','--phase','--data-dir'}
    if any(token.split('=',1)[0] in forbidden for token in study['arguments']):
        raise ValueError('Study arguments cannot override job paths or cohort selection')
    jobs = [{'eid':row['eid'], 'arguments': ['--session-ids',row['eid'],'--phase',phase]+study['arguments']}
            for row in cohort['sessions'] if row['phase']==phase]
    if len({j['eid'] for j in jobs}) != len(jobs) or not jobs:
        raise ValueError('Empty or duplicate job sessions')
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(cohort_sha256=sha256(cohort_path), study_sha256=sha256(study_path), jobs=jobs,
        rationale='Each session keeps its full candidate grid together: preprocessing/baselines run once and dimension selection uses all seeds.')
    path = output/'jobs.json'
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError('Existing job manifest differs; use a new directory')
    shutil.copy2(cohort_path,output/'cohort.json')
    shutil.copy2(study_path,output/'study.json')
    write_json(path,payload)
    return payload


def run(manifest, index, data_dir, root):
    payload=json.loads(manifest.read_text())
    if index<0 or index>=len(payload['jobs']):
        raise ValueError('Job index outside manifest')
    cohort=manifest.parent/'cohort.json'
    if sha256(cohort)!=payload['cohort_sha256']:
        raise ValueError('Cohort changed after job preparation')
    job=payload['jobs'][index]
    argv=['--data-dir',str(data_dir),'--cohort',str(cohort),'--output',str(root/job['eid']),*job['arguments']]
    destination=root/job['eid']
    destination.mkdir(parents=True,exist_ok=True)
    context = dict(job_manifest_sha256=sha256(manifest), eid=job['eid'], arguments=job['arguments'])
    context_path = destination/'job_context.json'
    if context_path.exists() and json.loads(context_path.read_text()) != context:
        raise ValueError('Output belongs to a different planned job')
    write_json(context_path, context)
    with (destination/'worker.log').open('a') as log:
        subprocess.run([sys.executable,'-m','xcebra_ibl.experiments',*argv],stdout=log,stderr=subprocess.STDOUT,check=True)


def merge(manifest, root, output):
    from xcebra_ibl.experiments import summarize
    payload=json.loads(manifest.read_text())
    planned={j['eid'] for j in payload['jobs']}
    if output.exists() and any(output.iterdir()):
        raise ValueError('Merge destination must be empty')
    sources=[]
    skipped=[]
    signatures=[]
    provenance={}
    for eid in sorted(planned):
        worker=root/eid
        manifest_data=json.loads((worker/'manifest.json').read_text())
        context=json.loads((worker/'job_context.json').read_text())
        expected=next(j for j in payload['jobs'] if j['eid']==eid)
        if context != dict(job_manifest_sha256=sha256(manifest), eid=eid, arguments=expected['arguments']):
            raise ValueError('Worker does not match the prepared job manifest')
        provenance[eid]=manifest_data
        if manifest_data['config']['cohort_sha256'] != payload['cohort_sha256']:
            raise ValueError(f'Worker {eid} has the wrong cohort')
        config=dict(manifest_data['config'])
        for key in ('output','session_ids','data_dir','cohort'):
            config.pop(key,None)
        signatures.append((config,manifest_data['code_sha256'],manifest_data['packages']))
        result=verified(worker/eid,'session_complete.json')
        if not result or result.get('status') not in {'complete','skipped'}:
            raise ValueError(f'Incomplete worker {eid}')
        if result['status']=='skipped':
            skipped.append(dict(eid=eid, **json.loads((worker/eid/'skipped.json').read_text())))
        sources.append((eid,worker/eid))
    if any(signature!=signatures[0] for signature in signatures):
        raise ValueError('Workers used inconsistent code, configuration or packages')
    output.mkdir(parents=True,exist_ok=True)
    for eid,source in sources:
        shutil.copytree(source,output/eid)
    summarize(output,draws=signatures[0][0]['bootstrap'],seed=signatures[0][0]['split_seed'])
    write_json(output/'worker_manifests.json',provenance)
    write_json(output/'merge_manifest.json',dict(job_manifest_sha256=sha256(manifest),sessions=sorted(planned),
        common_code_sha256=signatures[0][1],packages=signatures[0][2], skipped_sessions=skipped,
        analyzed_sessions=len(planned)-len(skipped)))


def dispatch(manifest, data_dir, root, gpus):
    """Keep one independent session worker running on each selected GPU."""
    payload = json.loads(manifest.read_text())
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError('GPU identifiers must be a nonempty unique list')
    for job in payload['jobs']:
        arguments = job['arguments']
        explicit_cuda = any(
            token == '--device=cuda' or
            (token == '--device' and i + 1 < len(arguments) and arguments[i + 1] == 'cuda')
            for i, token in enumerate(arguments)
        )
        if not explicit_cuda:
            raise ValueError('Two-GPU dispatch requires explicit --device cuda in the study')

    pending = deque(range(len(payload['jobs'])))
    lock = threading.Lock()
    stopped = threading.Event()
    failures = []

    def consume(gpu):
        while not stopped.is_set():
            with lock:
                if not pending:
                    return
                index = pending.popleft()
            eid = payload['jobs'][index]['eid']
            print(f'Dispatching job {index} ({eid}) to {gpu}', flush=True)
            environment = os.environ.copy()
            environment['CUDA_VISIBLE_DEVICES'] = gpu
            command = [sys.executable, '-m', 'xcebra_ibl.jobs', 'run',
                       '--manifest', str(manifest), '--index', str(index),
                       '--data-dir', str(data_dir), '--root', str(root)]
            try:
                subprocess.run(command, env=environment, check=True)
            except BaseException as exc:
                stopped.set()
                with lock:
                    failures.append((index, eid, gpu, exc))
                return

    with ThreadPoolExecutor(max_workers=len(gpus), thread_name_prefix='gpu-worker') as pool:
        futures = [pool.submit(consume, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    if failures:
        index, eid, gpu, exc = failures[0]
        raise RuntimeError(f'Job {index} ({eid}) failed on {gpu}; no new jobs were scheduled') from exc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    p=commands.add_parser('prepare')
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['exploratory','confirmatory'],default='exploratory')
    for action in ('run','merge','dispatch'):
        p=commands.add_parser(action)
        p.add_argument('--manifest',type=Path,required=True)
        p.add_argument('--root',type=Path,required=True)
        if action in {'run','dispatch'}:
            p.add_argument('--data-dir',type=Path,required=True)
        if action=='run':
            p.add_argument('--index',type=int,required=True)
        elif action=='merge':
            p.add_argument('--output',type=Path,required=True)
        else:
            p.add_argument('--gpus',nargs='+',required=True,
                           help='GPU UUIDs assigned to independent workers')
    args=parser.parse_args()
    if args.action=='prepare':
        result=prepare(args.cohort,args.study,args.output,args.phase)
        print(f"Prepared {len(result['jobs'])} session jobs; nothing submitted")
    elif args.action=='run':
        run(args.manifest,args.index,args.data_dir,args.root)
    elif args.action=='merge':
        merge(args.manifest,args.root,args.output)
    else:
        dispatch(args.manifest,args.data_dir,args.root,args.gpus)


if __name__=='__main__':
    main()
