"""Diagnostic of learned attribution on a known sparse generative graph.

Run from the repository root with python -m scripts.validate_synthetic_recovery.
This is a diagnostic, not a claim of canonical xCEBRA identifiability.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.signal import lfilter
from sklearn.metrics import roc_auc_score, average_precision_score
from xcebra_ibl.models.xcebra_model import XCEBRAModel
from xcebra_ibl.experiment.evaluation import split_trials, decode
from xcebra_ibl.experiment.artifacts import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--device',default='cuda_if_available')
    args=parser.parse_args()
    started=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(872)
    K,T,N=40,40,12
    z=lfilter([1],[1,-.8],rng.normal(size=(K,T,2)),axis=1)
    z=z/np.std(z,axis=(0,1))
    # Each latent drives exactly six neurons; the other six are disconnected.
    weights=np.zeros((2,N))
    weights[0,:6]=np.linspace(.6,1.4,6)
    weights[1,6:]=np.linspace(.6,1.4,6)
    neural=np.tanh(z @ weights)+rng.normal(scale=.05,size=(K,T,N))
    split=split_trials(K,42)
    mu=neural[split['train']].mean(axis=(0,1));std=neural[split['train']].std(axis=(0,1))
    neural=((neural-mu)/std).astype('float32').reshape(-1,N)
    ids=np.repeat(np.arange(K),T);times=np.tile(np.arange(T),K)
    masks={p:np.isin(ids,i) for p,i in split.items()}
    labels={v:z[:,:,i].reshape(-1).astype('float32') for i,v in enumerate(['wheel','whisker_max'])}
    rows=[]
    for seed in (2025,2026):
        for regularization in (0.,.01):
            destination=args.output/f'seed{seed}_reg{regularization}'
            model=XCEBRAModel(max_iterations=args.iterations,batch_size=64,num_hidden_units=32,
                embedding_dim_per_group=2,device=args.device,random_seed=seed,jacobian_reg_weight=regularization)
            model.fit_per_variable(neural[masks['train']],{v:y[masks['train']] for v,y in labels.items()},
                                   ids[masks['train']],times[masks['train']],T,verbose=False)
            model.save(destination)
            write_json(destination/'diagnostics.json',model.training_diagnostics_)
            embeddings={p:model.transform_per_variable(neural[m],ids[m],times[m],T) for p,m in masks.items()}
            attributions=model.compute_attribution_maps(neural[masks['test']],n_samples=128,
                trial_ids=ids[masks['test']],time_ids=times[masks['test']],trial_length=T)
            np.savez_compressed(destination/'attributions.npz',**attributions)
            for i,v in enumerate(labels):
                interior={p:(times[m]>=5)&(times[m]<T-4) for p,m in masks.items()}
                features={p:embeddings[p][v][interior[p]] for p in masks}
                target={p:labels[v][masks[p]][interior[p]] for p in masks}
                results=decode(features,target,False)
                rows.append(dict(seed=seed,regularization=regularization,variable=v,
                    attribution_auroc=float(roc_auc_score(weights[i]!=0,attributions[v])),
                    attribution_average_precision=float(average_precision_score(weights[i]!=0,attributions[v])),
                    decoding_test_r2={family:float(1-np.sum((target['test']-r['prediction'])**2)/np.sum((target['test']-target['test'].mean())**2)) for family,r in results.items()}))
    write_json(args.output/'report.json',dict(iterations=args.iterations,requested_device=args.device,
        seconds=time.monotonic()-started,shape=[K,T,N],weights=weights.tolist(),results=rows,
        interpretation='Learned sparse-graph diagnostic on one toy nonlinear observation model. No IBL or general identifiability claim.'))


if __name__=='__main__':
    main()
