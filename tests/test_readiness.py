import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import joblib
import numpy as np
from scipy import signal

from xcebra_ibl.experiment.artifacts import complete, verified
from xcebra_ibl.experiment.evaluation import interval, decode, shuffle_trials
from xcebra_ibl.experiment.design import select_sources, context_features
from xcebra_ibl.analysis.pilot import bh_adjust, ClusterConfig, gaussian_test, compare_published
from xcebra_ibl.data.preprocess import _find_best_delay_by_cc
from xcebra_ibl.models.xcebra_model import XCEBRAModel


class ReadinessTests(unittest.TestCase):
    def test_automatic_device_preference_and_explicit_cuda_failure(self):
        from cebra.integrations.sklearn.utils import check_device
        for cuda, mps, expected in ((True,True,'cuda'),(False,True,'mps'),(False,False,'cpu')):
            with patch('torch.cuda.is_available',return_value=cuda), patch('cebra.helper._is_mps_availabe',return_value=mps):
                self.assertEqual(check_device(XCEBRAModel().device),expected)
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaises(ValueError):
                check_device('cuda')

    def test_corrupted_or_missing_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'a.json').write_text('{}')
            complete(path, {'status':'complete'})
            self.assertEqual(verified(path)['status'],'complete')
            (path/'a.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                verified(path)
            (path/'a.json').unlink()
            with self.assertRaises(ValueError):
                verified(path)

    def test_paired_bootstrap_uses_same_observations(self):
        y=np.arange(20,dtype=float)
        result=interval(y,y,np.repeat(np.arange(5),4),False,50,0,baseline=np.zeros_like(y))
        self.assertEqual(result['score'],1.)
        self.assertGreater(result['improvement'],0.)
        self.assertGreater(result['improvement_ci95'][0],0.)

    def test_group_permutation_preserves_block_and_trajectories(self):
        block=np.repeat([0,1],5)
        labels={'block':np.repeat(block,3),'wheel':np.arange(30,dtype=float)}
        shuffled=shuffle_trials(labels,3,4,block)
        np.testing.assert_array_equal(shuffled['block'],labels['block'])
        np.testing.assert_array_equal(np.diff(shuffled['wheel'].reshape(-1,3)),np.ones((10,2)))

    def test_persisted_decoder_reproduces_predictions(self):
        rng=np.random.default_rng(4)
        features={p:rng.normal(size=(30,3)) for p in ('train','validation','test')}
        targets={p:x[:,0]*2 for p,x in features.items()}
        with tempfile.TemporaryDirectory() as directory:
            result=decode(features,targets,False,False,Path(directory),'wheel')
            loaded=joblib.load(Path(directory)/'wheel_linear.joblib')
            np.testing.assert_allclose(loaded.predict(features['test']),result['linear']['prediction'])

    def test_context_memmap_matches_window_order(self):
        x=np.arange(60).reshape(20,3)
        masks={'train':np.arange(20)<10,'test':np.arange(20)>=10}
        interior=np.tile(np.array([False,True,True,True,True,True,True,True,True,False]),2)
        with tempfile.TemporaryDirectory() as directory:
            features=context_features(x,masks,interior,1,2,Path(directory))
            for part,mask in masks.items():
                expected=np.stack([x[i-1:i+2].ravel() for i in np.flatnonzero(mask & interior)])
                np.testing.assert_array_equal(features[part],expected)

    def test_reserved_cohort_and_fixed_dimension_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'data_a.npz').touch()
            (root/'data_b.npz').touch()
            cohort=root/'cohort.json'
            cohort.write_text(json.dumps({'sessions':[{'eid':'a','phase':'exploratory'},{'eid':'b','phase':'confirmatory'}]}))
            args=SimpleNamespace(data_dir=root,cohort=cohort,phase='exploratory',session_ids=['b'],dimensions=[2],max_sessions=3)
            with self.assertRaises(ValueError):
                select_sources(args)
            args.phase='confirmatory';args.dimensions=[2,4]
            with self.assertRaises(ValueError):
                select_sources(args)
            args.dimensions=[2]
            self.assertEqual(select_sources(args)[0],[root/'data_b.npz'])

    def test_optimized_alignment_matches_reference(self):
        rng=np.random.default_rng(9)
        behavior=rng.normal(size=(6,50))
        neural=rng.normal(size=(6,30,8))
        b=behavior-behavior.mean(axis=1,keepdims=True)
        n=neural-neural.mean(axis=1,keepdims=True)
        cc=np.array([np.mean([signal.correlate(b[k],n[k,:,i],mode='valid') for k in range(6)],axis=0) for i in range(8)])
        lag=int(np.linalg.norm(cc,axis=0).argmax())
        success=0<lag<20
        self.assertEqual(_find_best_delay_by_cc(behavior,neural),(lag if success else 10,success))

    def test_fdr_and_gaussian_test(self):
        np.testing.assert_allclose(bh_adjust([.01,.04,.03]),[.03,.04,.04])
        self.assertEqual(bh_adjust([]),[])
        rng=np.random.default_rng(2)
        X=np.r_[rng.normal(-10,.1,(20,2)),rng.normal(10,.1,(20,2))]
        result=gaussian_test(X,ClusterConfig(max_clusters=2,repeats=2,null_draws=3))
        self.assertEqual(result['k'],2)
        self.assertGreater(result['silhouette'],.9)
        self.assertGreaterEqual(result['p_value'],.25)

    def test_rrr_placeholder_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'rrr.json'
            p.write_text('version https://git-lfs.github.com/spec/v1\n')
            with self.assertRaises(ValueError):
                compare_published(None,p,[])

    def test_real_cebra_diagnostics_and_variable_recovery(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(120,6)).astype('float32')
        ids=np.repeat(np.arange(4),30);time=np.tile(np.arange(30),4)
        labels={'choice':np.repeat([0,1,0,1],30),'wheel':rng.normal(size=120).astype('float32')}
        with tempfile.TemporaryDirectory() as directory:
            params=dict(max_iterations=2,batch_size=16,num_hidden_units=16,
                        embedding_dim_per_group=2,device='cpu',random_seed=42,recovery_dir=directory)
            model=XCEBRAModel(**params)
            model.fit_per_variable(x,labels,ids,time,30,verbose=False)
            expected=model.transform_per_variable(x,ids,time,30)
            for variable in labels:
                diag=model.training_diagnostics_[variable]
                np.testing.assert_allclose(np.array(diag['loss'])+diag['reg_lambda'],diag['loss_reg'],rtol=1e-6)
                self.assertEqual(len(diag['gradient_norm']),2)
            recovered=XCEBRAModel(**params)
            with patch.object(recovered,'_fit_cebra',side_effect=AssertionError('Completed model retrained')):
                recovered.fit_per_variable(x,labels,ids,time,30,verbose=False)
            actual=recovered.transform_per_variable(x,ids,time,30)
            for variable in labels:
                np.testing.assert_allclose(actual[variable],expected[variable])



class AnalysisAndJobsTests(unittest.TestCase):
    def test_two_gpu_dispatch_uses_disjoint_workers_and_explicit_cuda(self):
        from xcebra_ibl.jobs import dispatch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=root/'jobs.json'
            jobs=[{'eid':f's{i}','arguments':['--device','cuda']} for i in range(3)]
            manifest.write_text(json.dumps({'jobs':jobs}))
            calls=[]
            import threading
            first_pair=threading.Barrier(2)
            def record(command,env,check):
                calls.append((command,env['CUDA_VISIBLE_DEVICES'],check))
                if len(calls) <= 2:
                    first_pair.wait(timeout=1)
            with patch('xcebra_ibl.jobs.subprocess.run',side_effect=record):
                dispatch(manifest,root/'data',root/'workers',['gpu-a','gpu-b'])
            self.assertEqual({int(c[0][c[0].index('--index')+1]) for c in calls},{0,1,2})
            self.assertEqual({c[1] for c in calls},{'gpu-a','gpu-b'})
            self.assertTrue(all(c[2] for c in calls))
            jobs[0]['arguments']=['--device','cuda_if_available']
            manifest.write_text(json.dumps({'jobs':jobs}))
            with self.assertRaises(ValueError):
                dispatch(manifest,root/'data',root/'workers',['gpu-a','gpu-b'])

    def test_excluded_session_is_explicit_and_integrity_checked(self):
        from xcebra_ibl.experiments import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';data.mkdir()
            np.savez(data/'data_excluded.npz',behavior=np.array(['0.2_0_0']*6))
            main(['--data-dir',str(data),'--output',str(root/'out'),'--device','cpu'])
            result=verified(root/'out'/'excluded','session_complete.json')
            self.assertEqual(result['status'],'skipped')
            summary=json.loads((root/'out'/'complete.json').read_text())
            self.assertEqual(summary['completed_sessions'],0)
            self.assertEqual(summary['skipped_sessions'][0]['reason'],'fewer_than_10_retained_trials')

    def test_published_rrr_intercept_is_last_and_ids_are_matched(self):
        import pandas as pd
        table=pd.DataFrame(dict(eid=['s']*4,uuids=['a','b','c','d'],acronym=['VISp']*4,xcebra_attr_block=[1,2,3,4]))
        rows=[]
        for i,u in enumerate(['d','a','c','b']):
            value={'a':1,'b':2,'c':3,'d':4}[u]
            beta=np.zeros((9,3));beta[0]=value;beta[-1]=100-value
            rows.append(dict(eid='s',uuids=u,acronym='VISp',RRRglobal_beta=beta.tolist(),RRRglobal_r2=.5,meanact_r2=.1))
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'rrr.json';p.write_text(json.dumps(rows))
            result=compare_published(table,p,['block'])
            self.assertAlmostEqual(result['variables']['block'],1.)
            self.assertEqual(result['matched_neurons'],4)

    def test_job_preparation_and_merge_fail_on_incomplete_worker(self):
        from xcebra_ibl.jobs import prepare,merge
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cohort=root/'cohort.json';study=root/'study.json'
            cohort.write_text(json.dumps({'sessions':[{'eid':'a','phase':'exploratory'}]}))
            study.write_text(json.dumps({'arguments':['--seeds','2025','--dimensions','2']}))
            payload=prepare(cohort,study,root/'plan','exploratory')
            self.assertEqual(len(payload['jobs']),1)
            with self.assertRaises(FileNotFoundError):
                merge(root/'plan'/'jobs.json',root/'workers',root/'merged')
            study.write_text(json.dumps({'arguments':['--output=/unexpected']}))
            with self.assertRaises(ValueError):
                prepare(cohort,study,root/'other','exploratory')

    def test_one_resampling_unit_has_no_confidence_interval(self):
        y=np.arange(10,dtype=float)
        result=interval(y,y,np.zeros(10),False,20,0,baseline=y*0)
        self.assertIsNone(result['ci95'])
        self.assertIsNone(result['improvement_ci95'])

class InterruptionAndModelStateTests(unittest.TestCase):
    def test_signal_records_interruption_and_restores_handler(self):
        import signal
        from xcebra_ibl.experiment.artifacts import interruption_status
        original=signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InterruptedError):
                with interruption_status(directory):
                    signal.raise_signal(signal.SIGTERM)
            self.assertTrue((Path(directory)/'interrupted.json').exists())
        self.assertEqual(signal.getsignal(signal.SIGTERM),original)

    def test_model_recovery_rejects_changed_data_and_padding_matches_transform(self):
        import torch
        rng=np.random.default_rng(73)
        x=rng.normal(size=(120,6)).astype('float32')
        ids=np.repeat(np.arange(4),30);time=np.tile(np.arange(30),4)
        labels={'wheel':rng.normal(size=120).astype('float32')}
        with tempfile.TemporaryDirectory() as directory:
            params=dict(max_iterations=2,batch_size=16,num_hidden_units=16,
                        embedding_dim_per_group=2,device='cpu',recovery_dir=directory)
            model=XCEBRAModel(**params)
            model.fit_per_variable(x,labels,ids,time,30,verbose=False)
            fitted=model.models_['wheel']
            from xcebra_ibl.models.trials import install_trial_safe_expander
            _,_,loader,_=fitted._prepare_fit(x,labels['wheel'][:,None])
            install_trial_safe_expander(loader.dataset,ids,30)
            windows=loader.dataset[torch.arange(120)]
            net=model.models_['wheel'].solver_.model
            net.eval()
            with torch.no_grad():
                direct=net(windows).numpy()
            transformed=model.transform_per_variable(x,ids,time,30)['wheel']
            np.testing.assert_allclose(direct,transformed,atol=1e-5)
            labels['wheel'][0]+=1
            with self.assertRaises(ValueError):
                XCEBRAModel(**params).fit_per_variable(x,labels,ids,time,30,verbose=False)

class WarningReportTests(unittest.TestCase):
    def test_warnings_are_preserved_on_failure(self):
        import warnings
        from xcebra_ibl.experiment.artifacts import warning_report
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                with warning_report(directory):
                    warnings.warn('diagnostic example',RuntimeWarning)
                    raise RuntimeError('test')
            payload=json.loads((Path(directory)/'warnings.json').read_text())
            self.assertEqual(payload['emitted_warning_count'],1)
            self.assertEqual(payload['warnings'][0]['message'],'diagnostic example')


if __name__ == '__main__':
    unittest.main()
