import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from xcebra_ibl.data.preprocess import preprocess_session
from xcebra_ibl.experiments import split_trials, shuffle_trials
from xcebra_ibl.models.xcebra_model import XCEBRAModel


def fixture(path, count=110):
    rng = np.random.default_rng(19)
    n, t = 6, 35
    payload = dict(spike_count_matrix=rng.poisson(10, (count, t, n)).astype(float),
        behavior=np.array([f'{.2 if i % 2 else .8}_{1 if i % 2 else -1}_{.25 if i % 3 else 0}_{1 if i % 3 else -1}' for i in range(count)]),
        clusters_g=dict(label=np.ones(n), acronym=np.array(['VISp']*n), uuids=np.array([str(i) for i in range(n)])),
        wheel_vel=rng.normal(size=(count, t, 1)), licks=rng.normal(size=(count, t, 1)),
        whisker_motion=rng.normal(size=(count, t, 2)))
    np.savez(path, **payload)
    return payload


class ExperimentTests(unittest.TestCase):
    def test_split_and_shuffle(self):
        split = split_trials(50, 42)
        self.assertEqual(len(set(np.concatenate(list(split.values())))), 50)
        y = np.arange(150)
        shuffled = shuffle_trials({'a': y, 'b': y*2}, 3, 5)
        np.testing.assert_array_equal(shuffled['a']*2, shuffled['b'])
        np.testing.assert_array_equal(np.diff(shuffled['a'].reshape(-1, 3)), np.ones((50, 2)))

    def test_test_data_cannot_change_fitted_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data_fixture.npz'
            payload = fixture(path)
            train = split_trials(110, 42)['train']
            a = preprocess_session(path, fit_trials=train)
            heldout = np.setdiff1d(np.arange(110), train)
            payload['spike_count_matrix'][heldout] *= 100
            payload['wheel_vel'][heldout] += 1000
            np.savez(path, **payload)
            b = preprocess_session(path, fit_trials=train)
            np.testing.assert_array_equal(a['y_3d'][train], b['y_3d'][train])
            np.testing.assert_array_equal(a['X_3d'][train], b['X_3d'][train])
            self.assertEqual(a['metadata']['best_delays'], b['metadata']['best_delays'])

    def test_known_linear_inverse_and_disconnected_neuron(self):
        net = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            net.weight.copy_(torch.tensor([[2., 0., 0.], [0., 4., 0.]]))
        wrapped = SimpleNamespace(solver_=SimpleNamespace(model=net))
        model = XCEBRAModel(model_architecture='linear')
        attr = model._jacobian_attribution(wrapped, np.ones((7, 3)), batch_size=3)
        np.testing.assert_allclose(attr, [.25, .125, 0.])

    def test_attribution_independent_of_batch_partition(self):
        net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
        wrapped = SimpleNamespace(solver_=SimpleNamespace(model=net))
        model = XCEBRAModel(model_architecture='linear')
        data = np.random.default_rng(42).normal(size=(11, 3)).astype('float32')
        a = model._jacobian_attribution(wrapped, data, batch_size=4)
        b = model._jacobian_attribution(wrapped, data, batch_size=11)
        np.testing.assert_allclose(a, b, rtol=1e-5)



class TrialSamplerTests(unittest.TestCase):
    def prepare(self):
        from cebra import CEBRA
        from xcebra_ibl.models.trials import install_trial_safe_expander, install_trial_safe_differences
        self.ids = np.repeat(np.arange(4), 30)
        x = np.random.default_rng(0).normal(size=(120, 6)).astype('float32')
        labels = (np.arange(120) % 30 + self.ids*100).astype('float32')[:, None]
        model = CEBRA(model_architecture='offset10-model', output_dimension=4,
                      batch_size=16, max_iterations=1, time_offsets=10, device='cpu')
        _, _, self.loader, _ = model._prepare_fit(x, labels)
        install_trial_safe_expander(self.loader.dataset, self.ids, 30)
        install_trial_safe_differences(self.loader, self.ids)

    def test_differences_never_cross_trials(self):
        self.prepare()
        self.assertEqual(len(self.loader.distribution.time_difference), 80)
        np.testing.assert_allclose(self.loader.distribution.time_difference.numpy(), 10.)

    def test_padding_preserves_center_and_trial(self):
        self.prepare()
        left = int(self.loader.dataset.offset.left)
        for center in (0, 29, 30, 59, 60, 119):
            window = self.loader.dataset.expand_index(np.array([center])).numpy()[0]
            self.assertEqual(window[left], center)
            self.assertTrue(np.all(self.ids[window] == self.ids[center]))

    def test_discontiguous_and_short_trials_rejected(self):
        from xcebra_ibl.models.trials import validate_trials, install_trial_safe_expander
        with self.assertRaises(ValueError):
            validate_trials([0,0,1,1,0,0], 2)
        self.prepare()
        with self.assertRaises(ValueError):
            install_trial_safe_expander(self.loader.dataset, np.repeat(np.arange(60),2),2)


if __name__ == '__main__':
    unittest.main()
