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


if __name__ == '__main__':
    unittest.main()
