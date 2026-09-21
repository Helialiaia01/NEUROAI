import unittest

import numpy as np
import torch

from scripts.benchmark_multiobjective import attribution, generate, perturbation_check
from xcebra_ibl.models.randomness import seed_loader_generators
from cebra.distributions.continuous import TimedeltaDistribution
from types import SimpleNamespace


class BenchmarkTests(unittest.TestCase):
    def test_new_data_seed_changes_realization_not_support(self):
        x, _, w = generate('redundant', 873)
        y, _, v = generate('redundant', 874)
        np.testing.assert_array_equal(w, v)
        self.assertFalse(np.array_equal(x['train'], y['train']))

    def test_perturbation_known_linear_reliance(self):
        class Net(torch.nn.Module):
            def forward(self, x):
                return x.mean(-1)
        class Decoder:
            def predict(self, x):
                return x[:, 0]+x[:, 1]
        x = torch.tensor(np.random.default_rng(2).normal(size=(100, 4, 1)), dtype=torch.float32)
        original = x.clone()
        y = (x[:, 0, 0]+x[:, 1, 0]).numpy()
        result = perturbation_check(Net(), slice(None), {'test': (x, None)},
            {'test': y}, np.array([4., 3., 2., 1.]), np.array([True, True, False, False]), Decoder())
        self.assertAlmostEqual(result['baseline_r2'], 1.)
        self.assertAlmostEqual(result['r2_drop']['disconnected'], 0.)
        self.assertGreater(result['r2_drop']['connected'], .99)
        self.assertTrue(result['top_exceeds_bottom'])
        self.assertTrue(torch.equal(original, x))

    def test_private_generators_repeat_and_respond_to_seed(self):
        def sample(seed):
            distribution = TimedeltaDistribution(torch.arange(100.).reshape(-1, 1))
            loader = SimpleNamespace(distribution=distribution)
            self.assertEqual(seed_loader_generators(loader, seed), 2)
            return distribution.sample_prior(32)
        self.assertTrue(torch.equal(sample(42), sample(42)))
        self.assertFalse(torch.equal(sample(42), sample(43)))

    def test_generator_uses_train_scaling_and_independent_sequences(self):
        x, z, weights = generate('anchor')
        np.testing.assert_allclose(x['train'].mean(0), 0, atol=1e-6)
        np.testing.assert_allclose(x['train'].std(0), 1, atol=1e-6)
        self.assertEqual(int(np.count_nonzero(weights)), 2)
        self.assertFalse(np.array_equal(z['validation'], z['test']))
        self.assertGreater(np.abs(x['test'].mean(0)).max(), .01)

    def test_full_inverse_precedes_slice(self):
        class Linear(torch.nn.Module):
            def forward(self, x):
                return torch.stack((x[:, 0, 0]+x[:, 1, 0], x[:, 1, 0]), dim=1)
        inverse = attribution(Linear(), torch.zeros(3, 2, 1))
        expected = np.broadcast_to([[1., -1.], [0., 1.]], (3, 2, 2))
        np.testing.assert_allclose(inverse, expected, atol=1e-12)
        # Independently inverting only the first row would instead give [.5,.5].
        np.testing.assert_allclose(inverse[:, :, 0], [[1., 0.]]*3, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
