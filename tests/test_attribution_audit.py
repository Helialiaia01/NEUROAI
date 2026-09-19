import unittest
from types import SimpleNamespace

import numpy as np
import torch

from xcebra_ibl.models.xcebra_model import XCEBRAModel


def wrapped(net):
    return SimpleNamespace(solver_=SimpleNamespace(model=net))


class AttributionAuditTests(unittest.TestCase):
    def test_joint_inverse_is_selected_after_inversion(self):
        net = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            net.weight.copy_(torch.tensor([[1., 1.], [0., 1.]]))
        model = XCEBRAModel(model_architecture='linear')
        actual = model._jacobian_attribution(wrapped(net), np.ones((3, 2)), output_slice=(0, 1))
        np.testing.assert_allclose(actual, [1., 0.], atol=1e-7)

    def test_temporal_jacobian_matches_finite_differences(self):
        torch.manual_seed(13)
        net = torch.nn.Sequential(torch.nn.Conv1d(3, 2, 4), torch.nn.Tanh())
        data = np.random.default_rng(5).normal(size=(7, 3, 4)).astype('float32')
        # Independent central finite differences, one sample and input at a time.
        numerical = np.zeros((7, 2, 3, 4))
        for n in range(3):
            for t in range(4):
                hi, lo = data.copy(), data.copy()
                hi[:, n, t] += .001
                lo[:, n, t] -= .001
                with torch.no_grad():
                    numerical[:, :, n, t] = ((net(torch.from_numpy(hi)) - net(torch.from_numpy(lo))) / .002).squeeze(-1).numpy()
        expected = np.abs(np.linalg.pinv(numerical.mean(-1), rcond=1e-5)).mean(axis=(0, 2))
        model = XCEBRAModel()
        actual = model._jacobian_attribution(wrapped(net), data, batch_size=3)
        np.testing.assert_allclose(actual, expected, rtol=.003, atol=.001)

    def test_invalid_temporal_inputs_fail_instead_of_returning_scores(self):
        model = XCEBRAModel()
        net = wrapped(torch.nn.Conv1d(3, 2, 4))
        for data in (np.ones((8, 3)), np.ones((0, 3, 4)), np.full((2, 3, 4), np.nan)):
            with self.assertRaises(ValueError):
                model._jacobian_attribution(net, data)

    def test_output_rotation_changes_l1_inverse_summary(self):
        # The current L1 summary is not invariant to rotations of output axes.
        # Document this mathematical limitation; do not treat it as a solver bug.
        inverse = np.diag([1., 2.])
        angle = np.pi/4
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.assertFalse(np.allclose(np.abs(inverse).mean(1), np.abs(inverse @ rotation).mean(1)))
        np.testing.assert_allclose(np.linalg.norm(inverse, axis=1), np.linalg.norm(inverse @ rotation, axis=1))
