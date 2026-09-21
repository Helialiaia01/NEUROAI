import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
from xcebra_ibl.models.xcebra_model import XCEBRAModel


class AttributionSamplingTests(unittest.TestCase):
    def model(self, training_seed, attribution_seed=42):
        m = XCEBRAModel(model_architecture='linear', random_seed=training_seed,
                       attribution_seed=attribution_seed)
        m.models_['wheel'] = SimpleNamespace(solver_=SimpleNamespace(model=torch.nn.Linear(3, 2)))
        return m

    def test_training_seed_does_not_change_attribution_samples(self):
        data = np.arange(300).reshape(100, 3)
        a, b = self.model(2025), self.model(2026)
        np.testing.assert_array_equal(a._attribution_windows(data, 20), b._attribution_windows(data, 20))
        np.testing.assert_array_equal(a.attribution_centers_, b.attribution_centers_)
        self.assertEqual(len(np.unique(a.attribution_centers_)), 20)

    def test_sampling_seed_changes_samples_without_changing_training_seed(self):
        data = np.arange(300).reshape(100, 3)
        a, b = self.model(2025, 42), self.model(2025, 43)
        self.assertFalse(np.array_equal(a._attribution_windows(data, 20), b._attribution_windows(data, 20)))
        self.assertEqual(a.random_seed, b.random_seed)

    def test_old_checkpoint_retains_legacy_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'xcebra_meta.json'
            p.write_text(json.dumps({'random_seed': 2027}))
            m = XCEBRAModel()
            m.load(directory)
            self.assertEqual(m.attribution_seed, 2027)
            p.write_text(json.dumps({'random_seed': 2027, 'attribution_seed': 42}))
            m.load(directory)
            self.assertEqual(m.attribution_seed, 42)
