import hashlib
import struct
import unittest

import torch

from timestep_weights import EMPIRICAL_1000
from train_t2i import sample_timesteps
from training_utils import timestep_weight


class WeightedTimestepTests(unittest.TestCase):
    def test_embedded_table_is_exact(self):
        raw = struct.pack("<1000f", *EMPIRICAL_1000)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "485f5c69b93051c920b737f4ab87a90b9cc362c4501e422579ef6fedffc070fb",
        )
        self.assertAlmostEqual(sum(EMPIRICAL_1000) / 1000, 1.0, places=6)

    def test_lookup_orientation_matches_noise_fraction(self):
        t = torch.tensor([1.0, 0.5, 0.001])
        got = timestep_weight(t, "weighted")
        expected = torch.tensor([EMPIRICAL_1000[0], EMPIRICAL_1000[500], EMPIRICAL_1000[999]])
        self.assertTrue(torch.allclose(got, expected))

    def test_weighted_sampling_bypasses_dynamic_shift(self):
        def forbidden(_):
            raise AssertionError("dynamic schedule must not run")

        gen = torch.Generator().manual_seed(4)
        got = sample_timesteps(forbidden, 8, "cpu", gen, "weighted")
        self.assertTrue(torch.all(got >= 0.001))
        self.assertTrue(torch.all(got <= 1.0))

    def test_default_sampling_still_uses_dynamic_shift(self):
        got = sample_timesteps(lambda x: x * 0 + 0.25, 3, "cpu", weighting="uniform")
        self.assertTrue(torch.equal(got, torch.full((3,), 0.25)))


if __name__ == "__main__":
    unittest.main()
