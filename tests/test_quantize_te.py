import unittest

import torch
from torch import nn

from quantize import Fp8Linear, quantize_frozen_linears_fp8


class TextEncoderQuantizationTests(unittest.TestCase):
    def test_frozen_linears_are_replaced(self):
        model = nn.Sequential(nn.Linear(8, 6), nn.GELU(), nn.Linear(6, 4)).eval()
        model.requires_grad_(False)
        count = quantize_frozen_linears_fp8(model)
        self.assertEqual(count, 2)
        self.assertIsInstance(model[0], Fp8Linear)
        self.assertIsInstance(model[2], Fp8Linear)
        out = model(torch.randn(3, 8, dtype=torch.bfloat16))
        self.assertEqual(out.shape, (3, 4))
        self.assertTrue(torch.isfinite(out).all())

    def test_trainable_linear_is_rejected(self):
        model = nn.Sequential(nn.Linear(4, 4))
        with self.assertRaises(ValueError):
            quantize_frozen_linears_fp8(model)


if __name__ == "__main__":
    unittest.main()
