"""ConvRot 4-bit: packing, the quantizer's error, the forward/backward contract, and the LoRA base
interface.

Unlike the 8-bit path, this one needs no int8 tensor cores, so the whole layer — forward and gradient
— runs on CPU and is checked here rather than only on a GPU. The properties that matter are that
four-bit codes survive the byte packing IN THE RIGHT NIBBLE, that the per-group scale and zero point
are actually applied per group, that the rotation and the grouping each earn their place, and that
the custom op does not quietly keep a dequantized weight alive.
"""
import unittest

import torch
from torch import nn

from convrot_int4 import (GROUP_SIZE, ConvRotInt4Linear, dequantize_int4, eligible, pack_int4,
                          quantize_dit_int4, quantize_int4_groups, unpack_int4, weight_error)
from convrot_int8 import ConvRotInt8Linear, largest_pow4_divisor, rotate
from lora import DoRALinear, LoRALinear


def _outliered(rows=64, cols=256, seed=0):
    """A weight matrix shaped like a trained one: per-channel scale spread plus outlier channels.

    An iid draw is the one case where 4-bit quantization looks easy and the rotation looks useless,
    so testing on it would prove the opposite of what these tests are for.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g)
    w *= torch.exp(torch.randn(1, cols, generator=g) * 0.8)
    w[:, :4] *= 15.0
    return w


class PackingTests(unittest.TestCase):
    def test_every_representable_code_round_trips(self):
        # The whole grid at once, so a nibble bug cannot hide in an untested value.
        codes = torch.arange(0, 16, dtype=torch.uint8).repeat(4, 2)   # (4, 32), even column count
        self.assertTrue(torch.equal(unpack_int4(pack_int4(codes)), codes))

    def test_packing_halves_the_bytes(self):
        codes = torch.randint(0, 16, (16, 64), dtype=torch.uint8)
        packed = pack_int4(codes)
        self.assertEqual(packed.shape, (16, 32))
        self.assertEqual(packed.dtype, torch.uint8)

    def test_column_order_is_preserved(self):
        # Low nibble is the even column, high the odd; swapping them would still round-trip as a
        # pair while silently transposing adjacent weights.
        packed = pack_int4(torch.tensor([[1, 2, 3, 4]], dtype=torch.uint8))
        self.assertEqual(int(packed[0, 0]), 1 | (2 << 4))
        self.assertEqual(int(packed[0, 1]), 3 | (4 << 4))

    def test_odd_column_count_is_rejected(self):
        with self.assertRaises(ValueError):
            pack_int4(torch.zeros(2, 3, dtype=torch.uint8))


class QuantizerTests(unittest.TestCase):
    def test_codes_stay_inside_the_four_bit_grid(self):
        q, scales, mins = quantize_int4_groups(torch.randn(32, 128) * 10)
        self.assertGreaterEqual(int(q.min()), 0)
        self.assertLessEqual(int(q.max()), 15)

        self.assertEqual(scales.shape, (32, 128 // GROUP_SIZE))
        self.assertEqual(mins.shape, scales.shape)
        self.assertTrue(torch.all(scales > 0))

    def test_the_offset_places_the_levels_over_the_group_range(self):
        # An all-positive group must not waste half its levels on negatives: the group minimum gets
        # code 0 and the maximum code 15. An integer zero point clamped to [0, 15] fails exactly
        # here, saturating every code.
        x = torch.linspace(3.0, 4.0, GROUP_SIZE).unsqueeze(0)
        q, scales, mins = quantize_int4_groups(x)
        self.assertEqual(int(q[0, 0]), 0)
        self.assertEqual(int(q[0, -1]), 15)
        dq = dequantize_int4(pack_int4(q), scales, mins, GROUP_SIZE, torch.float32)
        self.assertLess(float((x - dq).abs().max()), (4.0 - 3.0) / 15)

    def test_scales_are_per_group_not_per_row(self):
        # Two groups with wildly different magnitudes: a per-row scale would crush the small one.
        x = torch.cat([torch.full((1, GROUP_SIZE), 1000.0), torch.full((1, GROUP_SIZE), 0.001)], 1)
        q, scales, mins = quantize_int4_groups(x)
        self.assertEqual(scales.shape, (1, 2))
        dq = dequantize_int4(pack_int4(q), scales, mins, GROUP_SIZE, torch.float32)
        # Reconstructed to the precision of the stored offset, not to fp32: the scale and offset are
        # held at META_DTYPE, which is what makes their per-group storage affordable.
        self.assertLess(abs(float(dq[0, -1]) - 0.001) / 0.001, 0.01)

    def test_a_zero_group_does_not_divide_by_zero(self):
        x = torch.randn(4, 2 * GROUP_SIZE)
        x[2, :GROUP_SIZE] = 0
        q, scales, mins = quantize_int4_groups(x)
        self.assertTrue(torch.isfinite(q.float()).all())
        self.assertTrue(torch.all(scales > 0))
        dq = dequantize_int4(pack_int4(q), scales, mins, GROUP_SIZE, torch.float32)
        self.assertTrue(torch.all(dq[2, :GROUP_SIZE] == 0))     # zero is exact in any float dtype

    def test_indivisible_width_is_rejected(self):
        with self.assertRaises(ValueError):
            quantize_int4_groups(torch.randn(2, GROUP_SIZE + 1))

    def test_grouping_beats_per_row_at_four_bits(self):
        # The measurement this module's design rests on: per-row 4-bit is not viable, grouping is.
        # The gap WIDENS with the layer's width, so a realistic width is used here — on a narrow
        # matrix a per-row scale still covers few enough columns to look acceptable.
        w = _outliered(cols=1024)

        def err(scales_per_row):
            g = w.shape[1] if scales_per_row else GROUP_SIZE
            q, s, lo = quantize_int4_groups(w, g)
            dq = dequantize_int4(pack_int4(q), s, lo, g, torch.float32)
            return float((w - dq).norm() / w.norm())

        self.assertLess(err(scales_per_row=False), 0.6 * err(scales_per_row=True))

    def test_rotation_lowers_the_error_on_top_of_grouping(self):
        # The rotation is not made redundant by grouping: it is what stops one outlier from setting
        # the scale for its whole group.
        w = _outliered(cols=256)
        rot = min(256, largest_pow4_divisor(256))

        def err(mat):
            q, s, lo = quantize_int4_groups(mat)
            dq = dequantize_int4(pack_int4(q), s, lo, GROUP_SIZE, torch.float32)
            return float((mat - dq).norm() / mat.norm())

        self.assertLess(err(rotate(w, rot)), err(w))


class ForwardTests(unittest.TestCase):
    def test_forward_matches_a_dequantized_reference(self):
        torch.manual_seed(0)
        lin = nn.Linear(256, 128)
        q = ConvRotInt4Linear(lin)
        x = torch.randn(8, 256)
        ref = torch.nn.functional.linear(rotate(x, q.rot_size), q.dequantize(), q.bias)
        self.assertTrue(torch.allclose(q(x), ref, atol=1e-5))

    def test_it_approximates_the_original_layer(self):
        # End to end: rotation cancels inside the matmul, so the only error left is the quantizer's.
        torch.manual_seed(0)
        lin = nn.Linear(512, 256)
        with torch.no_grad():
            lin.weight.copy_(_outliered(256, 512))
        x = torch.randn(16, 512)
        with torch.no_grad():
            rel = float((ConvRotInt4Linear(lin)(x) - lin(x)).norm() / lin(x).norm())
        self.assertLess(rel, 0.2)

    def test_the_input_gradient_flows_and_uses_the_dequantized_weight(self):
        # A frozen quantized base sits between trainable adapters, so the INPUT gradient is the whole
        # point; the straight-through estimate must equal grad @ dequant(W).
        torch.manual_seed(0)
        q = ConvRotInt4Linear(nn.Linear(128, 64))
        x = torch.randn(4, 128, requires_grad=True)
        q(x).sum().backward()
        expected = rotate(torch.ones(4, 64) @ q.dequantize(), q.rot_size)
        self.assertTrue(torch.allclose(x.grad, expected, atol=1e-4))

    def test_no_float_weight_is_retained_after_the_forward(self):
        # The saving would evaporate if autograd held a dequantized copy of every layer until
        # backward, which is exactly what a plain dequantize-then-F.linear forward would do.
        q = ConvRotInt4Linear(nn.Linear(128, 64))
        x = torch.randn(4, 128, requires_grad=True)
        saved = getattr(q(x).grad_fn, "saved_tensors", ())
        self.assertTrue(all(t.dtype in (torch.uint8, torch.int8) for t in saved),
                        [t.dtype for t in saved])


class ModuleInterfaceTests(unittest.TestCase):
    def _layer(self):
        torch.manual_seed(0)
        return ConvRotInt4Linear(nn.Linear(256, 128))

    def test_it_presents_the_linear_interface_an_adapter_needs(self):
        q = self._layer()
        self.assertTrue(q.is_quant_linear)
        self.assertEqual((q.in_features, q.out_features), (256, 128))
        self.assertEqual(q.weight.dtype, torch.uint8)          # device/dtype probe, packed codes
        self.assertEqual(q.qdata.shape, (128, 128))            # K/2 bytes per row

    def test_scales_survive_a_dtype_cast(self):
        # The fp32 scales are held as a uint8 view precisely so .to(bf16) cannot recast them; if it
        # did, every dequantized weight would silently change.
        q = self._layer()
        before = q.scales_u8.view(torch.float32).clone()
        q.to(torch.bfloat16)
        self.assertTrue(torch.equal(q.scales_u8.view(torch.float32), before))

    def test_dequantize_reconstructs_the_rotated_weight(self):
        torch.manual_seed(0)
        lin = nn.Linear(256, 128)
        q = ConvRotInt4Linear(lin)
        target = rotate(lin.weight.data.float(), q.rot_size)
        rel = float((q.dequantize() - target).norm() / target.norm())
        self.assertAlmostEqual(rel, weight_error(lin), places=6)

    def test_a_lora_adapter_wraps_it_and_dora_refuses_it(self):
        q = self._layer()
        LoRALinear(q, rank=4, alpha=4.0)                        # must not raise
        with self.assertRaises(RuntimeError):
            DoRALinear(q, rank=4, alpha=4.0)                    # needs a dequantized base weight

    def test_it_is_half_the_bytes_of_the_int8_path(self):
        lin = nn.Linear(256, 128)
        self.assertEqual(ConvRotInt4Linear(lin).qdata.numel() * 2,
                         ConvRotInt8Linear(lin).qdata.numel())

    def test_eligibility_matches_the_grouping_and_packing_constraints(self):
        self.assertTrue(eligible(nn.Linear(256, 128)))
        self.assertFalse(eligible(nn.Linear(GROUP_SIZE + 16, 128)))   # not divisible by the group

    def test_quantize_dit_replaces_eligible_block_linears_only(self):
        # The entry point the trainer calls: blocks are converted, everything outside them is left
        # alone (`first`/`last`/text stages are small and sensitive, and more so at four bits).
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = nn.Linear(256, 128)
                self.odd = nn.Linear(GROUP_SIZE + 16, 128)   # ineligible width

        class Dit(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block(), Block()])
                self.last = nn.Linear(256, 128)

        dit = Dit()
        self.assertEqual(quantize_dit_int4(dit), 2)
        self.assertIsInstance(dit.blocks[0].wq, ConvRotInt4Linear)
        self.assertIsInstance(dit.blocks[0].odd, nn.Linear)
        self.assertIsInstance(dit.last, nn.Linear)


if __name__ == "__main__":
    unittest.main()
