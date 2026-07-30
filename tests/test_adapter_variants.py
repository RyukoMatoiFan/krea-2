"""Variant-agnostic plumbing around the six adapter parameterisations.

``check_adapters.py`` already proves each variant is a no-op at init, learns, and round-trips. What
is tested here is everything the trainer wraps AROUND an adapter -- EMA naming, save metadata and
reconstructing the injection spec from a saved file -- because those paths were written for plain
LoRA's ``lora_A``/``lora_B`` and are the ones that break silently on a variant that has neither.
"""
import os
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from lora import (VARIANTS, adapter_tensors, ema_named_tensors, lora_spec_from_state, save_lora,
                  variant_of)


def _adapter(variant, in_f=32, out_f=32, rank=4):
    return VARIANTS[variant](nn.Linear(in_f, out_f), rank, float(rank))


class VariantPlumbingTests(unittest.TestCase):
    def test_ema_covers_every_trainable_tensor_of_every_variant(self):
        for variant in VARIANTS:
            adapters = {"blocks.0.attn.wq": _adapter(variant)}
            named = ema_named_tensors({"dit": adapters})
            expected = {f"dit.blocks.0.attn.wq.{p}" for p in adapter_tensors(adapters["blocks.0.attn.wq"])}
            self.assertEqual(set(named), expected, variant)
            # The EMA must hold the LIVE tensors, not copies, or it averages a frozen snapshot.
            for key, tensor in named.items():
                self.assertTrue(any(tensor is p for p in adapter_tensors(
                    adapters["blocks.0.attn.wq"]).values()), key)

    def test_plain_lora_ema_keys_are_unchanged(self):
        # Guards resume compatibility: an in-flight LoRA run's EMA shadow is keyed by these names.
        named = ema_named_tensors({"dit": {"blocks.0.attn.wq": _adapter("lora")}, "te": {}})
        self.assertEqual(sorted(named), ["dit.blocks.0.attn.wq.lora_A", "dit.blocks.0.attn.wq.lora_B"])

    def test_variant_of_distinguishes_subclasses(self):
        for variant in VARIANTS:
            self.assertEqual(variant_of(_adapter(variant)), variant)
            self.assertEqual(variant_of({"a": _adapter(variant)}), variant)

    def test_saved_file_declares_its_variant_and_round_trips_the_spec(self):
        for variant in VARIANTS:
            adapters = {"blocks.0.attn.wq": _adapter(variant),
                        "txtfusion.refiner_blocks.0.mlp.up": _adapter(variant),
                        "txtmlp.0": _adapter(variant)}
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "a.safetensors")
                save_lora(adapters, path, rank=4, alpha=8, metadata={"variant": "WRONG"})
                sd = load_file(path)
                with safe_open(path, framework="pt") as f:
                    meta = f.metadata()
            # Derived from the modules, so a stale caller-supplied variant cannot mislabel the file.
            self.assertEqual(meta["variant"], variant)
            spec = lora_spec_from_state(sd, meta)
            self.assertEqual(spec["variant"], variant)
            self.assertEqual(spec["rank"], 4)
            self.assertEqual(spec["alpha"], 8.0)      # alpha != rank must survive: it scales the adapter
            self.assertTrue(spec["include_txtfusion"])
            self.assertTrue(spec["include_txtmlp"])

    def test_spec_infers_variant_and_rank_without_metadata(self):
        # Files written before the metadata existed must still load, including oft vs boft, which
        # share a tensor NAME and differ only in its rank.
        for variant, rank in (("lora", 4), ("dora", 4), ("loha", 4), ("lokr", 4),
                              ("oft", 8), ("boft", 8)):
            adapters = {"blocks.0.attn.wq": _adapter(variant, rank=rank)}
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "a.safetensors")
                save_lora(adapters, path, rank=rank, alpha=float(rank))
                sd = load_file(path)
            spec = lora_spec_from_state(sd, {})
            self.assertEqual(spec["variant"], variant)
            self.assertEqual(spec["rank"], rank, variant)

    def test_spec_rejects_a_file_with_no_adapter_tensors(self):
        with self.assertRaises(ValueError):
            lora_spec_from_state({"diffusion_model.blocks.0.attn.wq.base.weight": torch.zeros(2, 2)}, {})

    def test_magnitude_and_factor_tensors_are_saved_fp32(self):
        # dora_m holds the base weight's column norms; bf16 there is a ~4e-3 relative perturbation of
        # the effective weight on every save/resume cycle, not a rounding detail.
        adapters = {"blocks.0.attn.wq": _adapter("dora")}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.safetensors")
            save_lora(adapters, path, rank=4, alpha=4)
            sd = load_file(path)
        self.assertEqual(sd["diffusion_model.blocks.0.attn.wq.dora_m.weight"].dtype, torch.float32)
        self.assertEqual(sd["diffusion_model.blocks.0.attn.wq.lora_A.weight"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
