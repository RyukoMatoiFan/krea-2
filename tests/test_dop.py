"""Differential output preservation: the prior term itself, and the caption -> prior-prompt rewrite.

Exercised against a stand-in DiT rather than the real one, because what needs proving is the
CONTRACT: the term is exactly zero while the adapter is a no-op, it is driven by the difference
between the adapted and the base output on the SAME input, and it enters the total loss at exactly
the configured multiplier. A term that quietly reduces to zero -- or to a constant -- is the failure
mode this file exists to catch.
"""
import unittest

import torch
from torch import nn

from lora import LoRALinear, adapter_modules
from train_t2i import strip_trigger, t2i_training_step


class _TinyDiT(nn.Module):
    """Minimal ``dit(img, context, t, pos, mask)`` -> (B, n, 64) that actually reads the context.

    The prediction must depend on the text, or a preservation term evaluated at a different prompt
    would be trivially satisfied and the test would pass for the wrong reason.
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 64)
        self.gradient_checkpointing = False

    def forward(self, *, img, context, t, pos, mask, ref_len=0, ref_lens=None):
        ctx = context.float().mean(dim=(1, 2))[:, :64].unsqueeze(1)   # (B, 1, 64)
        return self.proj(img.float()) + ctx + t.view(-1, 1, 1)


class _Flow:
    timestep_weighting = "uniform"
    min_snr_gamma = 0.0
    noise_offset = 0.0
    input_perturbation = 0.0


def _schedule(_n, **_kw):
    class S:
        def __call__(self, u):
            return u
    return S()


def _batch(seed=0, B=2, n=8, L=3):
    g = torch.Generator().manual_seed(seed)
    return {
        "z0": torch.randn(B, n, 64, generator=g),
        "context": torch.randn(B, L, 12, 2560, generator=g),
        "text_mask": torch.ones(B, L, dtype=torch.bool),
        "prior": torch.randn(B, L, 12, 2560, generator=g),
    }


def _step(dit, data, *, preserve=True, weight=1.0, parts=None, seed=0):
    return t2i_training_step(
        dit, z0=data["z0"], context=data["context"], text_mask=data["text_mask"],
        grid_h=2, grid_w=4, schedule=_schedule(8), flow_cfg=_Flow(),
        generator=torch.Generator().manual_seed(seed), t_override=0.5,
        preserve_context=data["prior"] if preserve else None,
        preserve_text_mask=data["text_mask"] if preserve else None,
        preserve_weight=weight, loss_parts=parts)


class DopTests(unittest.TestCase):
    def _dit(self):
        torch.manual_seed(0)
        dit = _TinyDiT()
        dit.proj = LoRALinear(dit.proj, rank=4, alpha=4.0)
        return dit

    def test_term_is_exactly_zero_while_the_adapter_is_a_noop(self):
        dit, data = self._dit(), _batch()
        parts = {}
        total = _step(dit, data, parts=parts)
        self.assertEqual(parts["loss_dop"], 0.0)
        self.assertAlmostEqual(float(total.detach()), parts["loss_flow"], places=6)

    def test_term_is_positive_once_the_adapter_moves_and_scales_by_the_multiplier(self):
        data = _batch()
        dit = self._dit()
        with torch.no_grad():
            dit.proj.lora_B.normal_(0.0, 0.5)     # the adapter now changes the output
        parts = {}
        total = _step(dit, data, weight=1.0, parts=parts)
        self.assertGreater(parts["loss_dop"], 1e-6)
        self.assertAlmostEqual(float(total.detach()), parts["loss_flow"] + parts["loss_dop"], places=5)

        parts3 = {}
        total3 = _step(dit, data, weight=3.0, parts=parts3)
        self.assertAlmostEqual(parts3["loss_dop"], parts["loss_dop"], places=6)
        self.assertAlmostEqual(float(total3.detach()), parts3["loss_flow"] + 3.0 * parts3["loss_dop"], places=5)

    def test_the_term_carries_gradient_to_the_adapter(self):
        # The preserved forward must be the one WITH grad; if the two forwards were swapped the loss
        # would still look right and train nothing.
        data = _batch()
        dit = self._dit()
        with torch.no_grad():
            dit.proj.lora_B.normal_(0.0, 0.5)
        flow_only = _step(dit, data, preserve=False)
        flow_only.backward()
        g_plain = dit.proj.lora_B.grad.clone()
        dit.proj.lora_B.grad = None
        _step(dit, data, weight=5.0).backward()
        self.assertFalse(torch.allclose(g_plain, dit.proj.lora_B.grad, atol=1e-6))

    def test_it_refuses_to_run_without_adapters(self):
        # With no adapters, base and adapted forwards are identical -> a silent no-op regulariser.
        dit = _TinyDiT()
        self.assertEqual(adapter_modules(dit), [])
        with self.assertRaises(RuntimeError):
            _step(dit, _batch())

    def test_a_prior_context_without_its_mask_is_rejected(self):
        dit, data = self._dit(), _batch()
        with self.assertRaises(ValueError):
            t2i_training_step(dit, z0=data["z0"], context=data["context"],
                              text_mask=data["text_mask"], grid_h=2, grid_w=4,
                              schedule=_schedule(8), flow_cfg=_Flow(), t_override=0.5,
                              preserve_context=data["prior"])

    def test_zero_weight_skips_the_term_entirely(self):
        dit = _TinyDiT()          # no adapters: proves the term was never evaluated
        parts = {}
        _step(dit, _batch(), weight=0.0, parts=parts)
        self.assertEqual(parts["loss_dop"], 0.0)


class DopBatchShapeTests(unittest.TestCase):
    """The two shapes the trainer really passes: a broadcast fixed prompt, and the edit stream."""

    def _dit(self):
        torch.manual_seed(0)
        dit = _TinyDiT()
        dit.proj = LoRALinear(dit.proj, rank=4, alpha=4.0)
        with torch.no_grad():
            dit.proj.lora_B.normal_(0.0, 0.5)
        return dit

    def test_fixed_prompt_is_broadcast_across_the_batch(self):
        # dop.prompt is encoded once as (1, L, 12, 2560) and expanded to the batch, so the prior
        # tensors reaching the DiT are non-contiguous views -- pos/mask building must cope.
        data = _batch(B=3)
        prior = torch.randn(1, 3, 12, 2560, generator=torch.Generator().manual_seed(1))
        pmask = torch.ones(1, 3, dtype=torch.bool)
        parts = {}
        t2i_training_step(
            self._dit(), z0=data["z0"], context=data["context"], text_mask=data["text_mask"],
            grid_h=2, grid_w=4, schedule=_schedule(8), flow_cfg=_Flow(), t_override=0.5,
            preserve_context=prior.expand(3, *prior.shape[1:]),
            preserve_text_mask=pmask.expand(3, 3), loss_parts=parts)
        self.assertGreater(parts["loss_dop"], 1e-6)

    def test_edit_path_preserves_only_the_target_tokens(self):
        from train_t2i import edit_training_step

        B, n_tgt, n_ref = 2, 8, 6
        g = torch.Generator().manual_seed(0)
        z0 = torch.randn(B, n_tgt, 64, generator=g)
        refs = [torch.randn(B, n_ref, 64, generator=g)]
        ctx = torch.randn(B, 3, 12, 2560, generator=g)
        prior = torch.randn(B, 3, 12, 2560, generator=g)
        tmask = torch.ones(B, 3, dtype=torch.bool)
        parts = {}
        loss = edit_training_step(
            self._dit(), z0=z0, refs=refs, ref_grids=[(2, 3)], context=ctx, text_mask=tmask,
            grid_h=2, grid_w=4, schedule=_schedule(14), flow_cfg=_Flow(), t_override=0.5,
            preserve_context=prior, preserve_text_mask=tmask, preserve_weight=2.0,
            loss_parts=parts)
        # A slice over the wrong axis length would raise; a missing slice would compare reference
        # tokens too and quietly dilute the term.
        self.assertGreater(parts["loss_dop"], 1e-6)
        self.assertAlmostEqual(float(loss.detach()),
                               parts["loss_flow"] + 2.0 * parts["loss_dop"], places=5)


class StripTriggerTests(unittest.TestCase):
    def test_removes_the_trigger_and_tidies_the_punctuation_it_orphans(self):
        self.assertEqual(strip_trigger("photo of sks dog, outdoors", "sks"),
                         "photo of dog, outdoors")
        self.assertEqual(strip_trigger("sks, a portrait", "sks"), "a portrait")
        self.assertEqual(strip_trigger("a portrait of sks", "sks"), "a portrait of")
        # Repeats collapse to ONE separator rather than vanishing: in a comma-delimited caption the
        # surviving items are still separate items.
        self.assertEqual(strip_trigger("a sks, sks, cat", "sks"), "a, cat")

    def test_matches_whole_words_case_insensitively(self):
        self.assertEqual(strip_trigger("A SKS dog", "sks"), "A dog")
        # A substring hit would corrupt unrelated words -- "asks" must survive intact.
        self.assertEqual(strip_trigger("she asks for sks", "sks"), "she asks for")

    def test_empty_trigger_is_a_passthrough(self):
        self.assertEqual(strip_trigger("unchanged, caption", ""), "unchanged, caption")


if __name__ == "__main__":
    unittest.main()
