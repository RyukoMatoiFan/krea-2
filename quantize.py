"""fp8 quantization of the frozen LoRA base, to roughly halve resident weight VRAM.

For a LoRA fine-tune the 12B DiT base is frozen — only the adapters train — so its weights can be
stored in 8-bit (e4m3) and dequantized on the fly in the forward. ``quantize_dit_fp8`` replaces the
transformer blocks' ``nn.Linear`` (the attention + MLP bulk) with :class:`Fp8Linear`; the small
input/output/text-fusion stages stay bf16 (sensitive, negligible VRAM).

Composes with block-swap and gradient checkpointing: the fp8 weight + scale are stored as
``requires_grad=False`` Parameters, so ``enable_block_swap`` (which pages ``.parameters()``) moves
them CPU<->GPU exactly like a bf16 base weight, and ``skip_trainable=True`` keeps the trainable
adapters resident. Gradients only ever touch the bf16 adapters, so the fp8 path never hits the
PyTorch fp8-norm / grad-clip limitations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

E4M3_MAX = 448.0  # max representable magnitude of float8_e4m3fn


class Fp8Linear(nn.Module):
    """Frozen ``nn.Linear`` with the weight in fp8 (e4m3) + a per-output-row scale.

    ``y = (W_fp8.to(x.dtype) * scale) @ x^T (+ bias)`` — dequantized per forward, the temporary bf16
    weight freed straight after. Exposes ``in_features`` / ``out_features`` / ``weight`` so it drops
    in as a LoRALinear base.
    """

    is_quant_linear = True

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.data.float()                                   # (out, in)
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / E4M3_MAX   # (out, 1)
        wq = (w / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        self.weight_fp8 = nn.Parameter(wq, requires_grad=False)
        self.scale = nn.Parameter(scale.to(torch.bfloat16), requires_grad=False)
        self.bias = (nn.Parameter(lin.bias.data.clone(), requires_grad=False)
                     if lin.bias is not None else None)
        self.in_features, self.out_features = lin.in_features, lin.out_features

    @property
    def weight(self) -> torch.Tensor:  # device/dtype probe for LoRALinear
        return self.weight_fp8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_fp8.to(x.dtype) * self.scale.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


def quantize_dit_fp8(dit) -> int:
    """Replace the transformer blocks' ``nn.Linear`` (attn + MLP) with :class:`Fp8Linear` in place.

    Leaves ``first`` / ``last`` / text-fusion / time-MLP in bf16. Returns the number of layers
    quantized. Call BEFORE ``inject_lora`` (the adapters then wrap the fp8 bases).
    """
    targets = []
    for block in dit.blocks:
        for mod in block.modules():
            for name, child in mod.named_children():
                if isinstance(child, nn.Linear):
                    targets.append((mod, name, child))
    for mod, name, child in targets:
        setattr(mod, name, Fp8Linear(child))
    return len(targets)
