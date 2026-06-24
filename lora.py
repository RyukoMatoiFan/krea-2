"""LoRA adapters for the Krea 2 ``SingleStreamDiT`` (and optionally its text-fusion stage).

Custom (not PEFT) so saved adapters use the ai-toolkit / ComfyUI key convention
``diffusion_model.<module path>.lora_{A,B}.weight`` -> a LoRA trained on **Raw** loads directly on
**Turbo** in ComfyUI (Krea's recommended "train on Raw, run on Turbo").

Targeted Linears per block (names from mmdit.py): the attention q/k/v/out projections and the SwiGLU
MLP gate/up/down. The base weight stays frozen bf16; adapters train in fp32 (``lora_B`` init 0 -> the
adapter is a no-op at step 0).
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
from safetensors.torch import save_file

# Linear submodule leaf-paths inside each SingleStreamBlock / TextFusionBlock to adapt.
# Includes attn.gate (the [dim,dim] sigmoid output-gate projection, mmdit.py Attention) to match
# the adaptation set used by the reference Krea 2 trainer (kohya-ss/musubi-tuner PR #979).
DEFAULT_TARGETS = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate",
                   "mlp.gate", "mlp.up", "mlp.down")


class LoRALinear(nn.Module):
    """``base(x) + scale * (x @ A^T) @ B^T`` with ``A:(rank,in)``, ``B:(out,rank)``, ``B`` init 0."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.scale_mul = 1.0  # runtime multiplier (slider knob); 1.0 -> normal LoRA, 0.0 -> off
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.scale_mul == 0.0:
            return out
        delta = (x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()
        return out + (self.scale * self.scale_mul) * delta.to(out.dtype)


@contextlib.contextmanager
def lora_scaled(model: nn.Module, factor: float):
    """Temporarily multiply every injected adapter's effect by ``factor`` (the slider knob).

    ``factor`` may be negative (invert the direction) or 0 (disable). Restores prior scales on exit.
    """
    mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
    prev = [m.scale_mul for m in mods]
    try:
        for m in mods:
            m.scale_mul = float(factor)
        yield
    finally:
        for m, p in zip(mods, prev):
            m.scale_mul = p


def lora_disabled(model: nn.Module):
    """Context manager: run with all injected adapters off (frozen-base prediction)."""
    return lora_scaled(model, 0.0)


def _resolve(parent: nn.Module, dotted: str):
    obj = parent
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def inject_lora(dit, rank: int, alpha: float | None = None, *, targets=DEFAULT_TARGETS,
                include_txtfusion: bool = False) -> dict:
    """Freeze the DiT and wrap each target Linear with :class:`LoRALinear`. Returns {name: module}."""
    alpha = alpha if alpha is not None else float(rank)
    dit.requires_grad_(False)
    groups = [("blocks", dit.blocks)]
    if include_txtfusion:
        groups.append(("txtfusion.layerwise_blocks", dit.txtfusion.layerwise_blocks))
        groups.append(("txtfusion.refiner_blocks", dit.txtfusion.refiner_blocks))
    adapters: dict[str, LoRALinear] = {}
    for prefix, modlist in groups:
        for i, blk in enumerate(modlist):
            for tgt in targets:
                try:
                    parent, leaf = _resolve(blk, tgt)
                    base = getattr(parent, leaf)
                except AttributeError:
                    continue
                if not isinstance(base, nn.Linear):
                    continue
                lora = LoRALinear(base, rank, alpha)
                setattr(parent, leaf, lora)
                adapters[f"{prefix}.{i}.{tgt}"] = lora
    if not adapters:
        raise RuntimeError("inject_lora matched no Linear targets; check target names vs mmdit.py")
    return adapters


def lora_parameters(adapters: dict) -> list:
    params = []
    for m in adapters.values():
        params.extend([m.lora_A, m.lora_B])
    return params


# Qwen3-VL language-model projection names (TE-LoRA targets); the vision tower is left unadapted.
QWEN_TE_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def inject_lora_te(qwen, rank: int, alpha: float | None = None, *, targets=QWEN_TE_TARGETS) -> dict:
    """Wrap the Qwen3-VL **language-model** Linear projections with :class:`LoRALinear` (TE-LoRA).

    The base Qwen3-VL stays frozen; only the adapters train. The vision tower (``visual``/``vision``)
    is left untouched -- we adapt text understanding, not image features. Robust to layer-path
    differences across transformers versions (matches by leaf name). Returns {module path: LoRALinear}.
    """
    alpha = alpha if alpha is not None else float(rank)
    qwen.requires_grad_(False)
    adapters: dict[str, LoRALinear] = {}
    for name, module in list(qwen.named_modules()):
        if not isinstance(module, nn.Linear) or name.split(".")[-1] not in targets:
            continue
        if "visual" in name or "vision" in name:
            continue
        parent = qwen.get_submodule(name.rsplit(".", 1)[0]) if "." in name else qwen
        lora = LoRALinear(module, rank, alpha)
        setattr(parent, name.split(".")[-1], lora)
        adapters[name] = lora
    if not adapters:
        raise RuntimeError("inject_lora_te matched no Qwen3-VL Linear targets (check QWEN_TE_TARGETS)")
    return adapters


def save_lora(adapters: dict, path: str, *, rank: int, alpha: float, metadata: dict | None = None,
              key_prefix: str = "diffusion_model") -> None:
    """Save adapters as safetensors with ``<key_prefix>.<name>.lora_{A,B}.weight`` keys (bf16).

    ``key_prefix`` is ``diffusion_model`` for DiT LoRA (ComfyUI/ai-toolkit convention, loads on Turbo)
    and ``text_encoder`` for TE-LoRA (Qwen3-VL adapters).
    """
    sd = {}
    for name, m in adapters.items():
        sd[f"{key_prefix}.{name}.lora_A.weight"] = m.lora_A.detach().to(torch.bfloat16).cpu().contiguous()
        sd[f"{key_prefix}.{name}.lora_B.weight"] = m.lora_B.detach().to(torch.bfloat16).cpu().contiguous()
    meta = {"format": "krea2-lora", "rank": str(rank), "alpha": str(alpha)}
    if metadata:
        meta.update({k: str(v) for k, v in metadata.items()})
    tmp = path + ".tmp"
    save_file(sd, tmp, metadata=meta)
    import os

    os.replace(tmp, path)


def load_lora_weights(adapters: dict, path: str, *, key_prefix: str = "diffusion_model") -> int:
    """Load saved LoRA weights into ALREADY-injected ``adapters`` (resume / reload).

    Copies ``<key_prefix>.<name>.lora_{A,B}.weight`` into the matching modules in place.
    Returns the count of adapters that had no saved weights (0 = clean load).
    """
    from safetensors.torch import load_file

    sd = load_file(path)
    missing = 0
    for name, m in adapters.items():
        a = sd.get(f"{key_prefix}.{name}.lora_A.weight")
        b = sd.get(f"{key_prefix}.{name}.lora_B.weight")
        if a is None or b is None:
            missing += 1
            continue
        with torch.no_grad():
            m.lora_A.copy_(a.to(m.lora_A.dtype))
            m.lora_B.copy_(b.to(m.lora_B.dtype))
    return missing


def load_lora(dit, path: str) -> dict:
    """Inject adapters matching a saved LoRA and load its weights (frozen, for inference).

    Infers rank + target set + whether the text-fusion stage was adapted from the saved keys.
    """
    from safetensors.torch import load_file

    sd = load_file(path)
    rank = None
    has_txt = False
    for k, v in sd.items():
        name = k[len("diffusion_model."):].rsplit(".lora_", 1)[0]
        if "txtfusion" in name:
            has_txt = True
        if k.endswith("lora_A.weight") and rank is None:
            rank = v.shape[0]
    if rank is None:
        raise ValueError(f"no lora_A weights found in {path}")
    adapters = inject_lora(dit, rank, include_txtfusion=has_txt)
    missing = load_lora_weights(adapters, path)
    if missing:
        print(f"[load_lora] warning: {missing} injected adapters had no saved weights")
    return adapters
