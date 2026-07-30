"""LoRA adapters for the Krea 2 ``SingleStreamDiT`` and text-side projection layers.

Custom (not PEFT) so saved adapters use the widely-consumed key convention
``diffusion_model.<module path>.lora_{A,B}.weight`` -> a LoRA trained on **Raw** loads directly on
**Turbo** in ComfyUI (Krea's recommended "train on Raw, run on Turbo").

Six update parameterisations are available via ``variant``: ``lora`` (low-rank product), ``dora``
(magnitude/direction decomposition), ``loha`` (Hadamard product of two low-rank pairs), ``lokr``
(Kronecker product), ``oft`` (orthogonal rotation via the Cayley transform) and ``boft`` (the same
rotation factorised into butterfly stages). ``loha``/``lokr`` are the LyCORIS algorithms. The first
four ADD to the weight; the last two ROTATE it, preserving neuron angles and norms by construction.
They share one interface, so scaling, saving, loading, EMA and parameter collection are
variant-agnostic -- anything that names ``lora_A``/``lora_B`` by hand silently or loudly breaks the
four variants that have no such tensors, so go through :func:`adapter_tensors`,
:func:`ema_named_tensors` and :func:`lora_spec_from_state` instead.

Targeted Linears per DiT/text-fusion block (names from mmdit.py): the attention q/k/v/out projections
and the SwiGLU MLP gate/up/down. The text MLP projection stage is also adapted by default. The base
weight stays frozen bf16; adapters train in fp32 (``lora_B`` init 0 -> the adapter is a no-op at step 0).
"""
from __future__ import annotations

import contextlib
import math
import os

import torch
import torch.nn as nn
from safetensors.torch import save_file

# Linear submodule leaf-paths inside each SingleStreamBlock / TextFusionBlock to adapt.
# Includes attn.gate (the [dim,dim] sigmoid output-gate projection, mmdit.py Attention): leaving it
# out costs the adapter control over how much attention output each head contributes.
DEFAULT_TARGETS = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate",
                   "mlp.gate", "mlp.up", "mlp.down")


class AdapterLinear(nn.Module):
    """Shared plumbing for every variant: a frozen base, the alpha/rank scale and the slider knob.

    Variants add their own tensors; nothing here assumes a low-rank pair, so a variant that does not
    use one (LoKr) does not inherit unused parameters into the optimizer and the checkpoint.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.scale_mul = 1.0  # runtime multiplier (slider knob); 1.0 -> normal, 0.0 -> off


class LoRALinear(AdapterLinear):
    """``base(x) + scale * (x @ A^T) @ B^T`` with ``A:(rank,in)``, ``B:(out,rank)``, ``B`` init 0."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__(base, rank, alpha)
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


class DoRALinear(LoRALinear):
    """Weight-decomposed adaptation: the weight is split into a magnitude and a direction, LoRA
    adapts the direction, and the magnitude is trained separately.

    ``W' = m * (W + dW) / ||W + dW||_c`` with the norm taken per input column. The magnitude starts
    at the base weight's own column norms, so the adapter is a no-op at step 0 exactly as LoRA is.
    Separating the two lets the update change direction without being forced to change scale, which
    targets the low-rank regime -- at the cost of materialising the full weight each forward, since the rescaling is
    not additive and cannot be folded into a second small matmul.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__(base, rank, alpha)
        w = self._base_weight()
        self.dora_m = nn.Parameter(w.float().norm(dim=0, keepdim=True))   # (1, in)

    def _base_weight(self) -> torch.Tensor:
        w = self.base.weight
        if getattr(self.base, "is_quant_linear", False):
            raise RuntimeError("DoRA needs a dequantized base weight; use variant=lora with a "
                               "quantized base, or keep the base in bf16")
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_mul == 0.0:
            return self.base(x)
        w = self._base_weight().float()
        dw = (self.scale * self.scale_mul) * (self.lora_B @ self.lora_A)
        v = w + dw
        w_eff = self.dora_m * v / v.norm(dim=0, keepdim=True).clamp(min=1e-6)
        bias = self.base.bias
        return nn.functional.linear(x, w_eff.to(x.dtype),
                                    bias.to(x.dtype) if bias is not None else None)


class LoHaLinear(LoRALinear):
    """Hadamard-product adaptation: ``dW = (B1 A1) * (B2 A2)``, elementwise.

    Two low-rank pairs instead of one, so the parameter count matches a rank-2r LoRA while the
    effective rank of the product can reach r^2. The elementwise product cannot be factored back
    into two matmuls, so the full ``dW`` is materialised per forward -- the cost paid for the extra
    expressivity.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__(base, rank, alpha)
        dev = base.weight.device
        self.loha_A2 = nn.Parameter(torch.empty(rank, base.in_features, device=dev,
                                                dtype=torch.float32))
        self.loha_B2 = nn.Parameter(torch.zeros(base.out_features, rank, device=dev,
                                                dtype=torch.float32))
        nn.init.kaiming_uniform_(self.loha_A2, a=math.sqrt(5))
        # Only the SECOND pair is zeroed. The product is elementwise, so zeroing the FIRST pair
        # would also zero the gradient reaching the second one and leave three of the four tensors
        # dead at init; this way the product still starts at zero but three tensors learn from
        # step 0 and only lora_A waits for one step.
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * (self.loha_B2 @ self.loha_A2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.scale_mul == 0.0:
            return out
        dw = self.delta_weight()
        delta = nn.functional.linear(x.to(dw.dtype), dw)
        return out + (self.scale * self.scale_mul) * delta.to(out.dtype)


class LoKrLinear(AdapterLinear):
    """Kronecker-product adaptation: ``dW = A (x) B``, with ``B`` optionally low-rank.

    Ranks multiply rather than add under the Kronecker product, so a small parameter budget can
    still reach a high effective rank -- the reason this variant produces very small files and is
    favoured for style. The factor shapes come from the largest balanced split of the layer's
    dimensions, so no configuration beyond ``rank`` is required.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, factor: int = 8):
        super().__init__(base, rank, alpha)
        dev = base.weight.device
        out_f, in_f = base.out_features, base.in_features
        a1, a2 = _kron_split(out_f, factor), _kron_split(in_f, factor)
        b1, b2 = out_f // a1, in_f // a2
        self.lokr_A = nn.Parameter(torch.empty(a1, a2, device=dev, dtype=torch.float32))
        # B is factored as B1 @ B2 so its rank is controllable; B1 init zero keeps dW zero at step 0.
        r = min(rank, b1, b2)
        self.lokr_B1 = nn.Parameter(torch.zeros(b1, r, device=dev, dtype=torch.float32))
        self.lokr_B2 = nn.Parameter(torch.empty(r, b2, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lokr_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lokr_B2, a=math.sqrt(5))

    def delta_weight(self) -> torch.Tensor:
        return torch.kron(self.lokr_A, self.lokr_B1 @ self.lokr_B2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.scale_mul == 0.0:
            return out
        dw = self.delta_weight()
        delta = nn.functional.linear(x.to(dw.dtype), dw)
        return out + (self.scale * self.scale_mul) * delta.to(out.dtype)


class OFTLinear(AdapterLinear):
    """Orthogonal fine-tuning: ``W' = R W`` with ``R`` block-diagonal and orthogonal.

    Every other variant here adds something to the weight; this one *rotates* it. An orthogonal map
    preserves the angles between neurons and the norm of every row, so the pretrained structure is
    kept by construction rather than by the update happening to stay small.

    ``R`` comes from the Cayley transform of a skew-symmetric matrix, ``R = (I - Q)^-1 (I + Q)``,
    which is orthogonal for *any* skew ``Q``. So the parameters are unconstrained: no projection, no
    penalty, no drift off the manifold during training. ``Q`` starts at zero, giving ``R = I`` and an
    exact no-op at step 0.

    The rotation is applied to the layer OUTPUT rather than to the weight. ``W' x = R (W x)``, so the
    two are identical, but rotating the output costs one small block matmul per token instead of
    rebuilding the full weight matrix every forward.

    ``rank`` here is the NUMBER OF BLOCKS, not a low-rank dimension -- and unlike LoRA a *larger*
    value means *fewer* parameters, since the blocks get smaller. Worth remembering when comparing
    variants at equal parameter count.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__(base, rank, alpha)
        out_f = base.out_features
        n = max(1, min(rank, out_f))
        while out_f % n:                      # largest divisor of out_features not exceeding rank
            n -= 1
        blk = out_f // n
        self.n_blocks, self.blk = n, blk
        # Stored full for simplicity; only the strict upper triangle is effective, so the honest
        # parameter count for a comparison is half of numel.
        self.oft_Q = nn.Parameter(torch.zeros(n, blk, blk, device=base.weight.device,
                                              dtype=torch.float32))

    def _rotation(self) -> torch.Tensor:
        q = self.oft_Q * (self.scale * self.scale_mul)
        q = q - q.transpose(-1, -2)                      # skew-symmetric
        eye = torch.eye(self.blk, device=q.device, dtype=q.dtype).expand_as(q)
        return torch.linalg.solve(eye - q, eye + q)      # Cayley -> orthogonal

    def _rotate(self, y2d: torch.Tensor) -> torch.Tensor:
        return torch.einsum("tnb,ncb->tnc", y2d.reshape(-1, self.n_blocks, self.blk),
                            self._rotation()).reshape(y2d.shape)

    def rotation_matrix(self) -> torch.Tensor:
        """The composed map, for verifying orthogonality of what the layer actually applies."""
        eye = torch.eye(self.n_blocks * self.blk, device=self.oft_Q.device, dtype=torch.float32)
        return self._rotate(eye)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.scale_mul == 0.0:
            return out
        bias = self.base.bias
        # The rotation applies to W only, so the bias is lifted out and added back afterwards. Both
        # steps stay in fp32: doing them in bf16 costs a full ULP on every forward, which shows up
        # as the adapter not being an exact no-op at init.
        y = out.float()
        if bias is not None:
            y = y - bias.float()
        shape = y.shape
        y = self._rotate(y.reshape(-1, shape[-1])).reshape(shape)
        if bias is not None:
            y = y + bias.float()
        return y.to(out.dtype)


class BOFTLinear(AdapterLinear):
    """Butterfly orthogonal fine-tuning: ``R`` as a PRODUCT of block-diagonal orthogonal factors
    separated by stride permutations.

    A single block-diagonal rotation can only ever mix a coordinate with the others inside its own
    block, so reaching dense connectivity that way means large blocks, and a block costs the square
    of its width in parameters. Factorising instead reaches the same connectivity from *small*
    blocks, by stacking a few stages and permuting between them -- the trick a radix-b FFT uses to
    connect every input to every output in log_b(d) stages -- at a parameter cost that grows
    linearly in the width rather than quadratically.

    Orthogonality is free here and needs no checking of the permutation choice: every factor is
    orthogonal by the Cayley transform, every permutation is orthogonal, and a product of orthogonal
    maps is orthogonal. The permutation therefore only affects how fast information spreads, never
    whether the constraint holds.

    The accumulated permutation is undone at the end, so with all ``Q`` at zero every factor is the
    identity and the adapter is an exact no-op at step 0 for any number of factors.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, n_factors: int | None = None):
        super().__init__(base, rank, alpha)
        out_f = base.out_features
        n = max(1, min(rank, out_f))
        while out_f % n:
            n -= 1
        blk = out_f // n
        if n_factors is None:                       # enough stages for a signal to cross the width
            m, reach = 1, blk
            while reach < out_f and m < 4:
                m, reach = m + 1, reach * blk
            n_factors = m
        self.n_blocks, self.blk, self.n_factors = n, blk, n_factors
        self.oft_Q = nn.Parameter(torch.zeros(n_factors, n, blk, blk,
                                              device=base.weight.device, dtype=torch.float32))

        # Stride permutation: index q*blk + r -> r*n + q. Applying it between factors makes the next
        # factor's blocks span coordinates the previous factor's blocks kept apart.
        perm = torch.arange(out_f).reshape(n, blk).t().reshape(-1)
        total = torch.arange(out_f)
        for _ in range(n_factors):
            total = total[perm]
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("unperm", torch.argsort(total), persistent=False)

    def _rotations(self) -> torch.Tensor:
        q = self.oft_Q * (self.scale * self.scale_mul)
        q = q - q.transpose(-1, -2)
        eye = torch.eye(self.blk, device=q.device, dtype=q.dtype).expand_as(q)
        return torch.linalg.solve(eye - q, eye + q)          # (n_factors, n_blocks, blk, blk)

    def _rotate(self, y2d: torch.Tensor) -> torch.Tensor:
        width = y2d.shape[-1]
        for r in self._rotations():
            y2d = torch.einsum("tnb,ncb->tnc", y2d.reshape(-1, self.n_blocks, self.blk), r)
            y2d = y2d.reshape(-1, width)[:, self.perm]
        return y2d[:, self.unperm]

    def rotation_matrix(self) -> torch.Tensor:
        """The composed map INCLUDING the permutations -- the part a factor-only check would miss."""
        eye = torch.eye(self.n_blocks * self.blk, device=self.oft_Q.device, dtype=torch.float32)
        return self._rotate(eye)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.scale_mul == 0.0:
            return out
        bias = self.base.bias
        y = out.float()
        if bias is not None:
            y = y - bias.float()
        shape = y.shape
        y = self._rotate(y.reshape(-1, shape[-1])).reshape(shape)
        if bias is not None:
            y = y + bias.float()
        return y.to(out.dtype)


def _kron_split(dim: int, factor: int) -> int:
    """Largest divisor of ``dim`` no greater than ``factor`` -- the Kronecker left-factor size."""
    for d in range(min(factor, dim), 0, -1):
        if dim % d == 0:
            return d
    return 1


VARIANTS = {"lora": LoRALinear, "dora": DoRALinear, "loha": LoHaLinear,
            "lokr": LoKrLinear, "oft": OFTLinear, "boft": BOFTLinear}

# Exact-type reverse map (not isinstance: DoRA subclasses LoRA, LoHa subclasses LoRA).
_VARIANT_BY_CLS = {cls: name for name, cls in VARIANTS.items()}


def _adapter_cls(variant: str):
    try:
        return VARIANTS[variant]
    except KeyError:
        raise SystemExit(f"unknown lora.variant {variant!r}; expected one of {sorted(VARIANTS)}")


def variant_of(adapters) -> str:
    """The variant name of an injected adapter (or of the first module in a {name: module} dict).

    Read off the live modules rather than passed in by the caller, so a saved file can never claim a
    variant the weights in it do not have.
    """
    if isinstance(adapters, dict):
        if not adapters:
            return "lora"
        adapters = next(iter(adapters.values()))
    return _VARIANT_BY_CLS.get(type(adapters), "lora")


def adapter_modules(model: nn.Module) -> list:
    """Every injected adapter in ``model`` -- empty if none were injected."""
    return [m for m in model.modules() if isinstance(m, AdapterLinear)]


def ema_named_tensors(pools: dict) -> dict:
    """``{tag: {name: adapter}}`` -> ``{f"{tag}.{name}.{tensor}": live tensor}`` for :class:`EmaModel`.

    Variant-agnostic on purpose. Naming the tensors by hand works only for plain LoRA: LoKr, OFT and
    BOFT have no ``lora_A``/``lora_B`` at all (so hand-naming raises), while DoRA's magnitude and
    LoHa's second pair simply would not be averaged -- an EMA silently covering half of an adapter is
    worse than none, since the saved EMA weights are then an inconsistent mixture of averaged and
    instantaneous tensors. Plain-LoRA keys are unchanged, so existing EMA resume state still loads.
    """
    named = {}
    for tag, pool in pools.items():
        for name, m in pool.items():
            for pname, p in adapter_tensors(m).items():
                named[f"{tag}.{name}.{pname}"] = p
    return named


@contextlib.contextmanager
def lora_scaled(model: nn.Module, factor: float):
    """Temporarily multiply every injected adapter's effect by ``factor`` (the slider knob).

    ``factor`` may be negative (invert the direction) or 0 (disable). Restores prior scales on exit.
    """
    mods = [m for m in model.modules() if isinstance(m, AdapterLinear)]
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
                include_txtfusion: bool = True, include_txtmlp: bool = True,
                variant: str = "lora") -> dict:
    """Freeze the DiT and wrap each target Linear with the adapter for ``variant``.

    ``variant`` selects the parameterisation of the update: ``lora`` | ``dora`` | ``loha`` | ``lokr``.
    Returns {name: module}.
    """
    cls = _adapter_cls(variant)
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
                # nn.Linear, or an fp8-quantized base (quantize.Fp8Linear) carrying the same interface
                if not (isinstance(base, nn.Linear) or getattr(base, "is_quant_linear", False)):
                    continue
                lora = cls(base, rank, alpha)
                setattr(parent, leaf, lora)
                adapters[f"{prefix}.{i}.{tgt}"] = lora
    if include_txtmlp:
        for name, base in list(dit.txtmlp.named_modules()):
            if not name or not (isinstance(base, nn.Linear) or getattr(base, "is_quant_linear", False)):
                continue
            parent = dit.txtmlp.get_submodule(name.rsplit(".", 1)[0]) if "." in name else dit.txtmlp
            leaf = name.rsplit(".", 1)[-1]
            lora = cls(base, rank, alpha)
            setattr(parent, leaf, lora)
            adapters[f"txtmlp.{name}"] = lora
    if not adapters:
        raise RuntimeError("inject_lora matched no Linear targets; check target names vs mmdit.py")
    return adapters


def adapter_tensors(m: nn.Module) -> dict:
    """The adapter's own trainable tensors, keyed by name -- everything except the frozen base.

    Variant-agnostic, so LoHa's second pair, LoKr's factors and DoRA's magnitude are all picked up
    without the caller knowing which variant it holds.
    """
    return {n: p for n, p in m.named_parameters() if not n.startswith("base.")}


def lora_parameters(adapters: dict) -> list:
    params = []
    for m in adapters.values():
        params.extend(adapter_tensors(m).values())
    return params


# Qwen3-VL language-model projection names (TE-LoRA targets); the vision tower is left unadapted.
QWEN_TE_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def inject_lora_te(qwen, rank: int, alpha: float | None = None, *, targets=QWEN_TE_TARGETS,
                   variant: str = "lora") -> dict:
    """Wrap the Qwen3-VL **language-model** Linear projections with the ``variant`` adapter (TE-LoRA).

    The base Qwen3-VL stays frozen; only the adapters train. The vision tower (``visual``/``vision``)
    is left untouched -- we adapt text understanding, not image features. Robust to layer-path
    differences across transformers versions (matches by leaf name). Returns {module path: adapter}.
    """
    cls = _adapter_cls(variant)
    alpha = alpha if alpha is not None else float(rank)
    qwen.requires_grad_(False)
    adapters: dict[str, LoRALinear] = {}
    for name, module in list(qwen.named_modules()):
        if not isinstance(module, nn.Linear) or name.split(".")[-1] not in targets:
            continue
        if "visual" in name or "vision" in name:
            continue
        parent = qwen.get_submodule(name.rsplit(".", 1)[0]) if "." in name else qwen
        lora = cls(module, rank, alpha)
        setattr(parent, name.split(".")[-1], lora)
        adapters[name] = lora
    if not adapters:
        raise RuntimeError("inject_lora_te matched no Qwen3-VL Linear targets (check QWEN_TE_TARGETS)")
    return adapters


def save_lora(adapters: dict, path: str, *, rank: int, alpha: float, metadata: dict | None = None,
              key_prefix: str = "diffusion_model") -> None:
    """Save adapters as safetensors with ``<key_prefix>.<name>.lora_{A,B}.weight`` keys (bf16).

    ``key_prefix`` is ``diffusion_model`` for DiT LoRA (the convention ComfyUI loads, so a LoRA
    trained on Raw runs on Turbo)
    and ``text_encoder`` for TE-LoRA (Qwen3-VL adapters).

    ``lora_A``/``lora_B`` are written bf16 (the consumed convention); every other tensor is written
    fp32. Those others are not small deltas around zero: DoRA's ``dora_m`` holds the base weight's
    own column norms, so rounding it to bf16 perturbs the effective weight by ~4e-3 relative on
    every save/resume cycle -- a real quality change, not a rounding detail.
    """
    sd = {}
    for name, m in adapters.items():
        # Keyed by the adapter's own parameter names, so plain LoRA still writes exactly
        # lora_A.weight / lora_B.weight while the variants carry their extra factors alongside.
        for pname, p in adapter_tensors(m).items():
            dt = torch.bfloat16 if pname in ("lora_A", "lora_B") else torch.float32
            sd[f"{key_prefix}.{name}.{pname}.weight"] = p.detach().to(dt).cpu().contiguous()
    meta = {"format": "krea2-lora", "rank": str(rank), "alpha": str(alpha)}
    if metadata:
        meta.update({k: str(v) for k, v in metadata.items()})
    # Derived last: the file's declared variant is whatever the saved modules actually are, so a
    # caller passing a stale metadata["variant"] cannot mislabel a checkpoint.
    meta["variant"] = variant_of(adapters)
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
        tensors = adapter_tensors(m)
        saved = {pname: sd.get(f"{key_prefix}.{name}.{pname}.weight") for pname in tensors}
        # A partially present adapter is a variant/rank mismatch, not a resumable state: refuse it
        # rather than silently loading half an adapter and training on top of the other half.
        if any(v is None for v in saved.values()):
            if all(v is None for v in saved.values()):
                missing += 1
                continue
            raise SystemExit(f"adapter {name!r}: checkpoint has only "
                             f"{sorted(k for k, v in saved.items() if v is not None)}, "
                             f"expected {sorted(tensors)} -- variant or rank mismatch")
        with torch.no_grad():
            for pname, p in tensors.items():
                p.copy_(saved[pname].to(p.dtype))
    return missing


def load_lora_stage_init(adapters: dict, te_adapters: dict, path: str) -> None:
    """Load a weights-only stage initialization without restoring training state."""
    if not path:
        return
    if not adapters:
        raise SystemExit("paths.lora_init is set, but lora.train_transformer is false")
    if not os.path.isfile(path):
        raise SystemExit(f"paths.lora_init does not exist: {path}")
    missing = load_lora_weights(adapters, path)
    if missing:
        raise SystemExit(
            f"paths.lora_init is missing {missing}/{len(adapters)} DiT adapters; "
            "rank, variant, or target modules do not match")
    if te_adapters:
        root, ext = os.path.splitext(path)
        te_path = f"{root}.te{ext or '.safetensors'}"
        if not os.path.isfile(te_path):
            raise SystemExit(
                f"TE-LoRA is enabled, but the paired stage initialization is missing: {te_path}")
        te_missing = load_lora_weights(te_adapters, te_path, key_prefix="text_encoder")
        if te_missing:
            raise SystemExit(
                f"paired TE initialization is missing {te_missing}/{len(te_adapters)} adapters")


def lora_spec_from_state(sd: dict, meta: dict | None = None, *,
                         key_prefix: str = "diffusion_model") -> dict:
    """Recover everything :func:`inject_lora` needs from a saved state dict (+ safetensors metadata).

    Returns ``{variant, rank, alpha, include_txtfusion, include_txtmlp}``.

    Metadata is authoritative when present (our own saves write ``variant``/``rank``/``alpha``), and
    the key shapes are the fallback for files that predate it. ``alpha`` matters: the adapter's effect
    is scaled by ``alpha/rank``, so defaulting it to ``rank`` silently renders every LoRA trained with
    a non-default alpha at the wrong strength.
    """
    meta = meta or {}
    tensors: dict[str, torch.Tensor] = {}
    include_txtfusion = include_txtmlp = False
    for k, v in sd.items():
        body = k[len(key_prefix) + 1:] if k.startswith(key_prefix + ".") else k
        name, pname = body.rsplit(".", 2)[0], body.rsplit(".", 2)[-2]
        if "txtfusion" in name:
            include_txtfusion = True
        if "txtmlp" in name:
            include_txtmlp = True
        tensors.setdefault(pname, v)

    variant = meta.get("variant")
    if variant not in VARIANTS:
        # Infer from which tensors are present. BOFT and OFT share the parameter NAME and are told
        # apart only by its rank: (n_factors, n_blocks, blk, blk) vs (n_blocks, blk, blk).
        if "oft_Q" in tensors:
            variant = "boft" if tensors["oft_Q"].dim() == 4 else "oft"
        elif "lokr_A" in tensors:
            variant = "lokr"
        elif "loha_A2" in tensors:
            variant = "loha"
        elif "dora_m" in tensors:
            variant = "dora"
        elif "lora_A" in tensors:
            variant = "lora"
        else:
            raise ValueError(f"no recognisable adapter tensors found (keys: {sorted(tensors)})")

    rank = None
    if str(meta.get("rank", "")).isdigit():
        rank = int(meta["rank"])
    if rank is None:
        # For oft/boft `rank` is the block COUNT and for lokr the inner rank of the B factor, so each
        # variant reads it from a different axis of a different tensor.
        if variant in ("oft", "boft"):
            q = tensors["oft_Q"]
            rank = int(q.shape[1] if q.dim() == 4 else q.shape[0])
        elif variant == "lokr":
            rank = int(tensors["lokr_B1"].shape[1])
        elif "lora_A" in tensors:
            rank = int(tensors["lora_A"].shape[0])
        else:
            raise ValueError(f"cannot infer rank for variant {variant!r} (keys: {sorted(tensors)})")
    try:
        alpha = float(meta["alpha"])
    except (KeyError, TypeError, ValueError):
        alpha = float(rank)
    return {"variant": variant, "rank": rank, "alpha": alpha,
            "include_txtfusion": include_txtfusion, "include_txtmlp": include_txtmlp}


def load_lora(dit, path: str) -> dict:
    """Inject adapters matching a saved LoRA and load its weights (frozen, for inference).

    Infers the variant, rank, alpha and target set from the file, so every variant this module can
    train can also be sampled -- OFT/BOFT/LoKr adapters carry no ``lora_A`` at all.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    sd = load_file(path)
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
    spec = lora_spec_from_state(sd, meta)
    adapters = inject_lora(dit, spec["rank"], spec["alpha"], variant=spec["variant"],
                           include_txtfusion=spec["include_txtfusion"],
                           include_txtmlp=spec["include_txtmlp"])
    missing = load_lora_weights(adapters, path)
    if missing:
        print(f"[load_lora] warning: {missing} injected adapters had no saved weights")
    print(f"[load_lora] {os.path.basename(path)}: variant={spec['variant']} rank={spec['rank']} "
          f"alpha={spec['alpha']:g} ({len(adapters)} adapters)")
    return adapters
