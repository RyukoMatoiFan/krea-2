import functools
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

# Whether the cuDNN SDPA backend can actually be pinned here. It is the fastest option where it
# exists, but pinning a backend that is not available turns into a hard failure inside SDPA,
# and three separate things can make it unavailable:
#   * ROCm -- torch.cuda is HIP under an alias, so is_available() is True and the enum member
#     still exists, yet there is no cuDNN. Only torch.version.hip distinguishes the builds.
#   * a CPU tensor -- the build having a GPU says nothing about where this tensor lives.
#   * a wheel compiled without the backend.
# Pin only when all three are ruled out; otherwise let SDPA choose, which is what it does on
# the compiled path anyway.
_CUDNN_SDPA = (
    hasattr(SDPBackend, "CUDNN_ATTENTION")
    and torch.cuda.is_available()
    and torch.version.hip is None
)


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    scale: float | None = None,
    gqa: bool = False,
) -> Tensor:
    if _is_block_mask(mask):
        # Block-sparse path: the kernel SKIPS whole blocks instead of computing and discarding
        # them, which a dense boolean mask cannot do. This is how reference<->reference attention
        # is dropped.
        x = _flex(q, k, v, block_mask=mask, scale=scale, enable_gqa=gqa)
    elif torch.compiler.is_compiling():
        # The backend-selection context is a graph break inside a compiled transformer block.
        # Default SDPA still selects the best available kernel for these tensors, while keeping
        # RMSNorm/modulation/gating/residual operations in the same Inductor graph.
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
        )
    elif _CUDNN_SDPA and q.is_cuda:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
            )
    else:
        # No cuDNN backend here (ROCm/HIP, a CPU tensor, or a wheel without it). Default
        # selection picks the best kernel the platform has rather than raising on a pin.
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
        )
    return rearrange(x, "B H L D -> B L (H D)")


def _mask(mask: Tensor) -> Tensor:
    """Broadcast a (B, L) key-padding mask without materialising an L-by-L tensor.

    Outputs at padded query positions are discarded, and padded positions are masked as keys, so
    masking queries as well cannot affect any retained token. The compact form is mathematically
    equivalent for retained outputs and is substantially cheaper for long edit sequences.
    """
    return mask.unsqueeze(1).unsqueeze(2)


def _is_block_mask(mask) -> bool:
    """True for a FlexAttention ``BlockMask`` (as opposed to a dense bool tensor or None)."""
    return mask is not None and not isinstance(mask, Tensor)


@functools.lru_cache(maxsize=8)
def _compiled_flex():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=False)


def _flex(q, k, v, *, block_mask, scale, enable_gqa):
    return _compiled_flex()(q, k, v, block_mask=block_mask, scale=scale, enable_gqa=enable_gqa)


def ref_block_mask(txt_len: int, ref_lens: list[int], tgt_len: int, total: int, device):
    """Block mask that drops attention BETWEEN DIFFERENT references.

    Sequence layout is ``[text | ref_0 .. ref_{n-1} | target]``. Each reference is a coherent image
    and may attend within itself, to the text and to the target; the target and text attend to
    everything. What is dropped is one reference attending to a *different* reference, which carries
    no meaning -- references are independent conditioning -- and which grows to dominate the
    attention matrix as references are added.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    # Segment id per position: 0 = text, 1 = target (and padding, which the key mask handles),
    # 2 + i = reference i.
    seg = torch.ones(total, dtype=torch.int32, device=device)
    seg[:txt_len] = 0
    pos = txt_len
    for i, rl in enumerate(ref_lens):
        seg[pos: pos + rl] = 2 + i
        pos += rl

    def mask_mod(b, h, qi, ki):
        sq, sk = seg[qi], seg[ki]
        return (~((sq >= 2) & (sk >= 2))) | (sq == sk)

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=total, KV_LEN=total, device=device)


def temb(
    t: Tensor,
    dim: int,
    period: float = 1e4,
    tfactor: float = 1e3,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(period)
        * torch.arange(half, dtype=torch.float32, device=device)
        / half
    )
    # t: (B,) -> args: (B, 1, half), so the embedding broadcasts as a per-sample vec.
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


@dataclass
class SingleMMDiTConfig:
    features: int
    tdim: int
    txtdim: int
    heads: int
    multiplier: int
    layers: int
    patch: int
    channels: int
    bias: bool = False
    theta: float = 1e3
    kvheads: int | None = None
    txtlayers: int = 1
    txtheads: int = 20
    txtkvheads: int = 20


class SimpleModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    # vec (b d)
    def forward(self, vec: Tensor):
        out = vec + rearrange(self.lin, "two d -> 1 two d")
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(6 * dim))

    # vec (b (6 d))
    def forward(self, vec: Tensor):
        out = vec + self.lin
        prescale, preshift, pregate, postscale, postshift, postgate = out.chunk(
            6, dim=-1
        )
        return prescale, preshift, pregate, postscale, postshift, postgate


class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims  # how to split the head dimension across the position axes
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat(
            [
                rope(pos[..., i], d, self.theta, self.ntk)
                for i, d in enumerate(self.axdims)
            ],
            dim=-3,
        )


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k), v


class RMSNorm(torch.nn.Module):
    def __init__(self, features: int, eps: float = 1e-05, device: torch.device = None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.zeros(features, device=device, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(
            t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0)
        )
        return t.to(dtype)


class SwiGLU(torch.nn.Module):
    def __init__(
        self, features: int, multiplier: int, bias: bool = False, multiple: int = 128
    ):
        super().__init__()

        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)

        self.gate = torch.nn.Linear(features, mlpdim, bias=bias)
        self.up = torch.nn.Linear(features, mlpdim, bias=bias)
        self.down = torch.nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(torch.nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads

        self.wq = torch.nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = torch.nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.gqa = self.heads != self.kvheads
        self.wo = torch.nn.Linear(dim, dim, bias=bias)

    def forward(
        self, qkv: Tensor, freqs: Tensor | None = None, mask: Tensor | None = None
    ) -> Tensor:
        q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)

        q, k, v = (
            rearrange(q, "B L (H D) -> B H L D", H=self.heads),
            rearrange(k, "B L (H D) -> B H L D", H=self.kvheads),
            rearrange(v, "B L (H D) -> B H L D", H=self.kvheads),
        )

        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        out = self.wo(attention(q, k, v, mask=mask, gqa=self.gqa) * F.sigmoid(gate))

        return out


class LastLayer(torch.nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = torch.nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        x = self.linear(x)
        return x


class TextFusionBlock(torch.nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.prenorm(x), mask=mask)
        x = x + self.mlp(self.postnorm(x))

        return x


class TextFusionTransformer(torch.nn.Module):
    # num_txt_layers is the number of selected encoder hidden-state layers fed in
    # (projected down to 1), NOT the transformer depth — that's fixed at 2 + 2 blocks.
    def __init__(
        self,
        num_txt_layers: int,
        txt_dim: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.layerwise_blocks = torch.nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )
        self.projector = torch.nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = torch.nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x)
        x = x.squeeze(-1)

        for block in self.refiner_blocks:
            x = block(x, mask=mask)

        return x


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(
        self, x: Tensor, vec: Tensor, freqs: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn(
            (1 + prescale) * self.prenorm(x) + preshift, freqs, mask
        )
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)

        return x


class SingleStreamDiT(nn.Module):
    def __init__(self, config: SingleMMDiTConfig):
        super().__init__()
        self.config = config
        # Attention-cost switches, both opt-in so the default path stays byte-identical.
        # ``pad_to_multiple = 0`` drops the sequence padding (and with it the key-padding mask);
        # ``skip_ref_cross_attention`` routes multi-reference batches through a block-sparse mask.
        self.pad_to_multiple = 256
        self.skip_ref_cross_attention = False

        headdim = config.features // config.heads
        axes = [
            headdim - 12 * (headdim // 16),
            6 * (headdim // 16),
            6 * (headdim // 16),
        ]
        assert sum(axes) == headdim, f"sum(axes) = {sum(axes)}, headdim = {headdim}"
        assert all(a % 2 == 0 for a in axes), f"axes = {axes}"

        self.posemb = PositionalEncoding(
            config.features, axes, theta=config.theta, ntk=1.0
        )
        self.first = nn.Linear(
            config.channels * config.patch**2, config.features, bias=True
        )

        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    config.features,
                    config.heads,
                    config.multiplier,
                    config.bias,
                    config.kvheads,
                )
                for _ in range(config.layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(config.tdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.txtfusion = TextFusionTransformer(
            config.txtlayers,
            config.txtdim,
            config.txtheads,
            config.multiplier,
            config.bias,
            config.txtkvheads,
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(config.txtdim),
            nn.Linear(config.txtdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.last = LastLayer(config.features, config.patch, config.channels)

        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features * 6)
        )

        # Set True by the trainer for full fine-tune memory savings; inference leaves it False.
        self.gradient_checkpointing = False
        # -1 checkpoints every transformer block. A smaller non-negative value checkpoints only
        # that many leading blocks, spending otherwise-free VRAM to avoid recompute losslessly.
        self.gradient_checkpointing_blocks = -1

        # Block-swap state (configured by enable_block_swap): the deepest blocks live on CPU and
        # page to the GPU only while in use. Empty set = disabled (zero overhead in forward).
        self._swap_blocks: set[int] = set()
        self._swap_device: torch.device | None = None
        self._swap_skip_trainable = True

    def compile_blocks(self, mode: str = "default", *, dynamic: bool = False) -> int:
        """Compile transformer block callables without changing registered module/state keys."""
        if self._swap_blocks:
            raise RuntimeError("block compilation must be configured before block swap")
        for block in self.blocks:
            block._compiled_forward = torch.compile(
                block.forward, mode=mode, fullgraph=True, dynamic=bool(dynamic))
        return len(self.blocks)

    def enable_block_swap(self, num_blocks: int, device, *, skip_trainable: bool = True) -> list[int]:
        """Offload the ``num_blocks`` deepest transformer blocks to CPU, paging each to ``device``
        for its forward/backward and back to CPU afterward. Trades DiT-resident VRAM for per-step
        host<->device copies. Pair with gradient checkpointing: the backward recompute re-pages the
        block in via the forward pre-hook, so the weights are present exactly when needed.

        ``skip_trainable=True`` (the LoRA path) pages only frozen base weights, keeping trainable
        params (e.g. LoRA adapters) GPU-resident. ``skip_trainable=False`` pages every param (FFT —
        correct but slow). Returns the indices marked for swapping.
        """
        n = max(0, min(int(num_blocks), len(self.blocks)))
        self._swap_device = torch.device(device)
        self._swap_skip_trainable = bool(skip_trainable)
        self._swap_blocks = set(range(len(self.blocks) - n, len(self.blocks)))
        for i in self._swap_blocks:
            block = self.blocks[i]
            self._swap_move(block, torch.device("cpu"))   # park on CPU now
            block.register_forward_pre_hook(self._swap_pre_hook)
        return sorted(self._swap_blocks)

    def _swap_move(self, block: nn.Module, device: torch.device) -> None:
        skip = self._swap_skip_trainable
        for p in block.parameters():
            if skip and p.requires_grad:        # keep trainable params (LoRA adapters) resident
                continue
            p.data = p.data.to(device, non_blocking=True)

    def _swap_pre_hook(self, module, args):
        # Fires for both the real forward and the checkpoint recompute -> weights on GPU when used.
        self._swap_move(module, self._swap_device)

    def _swap_offload(self, block: nn.Module) -> None:
        self._swap_move(block, torch.device("cpu"))

    def _mk_post_bwd_offload(self, block: nn.Module):
        # Tensor hook on a block's input: fires after that block's backward -> safe to page out.
        def hook(grad):
            self._swap_offload(block)
            return None
        return hook

    def forward(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
        attn_mask_override: Tensor | None = None,
        txt_attn_override: Tensor | None = None,
        ref_len: int = 0,
        ref_lens: list[int] | None = None,
    ) -> Tensor:
        # ``attn_mask_override`` / ``txt_attn_override`` are OPT-IN regional-attention masks
        # (bool, (B,1,L,L) / (B,1,txtlen,txtlen)) ANDed onto the default key-padding mask to route
        # image regions to their own text segment. Both default None -> the standard path below is
        # byte-identical; only sampling.sample_regions sets them.
        #
        # ``ref_len`` > 0 = in-context edit: the FIRST ``ref_len`` image tokens are CLEAN reference
        # latents (packed by ``edit_training_step`` / ``edit_sample`` as ``[refs, target]``). They get
        # a t=0 timestep modulation so the DiT reads them as clean conditioning (Kontext-style), while
        # text + the noised target keep the sampled ``t``. Default 0 -> single-``t`` path, byte-identical
        # (so t2i / style are untouched).
        img = self.first(img)
        t_hidden = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t_hidden)

        txtmask = _mask(mask[:, : context.shape[1]])
        if txt_attn_override is not None:
            txtmask = txtmask & txt_attn_override

        context = self.txtfusion(context, mask=txtmask)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)

        # Pad the combined sequence to a multiple of ``pad_to_multiple`` to stabilize compiled
        # kernel shapes. Set it to 0 to avoid computing padded tokens. If padding remains, `_mask`
        # supplies a compact key-only broadcast rather than materialising an L-by-L tensor.
        fulllen = combined.shape[1]
        _padlen = (-fulllen) % self.pad_to_multiple if self.pad_to_multiple else 0
        if _padlen > 0:
            combined = F.pad(combined, (0, 0, 0, _padlen))
            mask = F.pad(mask, (0, _padlen), value=False)
            pos = F.pad(pos, (0, 0, 0, _padlen))

        # An all-true mask carries no information; passing None instead lets the attention kernel
        # take its unmasked path. ``.all()`` costs one host sync per forward, negligible next to
        # the mask it avoids materialising.
        informative = attn_mask_override is not None or not bool(mask.all())
        if not informative and self.skip_ref_cross_attention and ref_len and ref_lens:
            mask = ref_block_mask(txtlen, ref_lens, imglen - ref_len,
                                  combined.shape[1], combined.device)
        elif not informative:
            mask = None
        else:
            mask = _mask(mask)
            if attn_mask_override is not None:
                if _padlen > 0:
                    attn_mask_override = F.pad(attn_mask_override, (0, _padlen, 0, _padlen),
                                               value=False)
                mask = mask & attn_mask_override

        freqs = self.posemb(pos)

        # Per-token timestep modulation for in-context edit: clean reference image tokens see t=0.
        # ``tvec`` broadcasts (B,1,6f) across all tokens by default; here we expand to (B,L,6f) and
        # overwrite the reference band [txtlen : txtlen+ref_len] with the t=0 projection. The block's
        # DoubleSharedModulation consumes (B,L,6f) unchanged (adds its bias, chunks on dim=-1). Trailing
        # pad tokens keep ``t`` — irrelevant (masked as keys, sliced off the output).
        if ref_len:
            B = combined.shape[0]
            t0 = torch.zeros_like(t)
            tvec0 = self.tproj(self.tmlp(temb(t0, self.config.tdim, device=img.device, dtype=img.dtype)))
            tok = tvec.expand(B, combined.shape[1], tvec.shape[-1]).contiguous()
            tok[:, txtlen: txtlen + ref_len, :] = tvec0.expand(B, ref_len, tvec0.shape[-1])
            tvec = tok

        swap = self._swap_blocks
        checkpoint_blocks = self.gradient_checkpointing_blocks
        if checkpoint_blocks < 0:
            checkpoint_blocks = len(self.blocks)
        for i, block in enumerate(self.blocks):
            swapped = i in swap
            # Sampling/preview stays eager: inference has different grad/mode/shape guards and
            # compiling it would add graphs to the long-lived training cache for no step-time gain.
            block_forward = (getattr(block, "_compiled_forward", block)
                             if self.training else block)
            if self.gradient_checkpointing and self.training and i < checkpoint_blocks:
                inp = combined
                combined = torch.utils.checkpoint.checkpoint(
                    block_forward, inp, tvec, freqs, mask, use_reentrant=False
                )
                if swapped:
                    # Real forward done; the checkpoint recompute re-pages the block in via the
                    # pre-hook. Page out after the block's backward (hook on its input grad).
                    self._swap_offload(block)
                    if inp.requires_grad:
                        inp.register_hook(self._mk_post_bwd_offload(block))
            else:
                inp = combined
                combined = block_forward(inp, tvec, freqs, mask)
                if swapped:
                    if not self.training:
                        self._swap_offload(block)                 # inference: free immediately
                    elif inp.requires_grad:
                        # No recompute here: keep weights resident through backward, free after.
                        inp.register_hook(self._mk_post_bwd_offload(block))

        final = self.last(combined, t_hidden)
        output = final[:, txtlen : txtlen + imglen, :]

        return output
