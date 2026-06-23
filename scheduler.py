"""Krea 2 flow-matching timestep schedule (resolution-aware "mu" shift).

Krea 2 has no fixed schedule object; the reference sampler (``sampling.py``) applies a dynamic
timestep shift whose parameter ``mu`` is linearly interpolated by the number of IMAGE TOKENS
(``grid_h * grid_w``, NOT pixels) between ``(SHIFT_BASE_SEQ_LEN, SHIFT_BASE)`` and
``(SHIFT_MAX_SEQ_LEN, SHIFT_MAX)``, then applied to a base time ``t`` as::

    t_shifted = exp(mu) / (exp(mu) + (1 / t - 1) ** sigma)

The module exposes a callable schedule object + a resolution factory + ``make_step_intervals``.
The schedule is used for BOTH:
  * training  - draw a shifted timestep ``t = schedule(rand)``  (SD3/Flux dynamic-shift training),
  * sampling  - map the linear Euler grid ``schedule(linspace)`` to the curved time axis.

Convention (see constants.py): t in (0, 1), t=1 noise, t=0 data.
"""
import math
from dataclasses import dataclass

import torch

from constants import (
    PIXEL_PER_TOKEN,
    SHIFT_BASE,
    SHIFT_BASE_SEQ_LEN,
    SHIFT_MAX,
    SHIFT_MAX_SEQ_LEN,
    SHIFT_SIGMA,
)


def mu_for_seq_len(
    image_seq_len: int,
    base_shift: float = SHIFT_BASE,
    max_shift: float = SHIFT_MAX,
    base_seq_len: int = SHIFT_BASE_SEQ_LEN,
    max_seq_len: int = SHIFT_MAX_SEQ_LEN,
) -> float:
    """Linear interpolation of the shift ``mu`` by image-token count (no clamping, as in Krea)."""
    frac = (image_seq_len - base_seq_len) / (max_seq_len - base_seq_len)
    return base_shift + (max_shift - base_shift) * frac


@dataclass(frozen=True)
class KreaShiftSchedule:
    """Callable mapping a base time ``t`` in (0, 1) to Krea's dynamic-shifted time.

    ``mu`` is resolution-derived (see :func:`get_schedule_for_seqlen`); ``sigma`` defaults to 1.0.
    """

    mu: float
    sigma: float = SHIFT_SIGMA

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.float64).clamp(1e-6, 1.0 - 1e-6)
        em = math.exp(self.mu)
        out = em / (em + (1.0 / t - 1.0) ** self.sigma)
        return out.to(torch.float32)


def get_schedule_for_seqlen(image_seq_len: int, sigma: float = SHIFT_SIGMA, **kw) -> KreaShiftSchedule:
    """Build the schedule for a given image-token count (``grid_h * grid_w``)."""
    return KreaShiftSchedule(mu=mu_for_seq_len(image_seq_len, **kw), sigma=sigma)


def get_schedule_for_resolution(height: int, width: int, sigma: float = SHIFT_SIGMA, **kw) -> KreaShiftSchedule:
    """Convenience: derive the image-token count from pixel HxW (a token = ``PIXEL_PER_TOKEN`` px/side)."""
    seq_len = (height // PIXEL_PER_TOKEN) * (width // PIXEL_PER_TOKEN)
    return get_schedule_for_seqlen(seq_len, sigma=sigma, **kw)


def make_step_intervals(num_steps: int, *, descending: bool = True) -> torch.Tensor:
    """Linear time grid the sampler indexes, length ``num_steps + 1``.

    ``descending`` matches the reference sampler's ``linspace(1, 0, steps + 1)`` (t: noise -> data).
    """
    if descending:
        return torch.linspace(1.0, 0.0, num_steps + 1)
    return torch.linspace(0.0, 1.0, num_steps + 1)
