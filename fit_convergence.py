#!/usr/bin/env python
"""Estimate how much of the achievable improvement a run has already captured.

Long runs are usually stopped by budget or by patience, neither of which answers the question the
budget actually poses: *how much better would this get if it kept going?* A held-out loss that is
still falling does not say whether the remaining descent is worth days of compute. Fitting the
curve does, because flow-matching validation loss over a long run is well described by a power law
approaching an irreducible floor::

    val(s) = a * s**-b + c

``c`` is the floor the data and objective impose, which no amount of further training removes.
Everything above it is the improvement still on the table, so the fraction captured by step ``s``
is the part of that gap already closed.

The useful identity is that the fraction does not depend on ``a`` or ``c`` at all::

    captured(s) = 1 - (s / s0)**-b        =>        s(f) = s0 * (1 - f)**(-1/b)

so the step at which a target fraction is reached is set by the exponent and the reference step
alone. That is convenient, and it is also the catch: the answer is *exponentially* sensitive to
``b``. At b=1.0 reaching 90% takes 10x the reference step; at b=0.5 it takes 100x. A curve fit
that pins ``b`` to within a few percent still leaves the target step uncertain by a large factor,
which is why this reports a bootstrap interval and not a single number. Treat a point estimate
without its interval as meaningless.

    python fit_convergence.py runs/<run>/metrics.jsonl
    python fit_convergence.py runs/<run>/metrics.jsonl --target 0.9 --cost-per-1k 0.955
"""
from __future__ import annotations

import argparse
import json
import math
import random


def read_series(path: str, key: str) -> list[tuple[float, float]]:
    """(step, value) pairs for ``key``, in step order, one per step."""
    seen: dict[int, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if key in rec and rec[key] is not None and "step" in rec:
                try:
                    v = float(rec[key])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    seen[int(rec["step"])] = v
    return sorted(seen.items())


def fit_b(points: list[tuple[float, float]], b_lo=0.02, b_hi=3.0, steps=400):
    """Best (b, a, c, sse) for val = a*s**-b + c.

    For a FIXED b the model is linear in (a, c), so each candidate exponent is solved exactly by
    least squares and only the one-dimensional search over b is done numerically. That is both
    faster and far more stable than a joint non-linear fit, which on noisy data happily wanders
    into a<0 or c<0 and reports a curve that rises.
    """
    best = None
    for i in range(steps + 1):
        b = b_lo * (b_hi / b_lo) ** (i / steps)          # geometric sweep
        xs = [s ** -b for s, _ in points]
        ys = [v for _, v in points]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 0:
            continue
        a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        c = my - a * mx
        if a <= 0:                                        # a<=0 means "not descending" -> reject
            continue
        sse = sum((y - (a * x + c)) ** 2 for x, y in zip(xs, ys))
        if best is None or sse < best[3]:
            best = (b, a, c, sse)
    return best


def step_for_fraction(s0: float, b: float, f: float) -> float:
    return s0 * (1.0 - f) ** (-1.0 / b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", help="metrics.jsonl written by the trainer")
    ap.add_argument("--key", default="val_loss")
    ap.add_argument("--target", type=float, default=0.9, help="fraction of achievable improvement")
    ap.add_argument("--min-step", type=int, default=0, help="ignore evals before this step")
    ap.add_argument("--cost-per-1k", type=float, default=0.0, help="$ per 1000 steps, for context")
    ap.add_argument("--boot", type=int, default=400, help="bootstrap resamples for the interval")
    args = ap.parse_args()

    pts = [(s, v) for s, v in read_series(args.metrics, args.key) if s >= max(1, args.min_step)]
    if len(pts) < 6:
        raise SystemExit(f"need at least 6 '{args.key}' points to fit, found {len(pts)}")

    fit = fit_b(pts)
    if fit is None:
        raise SystemExit("no descending power law fits this series -- the curve is flat or rising")
    b, a, c, sse = fit
    ss_tot = sum((v - sum(y for _, y in pts) / len(pts)) ** 2 for _, v in pts)
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
    s0, first_val = pts[0]
    s_last, last_val = pts[-1]

    print(f"points        : {len(pts)}  (steps {int(s0):,} .. {int(s_last):,})")
    print(f"fit           : val = {a:.4g} * s**-{b:.4g} + {c:.5g}      R^2 = {r2:.4f}")
    print(f"floor c       : {c:.5g}   (irreducible; improvement above it is what is on offer)")

    captured = 1.0 - (s_last / s0) ** -b
    print(f"captured by {int(s_last):,}: {100 * captured:.1f}% of the achievable improvement")

    # Bootstrap the exponent, because the target step is exponential in it.
    rng = random.Random(0)
    bs = []
    for _ in range(args.boot):
        sample = [pts[rng.randrange(len(pts))] for _ in range(len(pts))]
        sample.sort()
        f2 = fit_b(sample, steps=160)
        if f2 is not None:
            bs.append(f2[0])
    tgt = step_for_fraction(s0, b, args.target)
    line = f"\nstep for {100 * args.target:.0f}% : {tgt:,.0f}"
    if bs:
        bs.sort()
        lo_b, hi_b = bs[int(0.05 * len(bs))], bs[int(0.95 * len(bs)) - 1]
        # A LARGER exponent reaches the target sooner, so the interval inverts.
        s_lo = step_for_fraction(s0, hi_b, args.target)
        s_hi = step_for_fraction(s0, lo_b, args.target)
        line += f"   90% CI [{s_lo:,.0f} .. {s_hi:,.0f}]   (b in [{lo_b:.3g}, {hi_b:.3g}])"
    print(line)

    if args.cost_per_1k:
        print(f"cost at target : ${tgt / 1000 * args.cost_per_1k:,.0f}"
              + (f"   CI [${s_lo / 1000 * args.cost_per_1k:,.0f} .. "
                 f"${s_hi / 1000 * args.cost_per_1k:,.0f}]" if bs else ""))

    if tgt > 3 * s_last:
        print(f"\nWARNING: the target lies {tgt / s_last:.1f}x beyond the last observed step. This is"
              "\n         extrapolation, not measurement -- the fit constrains the region it has seen."
              "\n         Re-run this as the curve extends before spending against it.")
    print("\nReminder: this fits a flow-matching loss, which is only a proxy for sample quality."
          "\nPair it with rendered A/B comparisons before acting on the number.")


if __name__ == "__main__":
    main()
