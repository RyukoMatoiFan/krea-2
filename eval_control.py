"""Metric-driven checkpoint retention, log-spaced eval schedules, and patience early-stopping.

Addresses two failure modes of long training runs:

* **Rotation by step deletes the best model.** ``save_ckpt(..., keep_last=N)`` keeps the *newest* N
  checkpoints. If quality peaks mid-run and then drifts, the peak is rotated away — so a stop rule
  that *identifies* the best checkpoint is useless, because that checkpoint no longer exists.
  ``BestTracker`` keeps the best N *by metric*, on a filename namespace the step-rotation never
  touches (``dit_best_step*`` vs ``dit_step*``).

* **A uniform eval stride measures the wrong region.** Evaluating every N steps spends most of the
  budget on the flat tail and can miss the early climb entirely, leaving no evidence about when the
  model actually stopped improving. ``parse_eval_steps`` produces a log-spaced schedule instead:
  dense early, sparse late.

The scorer is pluggable (``module:function``) so the trainer stays task-agnostic — the metric that
defines "better" for an edit model is not the one an ordinary t2i run would use.
"""
from __future__ import annotations

import glob
import importlib
import json
import math
import os


# --------------------------------------------------------------------------- #
# Eval schedule
# --------------------------------------------------------------------------- #
def parse_eval_steps(spec, total, *, start_default=500, ratio_default=2.0):
    """Expand an eval-schedule spec into a sorted list of step numbers.

    ``""``                      -> ``[]`` (disabled)
    ``"log"``                   -> geometric from 500, x2  (500, 1000, 2000, 4000, …)
    ``"log:1.6"``               -> geometric, ratio 1.6
    ``"log:1.6:250"``           -> geometric, ratio 1.6, first eval at 250
    ``"500,1000,2000"``         -> explicit list
    ``"every:5000"``            -> uniform stride

    ``total`` is always included so the final model is evaluated.
    """
    spec = (spec or "").strip()
    if not spec:
        return []
    total = int(total)
    if spec.lower().startswith("every:"):
        stride = max(1, int(float(spec.split(":", 1)[1])))
        out = list(range(stride, total + 1, stride))
    elif spec.lower().startswith("log"):
        parts = spec.split(":")
        ratio = float(parts[1]) if len(parts) > 1 and parts[1] else ratio_default
        start = int(float(parts[2])) if len(parts) > 2 and parts[2] else start_default
        ratio = max(1.05, ratio)
        out, s = [], max(1, start)
        while s < total:
            out.append(int(round(s)))
            s *= ratio
    else:
        out = [int(float(x)) for x in spec.replace(";", ",").split(",") if x.strip()]
    out = [s for s in out if 0 < s <= total]
    if total > 0:
        out.append(total)
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# Pluggable scorer
# --------------------------------------------------------------------------- #
def load_scorer(spec):
    """``"module:function"`` -> callable. Returns None when ``spec`` is empty.

    The callable is invoked with keyword arguments only (currently ``sheet_path``, ``examples``,
    ``step``) and must return a float; unknown kwargs should be absorbed with ``**_`` so the
    calling convention can grow without breaking existing scorers.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    mod_name, _, fn_name = spec.partition(":")
    if not fn_name:
        raise ValueError(f"eval_scorer must be 'module:function', got {spec!r}")
    return getattr(importlib.import_module(mod_name), fn_name)


# --------------------------------------------------------------------------- #
# Best-N retention + early stopping
# --------------------------------------------------------------------------- #
def parse_criteria(spec):
    """``"correct:max:0.005,quality:max"`` -> [(name, mode, tie_tol), ...], ordered by priority.

    Lexicographic: the first criterion decides unless two values are within its tie tolerance, in
    which case the next criterion breaks the tie. This is NOT a weighted sum on purpose -- a sum
    lets a model trade a broken primary objective against a prettier picture.
    """
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        name = bits[0].strip()
        mode = (bits[1].strip().lower() if len(bits) > 1 else "max")
        tol = float(bits[2]) if len(bits) > 2 and bits[2].strip() else 0.0
        out.append((name, "min" if mode.startswith("min") else "max", tol))
    return out


def parse_guardrails(spec):
    """``"single_ref:min:0.80"`` -> [(name, mode, threshold), ...].

    A guardrail is a HARD gate, not a term: a checkpoint failing any guardrail can never be "best",
    regardless of how good the ranked criteria are. Used for "adding composition must not degrade
    single-reference editing".
    """
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) < 3:
            raise ValueError(f"guardrail needs name:mode:threshold, got {part!r}")
        out.append((bits[0].strip(), "min" if bits[1].strip().lower().startswith("min") else "max",
                    float(bits[2])))
    return out


class BestTracker:
    """Keep the best ``keep`` checkpoints by metric and decide when to stop.

    ``mode='max'`` treats larger values as better (quality scores); ``'min'`` is for losses.
    ``min_delta`` is the smallest improvement that *counts* as improvement — set it to the smallest
    difference you would actually act on, so noise cannot reset the patience counter.

    State is mirrored to ``best.json`` in the checkpoint dir so a resumed run does not forget which
    checkpoints are protected (and does not re-stop immediately).
    """

    def __init__(self, ckpt_dir, keep=0, *, mode="max", min_delta=0.0, patience=0,
                 fingerprint=None, criteria=None, guardrails=None,
                 patterns=("dit_best_{tag}.safetensors", "te_best_{tag}.safetensors",
                           "dit_ema_best_{tag}.safetensors")):
        self.dir = ckpt_dir
        self.keep = int(keep)
        self.mode = "min" if str(mode).lower().startswith("min") else "max"
        # Multi-criterion selection. A scalar score is wrapped as {"score": v} so the scalar path is
        # unchanged; `criteria` then defaults to a single entry using `mode`.
        self.criteria = list(criteria) if criteria else [("score", self.mode, 0.0)]
        self.guardrails = list(guardrails or [])
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        # Identifies the experiment this state belongs to (metric, manifest, seed, arm, ...).
        # Reusing a ckpt_dir across ablation arms would otherwise silently mix rankings.
        self.fingerprint = fingerprint
        self.patterns = tuple(patterns)
        self.entries = []          # [{"step": int, "value": float}] sorted best-first
        self.best_value = None
        self.since_improved = 0
        self.stale = False         # True when persisted state was discarded (fingerprint mismatch)
        self._load()

    # -- persistence ------------------------------------------------------- #
    @property
    def _path(self):
        return os.path.join(self.dir, "best.json")

    def _load(self):
        try:
            with open(self._path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            return
        # Refuse state from a different experiment: a reused ckpt_dir with a changed metric,
        # manifest, seed or ablation arm would otherwise contaminate ranking AND patience.
        prev_fp = d.get("fingerprint")
        # A state file with NO fingerprint cannot be shown to belong to this experiment, so when we
        # have one it is treated as stale rather than silently adopted (it may be from another arm).
        if self.fingerprint is not None and prev_fp != self.fingerprint:
            self.stale = True
            return
        if d.get("mode") and d["mode"] != self.mode:
            self.stale = True      # direction flipped -> old comparisons are meaningless
            return
        self.entries = list(d.get("entries") or [])
        self.best_value = d.get("best_value")
        self.since_improved = int(d.get("since_improved") or 0)

    def _save(self):
        os.makedirs(self.dir, exist_ok=True)
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"mode": self.mode, "keep": self.keep, "best_value": self.best_value,
                       "since_improved": self.since_improved, "entries": self.entries,
                       "fingerprint": self.fingerprint}, f, indent=1)
        os.replace(tmp, self._path)

    def drop_after(self, step):
        """Forget evals from *after* ``step`` (called on resume).

        Without this, resuming an older checkpoint replays training while carrying patience/best
        state from that checkpoint's future — which can stop the run instantly or rank a step
        against a model state it never had.
        """
        step = int(step)
        before = len(self.entries)
        kept = [e for e in self.entries if int(e["step"]) <= step]
        if len(kept) != before:
            for e in self.entries:
                if int(e["step"]) > step:
                    self._remove(e["step"])
            self.entries = kept
            self.best_value = (min(kept, key=self._rank_key)["value"] if kept else None)
            self.since_improved = 0
            self._save()
        return before - len(kept)

    # -- comparison -------------------------------------------------------- #
    @staticmethod
    def _as_scores(value):
        """Accept a scalar or a dict of named metrics; normalise to a dict."""
        if isinstance(value, dict):
            return {k: float(v) for k, v in value.items()
                    if isinstance(v, (int, float)) and math.isfinite(float(v))}
        return {"score": float(value)}

    def passes_guardrails(self, scores):
        """A checkpoint failing ANY guardrail is ineligible to be 'best', however good its ranks."""
        for name, mode, thr in self.guardrails:
            v = scores.get(name)
            if v is None:
                return False, f"{name} missing"
            if (mode == "min" and v < thr) or (mode == "max" and v > thr):
                return False, f"{name}={v:.4f} violates {mode} {thr}"
        return True, ""

    def _better(self, a, b):
        """Lexicographic comparison of two score dicts (or scalars). ``b=None`` -> True.

        The FIRST criterion must improve by more than ``min_delta`` to count; later criteria only
        break ties, where a tie is 'within that criterion's tie tolerance'.
        """
        if b is None:
            return True
        sa, sb = self._as_scores(a), self._as_scores(b)
        for i, (name, mode, tol) in enumerate(self.criteria):
            va, vb = sa.get(name), sb.get(name)
            if va is None or vb is None:
                continue
            delta = self.min_delta if i == 0 else tol
            if mode == "max":
                if va > vb + delta:
                    return True
                if vb > va + delta:
                    return False
            else:
                if va < vb - delta:
                    return True
                if vb < va - delta:
                    return False
            # within tolerance -> tied on this criterion, fall through to the next
        return False

    def _rank_key(self, e):
        """Sort key implementing the same lexicographic order (best first)."""
        s = self._as_scores(e.get("scores", e.get("value")))
        return tuple((-s.get(n, float("-inf")) if m == "max" else s.get(n, float("inf")))
                     for n, m, _ in self.criteria)

    # -- main API ---------------------------------------------------------- #
    def update(self, step, value, save_fn=None):
        """Record an eval result. Optionally persist a protected best checkpoint.

        ``save_fn(tag)`` should write the checkpoint files matching ``patterns`` and is called only
        when this step makes the top-``keep``. Returns a dict describing what happened.

        A non-finite score is REJECTED outright and leaves all state untouched. Accepting one would
        be silently catastrophic: ``_better(nan, None)`` is True, so it would write a bogus "best"
        checkpoint, and every later comparison against NaN is False, so patience would climb forever
        and trigger a spurious early stop on a multi-day run.
        """
        scores = self._as_scores(value) if isinstance(value, dict) else None
        if scores is None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = float("nan")
            if not math.isfinite(value):
                return {"improved": False, "saved_best": False, "should_stop": False,
                        "invalid": True, "best_value": self.best_value,
                        "since_improved": self.since_improved}
            scores = {"score": value}
        elif not scores:
            return {"improved": False, "saved_best": False, "should_stop": False,
                    "invalid": True, "best_value": self.best_value,
                    "since_improved": self.since_improved}

        # Guardrails gate ELIGIBILITY, not rank: a violating checkpoint is never "best" and never
        # counts as an improvement, but it is still a valid measurement, so patience advances.
        ok, why = self.passes_guardrails(scores)
        if not ok:
            self.since_improved += 1
            self._save()
            return {"improved": False, "saved_best": False, "guardrail_failed": why,
                    "should_stop": bool(self.patience) and self.since_improved >= self.patience,
                    "best_value": self.best_value, "since_improved": self.since_improved}

        improved = self._better(scores, self.best_value)
        if improved:
            self.best_value = scores if len(scores) > 1 else scores.get("score", value)
            self.since_improved = 0
        else:
            self.since_improved += 1

        saved = False
        if self.keep > 0 and save_fn is not None:
            # Drop any prior entry for this step (a replayed/resumed step must not appear twice and
            # then occupy two of the keep-N slots with the same checkpoint).
            cand = [e for e in self.entries if int(e["step"]) != int(step)]
            cand.append({"step": int(step), "value": scores.get("score", 0.0), "scores": scores})
            cand.sort(key=self._rank_key)
            kept = cand[: self.keep]
            if any(e["step"] == int(step) for e in kept):
                save_fn(f"step{int(step):06d}")
                saved = True
                dropped = [e for e in cand[self.keep:] if e["step"] != int(step)]
                for e in dropped:
                    self._remove(e["step"])
                self.entries = kept
            # else: this step didn't make the cut -> nothing written, nothing to prune
        self._save()

        should_stop = bool(self.patience) and self.since_improved >= self.patience
        return {"improved": improved, "saved_best": saved, "should_stop": should_stop,
                "best_value": self.best_value, "since_improved": self.since_improved}

    def _remove(self, step):
        tag = f"step{int(step):06d}"
        for pat in self.patterns:
            for p in glob.glob(os.path.join(self.dir, pat.format(tag=tag))):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def summary(self):
        """Report the TRUE best (from the retained entries), not the patience reference.

        These differ on purpose: ``best_value`` here is the best score actually seen, while
        ``patience_ref`` only advances on gains larger than ``min_delta`` (so sub-delta noise cannot
        keep resetting the counter). Reporting the reference as "best" would print a worse number
        than the checkpoint we actually kept.
        """
        best = min(self.entries, key=self._rank_key) if self.entries else None
        return {"best_step": best["step"] if best else None,
                "best_value": best["value"] if best else self.best_value,
                "patience_ref": self.best_value,
                "kept": [(e["step"], round(e["value"], 5)) for e in
                         sorted(self.entries, key=self._rank_key)]}
