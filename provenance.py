"""Record what a run ACTUALLY executed, so a finished run can be audited after the fact.

A configured setting can silently fail to take effect -- for example when the execution host runs a
stale copy of the module that forwards it -- and such a parameter is generally invisible in loss
curves, sample previews and downstream evaluation. Without a record of what actually ran, there is no
way after the fact to answer "what code and what settings did this run really use?".

So at startup a run writes a manifest containing:

* the RESOLVED config (post env-override), not the yaml on disk
* a hash of every repo module actually IMPORTED — taken from ``sys.modules``, so it reflects the
  files that were really loaded rather than a hand-maintained list that can drift
* dataset identity (cache dir, train list, counts) and environment/GPU/library versions

The digest is short enough to embed in checkpoint metadata, so a checkpoint can be traced back to
the exact code that produced it.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys


def _file_sha(path, chunk=1 << 20):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
    except OSError:
        return None
    return h.hexdigest()


def code_hashes(root=None):
    """SHA-256 of every imported module whose file lives under ``root`` (default: this file's dir).

    Reading from ``sys.modules`` is deliberate: a fixed filename list would itself go stale, and the
    failure being guarded against is precisely a file differing from what the author believes.
    """
    root = os.path.abspath(root or os.path.dirname(os.path.abspath(__file__)))
    out = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        f = os.path.abspath(f)
        if f.startswith(root + os.sep) and f.endswith(".py"):
            sha = _file_sha(f)
            if sha:
                out[os.path.relpath(f, root).replace("\\", "/")] = sha[:16]
    return dict(sorted(out.items()))


def _asdict(cfg):
    import dataclasses

    if dataclasses.is_dataclass(cfg):
        return {k: _asdict(v) for k, v in dataclasses.asdict(cfg).items()}
    if isinstance(cfg, dict):
        return {k: _asdict(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_asdict(v) for v in cfg]
    return cfg if isinstance(cfg, (str, int, float, bool, type(None))) else str(cfg)


def build_manifest(cfg, *, extra=None, root=None):
    """Everything needed to answer 'what did this run actually execute?'."""
    import torch

    codes = code_hashes(root)
    digest = hashlib.sha256(
        json.dumps({"code": codes, "config": _asdict(cfg)}, sort_keys=True).encode()).hexdigest()[:16]

    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "torch": getattr(torch, "__version__", "?")}
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    man = {"digest": digest, "config": _asdict(cfg), "code": codes, "env": env}
    # Every KREA2_* override, so the manifest shows what the launcher actually asked for.
    man["env_overrides"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith("KREA2_")}
    if extra:
        man["data"] = extra
    return man


def write_manifest(path, cfg, *, extra=None, root=None):
    man = build_manifest(cfg, extra=extra, root=root)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
    return man


def diff_manifests(a, b):
    """What changed between two runs — config keys and code files that differ."""
    out = {"config": {}, "code": {}}

    def walk(x, y, prefix=""):
        keys = set(x) | set(y)
        for k in sorted(keys):
            xa, yb = x.get(k), y.get(k)
            if isinstance(xa, dict) and isinstance(yb, dict):
                walk(xa, yb, f"{prefix}{k}.")
            elif xa != yb:
                out["config"][f"{prefix}{k}"] = [xa, yb]

    walk(a.get("config") or {}, b.get("config") or {})
    ca, cb = a.get("code") or {}, b.get("code") or {}
    for k in sorted(set(ca) | set(cb)):
        if ca.get(k) != cb.get(k):
            out["code"][k] = [ca.get(k), cb.get(k)]
    return out
