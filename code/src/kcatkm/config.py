"""YAML configs with inheritance and command-line overrides.

    # configs/lgbm_morgan.yaml
    base: base.yaml          # deep-merged underneath this file
    name: lgbm_morgan
    model: {name: lgbm, params: {n_estimators: 1000}}

Override anything from the CLI:  --set model.params.learning_rate=0.05 split.n_folds=3
"""
import copy
import hashlib
import json
from pathlib import Path

import yaml

from . import paths


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path) -> dict:
    path = Path(path)
    if not path.exists() and (paths.CONFIGS / path).exists():
        path = paths.CONFIGS / path
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base", None)
    if base:
        parent = load(path.parent / base)
        parent.pop("name", None)
        parent.pop("notes", None)
        # Switching model type (e.g. lgbm -> mlp) starts the model section fresh:
        # one model's hyper-parameters are meaningless (or invalid) for another.
        child_model = (cfg.get("model") or {}).get("name")
        if child_model and child_model != (parent.get("model") or {}).get("name"):
            parent["model"] = {}
        cfg = deep_merge(parent, cfg)
    cfg.setdefault("name", path.stem)
    return cfg


def apply_overrides(cfg: dict, overrides) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:  # numeric parts index into lists, e.g. model.members.1.params.epochs=60
            node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
        if isinstance(node, list):
            node[int(leaf)] = yaml.safe_load(raw)
        else:
            node[leaf] = yaml.safe_load(raw)
    return cfg


def fingerprint(cfg: dict) -> str:
    """Hash of everything that affects results (ignores name/notes)."""
    relevant = {k: v for k, v in cfg.items() if k not in ("name", "notes", "tags")}
    return hashlib.sha1(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()[:12]
