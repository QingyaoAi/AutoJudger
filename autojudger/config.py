"""Config loading: deep-merge a user YAML over packaged defaults."""

from __future__ import annotations

import copy
import os

import yaml

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "default.yaml")

_REQUIRED = ("judgment_prompt", "apis", "tasks")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(user_path: str) -> dict:
    """Load user config and merge it over defaults, validating required keys."""
    with open(_DEFAULT_PATH, encoding="utf-8") as f:
        defaults = yaml.safe_load(f) or {}
    with open(user_path, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}

    config = _deep_merge(defaults, user)

    missing = [k for k in _REQUIRED if not config.get(k)]
    if missing:
        raise ValueError(f"Missing required config field(s): {', '.join(missing)}")
    if config["mode"] not in ("pairwise", "pointwise"):
        raise ValueError(f"Invalid mode: {config['mode']!r} (expected pairwise|pointwise)")
    return config
