from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the YAML config (defaults to ``configs/default.yaml``)."""
    if path is None:
        path = REPO_ROOT / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the CUDA device when available, else CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_path(path: str | os.PathLike) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | os.PathLike) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def load_json(path: str | os.PathLike) -> Any:
    with open(resolve_path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
