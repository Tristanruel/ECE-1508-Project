"""PyTorch data loading from the HDF5 dataset produced by build_dataset"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rfgen.utils import resolve_path


def dataset_path(cfg: dict) -> Path:
    return resolve_path(cfg["data"]["out_dir"]) / "dataset.h5"


class SpectrogramDataset(Dataset):

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(np.ascontiguousarray(x)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def _read(path, group, key):
    with h5py.File(path, "r") as f:
        return f[group][key][...]


def load_split(cfg: dict, split: str) -> SpectrogramDataset:
    path = dataset_path(cfg)
    with h5py.File(path, "r") as f:
        x = f[split]["x"][...]
        y = f[split]["y"][...]
    return SpectrogramDataset(x, y)


def load_split_iq(cfg: dict, split: str):
    path = dataset_path(cfg)
    with h5py.File(path, "r") as f:
        return f[split]["iq"][...], f[split]["y"][...]


def load_anomalies(cfg: dict):
    path = dataset_path(cfg)
    with h5py.File(path, "r") as f:
        x = f["anom"]["x"][...]
        y_type = f["anom"]["y_type"][...]
        types = json.loads(f.attrs["anomaly_types"])
    return x, y_type, types


def load_anomalies_iq(cfg: dict):
    path = dataset_path(cfg)
    with h5py.File(path, "r") as f:
        return f["anom"]["iq"][...], f["anom"]["y_type"][...]


def class_names(cfg: dict) -> list[str]:
    with h5py.File(dataset_path(cfg), "r") as f:
        return json.loads(f.attrs["class_names"])


def make_loaders(cfg: dict, batch_size: int | None = None):
    bs = int(batch_size or cfg["train"]["batch_size"])
    train = load_split(cfg, "train")
    val = load_split(cfg, "val")
    train_loader = DataLoader(train, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=bs, shuffle=False)
    return train_loader, val_loader


class IQDataset(Dataset):

    def __init__(self, iq: np.ndarray, y: np.ndarray):
        # iq: [N_examples, N] complex -> [N_examples, 2, N] float32 (I, Q).
        iq = np.ascontiguousarray(iq)
        self.x = torch.from_numpy(np.stack([iq.real, iq.imag], axis=1)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def make_iq_loaders(cfg: dict, batch_size: int | None = None):
    bs = int(batch_size or cfg["train"].get("batch_size_mamba", cfg["train"]["batch_size"]))
    iq_tr, y_tr = load_split_iq(cfg, "train")
    iq_va, y_va = load_split_iq(cfg, "val")
    train = IQDataset(iq_tr, y_tr)
    val = IQDataset(iq_va, y_va)
    return (DataLoader(train, batch_size=bs, shuffle=True, drop_last=True),
            DataLoader(val, batch_size=bs, shuffle=False))
