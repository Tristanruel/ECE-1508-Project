"""how well does x match the closest known protocol? """
from __future__ import annotations

import numpy as np
import torch

from rfgen.models import (
    ConditionalDDPM,
    ConditionalRealNVP,
    ConditionalVAE,
    SpectrogramCNN,
)
from rfgen.train.common import load_best
from rfgen.utils import get_device


def build_models(cfg: dict, device=None) -> dict:
    n_classes = len(cfg["signal"]["classes"])
    s = cfg["spectrogram"]
    vae = ConditionalVAE(
        n_classes=n_classes,
        latent_dim=int(cfg["models"]["vae"]["latent_dim"]),
        base_ch=int(cfg["models"]["vae"]["base_ch"]),
        beta=float(cfg["models"]["vae"]["beta"]),
    )
    flow = ConditionalRealNVP(
        n_classes=n_classes,
        img_shape=(1, int(s["n_freq"]), int(s["n_time"])),
        n_coupling=int(cfg["models"]["flow"]["n_coupling"]),
        hidden=int(cfg["models"]["flow"]["hidden"]),
    )
    ddpm = ConditionalDDPM(
        n_classes=n_classes,
        base_ch=int(cfg["models"]["diffusion"]["base_ch"]),
        timesteps=int(cfg["models"]["diffusion"]["timesteps"]),
        beta_start=float(cfg["models"]["diffusion"]["beta_start"]),
        beta_end=float(cfg["models"]["diffusion"]["beta_end"]),
    )
    return {"vae": vae, "flow": flow, "diffusion": ddpm}


def load_all(cfg: dict, device=None):
    device = device or get_device()
    models = build_models(cfg, device)
    for tag, model in models.items():
        load_best(model, cfg, tag, device)
    clf = SpectrogramCNN(
        n_classes=len(cfg["signal"]["classes"]),
        base_ch=int(cfg["models"]["classifier"]["base_ch"]),
    )
    load_best(clf, cfg, "classifier", device)
    return models, clf


def load_mamba(cfg: dict, device=None):
    from rfgen.train.train_mamba import build_mamba
    device = device or get_device()
    model = build_mamba(cfg)
    load_best(model, cfg, "mamba", device)
    return model


@torch.no_grad()
def min_class_score_iq(model, iq2: np.ndarray, n_classes: int, device, batch: int = 128) -> np.ndarray:
    xt = torch.from_numpy(np.ascontiguousarray(iq2)).float()
    out = np.empty(xt.shape[0], dtype=np.float64)
    for i in range(0, xt.shape[0], batch):
        xb = xt[i : i + batch].to(device)
        per_class = []
        for c in range(n_classes):
            yb = torch.full((xb.shape[0],), c, device=device, dtype=torch.long)
            per_class.append(model.nll_iq(xb, yb))
        out[i : i + xb.shape[0]] = torch.stack(per_class, 0).min(0).values.cpu().numpy()
    return out


def iq_complex_to_channels(iq: np.ndarray) -> np.ndarray:
    return np.stack([iq.real, iq.imag], axis=1).astype(np.float32)


SCORE_FNS = {
    "vae": lambda m, xb, yb: m.neg_elbo(xb, yb),
    "flow": lambda m, xb, yb: m.nll(xb, yb),
    "diffusion": lambda m, xb, yb: m.denoising_error(xb, yb),
}


@torch.no_grad()
def min_class_score(model, tag: str, x: np.ndarray, n_classes: int, device, batch: int = 256) -> np.ndarray:
    score_fn = SCORE_FNS[tag]
    xt = torch.from_numpy(np.ascontiguousarray(x)).float()
    out = np.empty(xt.shape[0], dtype=np.float64)
    for i in range(0, xt.shape[0], batch):
        xb = xt[i : i + batch].to(device)
        per_class = []
        for c in range(n_classes):
            yb = torch.full((xb.shape[0],), c, device=device, dtype=torch.long)
            per_class.append(score_fn(model, xb, yb))
        stacked = torch.stack(per_class, dim=0)
        out[i : i + xb.shape[0]] = stacked.min(dim=0).values.cpu().numpy()
    return out
