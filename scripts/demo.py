#!/usr/bin/env python
"""Minimal input -> output demo of the trained RF-GenLab models.

Shows, for the three protocols:
  * a real synthesised signal and its spectrogram,
  * one class-conditional sample from each generative model, and
  * the anomaly-detection verdict (flow negative log-likelihood) for a clean
    signal versus a corrupted one, with a threshold calibrated on normal data.

Requires the trained checkpoints in ``runs/`` (run ``scripts/run_all.py`` first,
or at least train the flow + classifier). Writes ``results/figures/fig_demo.png``.

    python scripts/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rfgen.dsp.generators import CLASS_NAMES
from rfgen.dsp.impairments import make_anomaly, make_normal
from rfgen.dsp.spectrogram import SpectrogramConfig, iq_to_spectrogram
from rfgen.eval.common import load_all, min_class_score
from rfgen.utils import ensure_dir, get_device, load_config


def main() -> None:
    cfg = load_config()
    device = get_device()
    fs = float(cfg["signal"]["sample_rate_hz"])
    n = int(cfg["signal"]["num_samples"])
    sc = SpectrogramConfig.from_config(cfg)
    rng = np.random.default_rng(7)
    models, clf = load_all(cfg, device)


    fig, ax = plt.subplots(4, 3, figsize=(7.5, 9.5))
    row_labels = ["Real", "VAE sample", "Flow sample", "DDPM sample"]
    for c, cls in enumerate(CLASS_NAMES):
        real = iq_to_spectrogram(make_normal(cls, n, fs, rng, cfg), sc)
        ax[0, c].imshow(real[0], origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax[0, c].set_title(cls, fontsize=12)
        y = torch.tensor([c], device=device)
        for r, tag in enumerate(["vae", "flow", "diffusion"], start=1):
            with torch.no_grad():
                samp = models[tag].sample(1, y).cpu().numpy()[0, 0]
            ax[r, c].imshow(samp, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        for r in range(4):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    for r, lab in enumerate(row_labels):
        ax[r, 0].set_ylabel(lab, fontsize=12)
    fig.suptitle("RF-GenLab demo: real vs. class-conditional generated spectrograms", fontsize=12)
    fig.tight_layout()
    out = ensure_dir(f"{cfg['results_dir']}/figures") / "fig_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


    from sklearn.metrics import roc_auc_score
    n_classes = len(CLASS_NAMES)


    cal = np.stack([iq_to_spectrogram(make_normal(rng.choice(CLASS_NAMES), n, fs, rng, cfg), sc)
                    for _ in range(400)])
    cal_scores = min_class_score(models["flow"], "flow", cal, n_classes, device)
    threshold = float(np.percentile(cal_scores, 95))


    clean = np.stack([iq_to_spectrogram(make_normal(CLASS_NAMES[i % n_classes], n, fs, rng, cfg), sc)
                      for i in range(150)])
    atypes = cfg["data"]["anomaly_types"]
    anom = np.stack([iq_to_spectrogram(make_anomaly(atypes[i % len(atypes)], n, fs, rng, cfg), sc)
                     for i in range(150)])
    s_clean = min_class_score(models["flow"], "flow", clean, n_classes, device)
    s_anom = min_class_score(models["flow"], "flow", anom, n_classes, device)
    y_true = np.r_[np.zeros(len(s_clean)), np.ones(len(s_anom))]
    y_score = np.r_[s_clean, s_anom]
    auc = roc_auc_score(y_true, y_score)
    acc = (np.r_[s_clean <= threshold, s_anom > threshold]).mean()

    print("\n=== Anomaly-detection demo (RealNVP flow, negative log-likelihood) ===")
    print(f"operating threshold (95th pct of normal scores) = {threshold:,.1f}")
    print(f"batch of 150 clean + 150 anomalous:  ROC-AUC = {auc:.3f}   "
          f"accuracy @ threshold = {acc:.3f}")


    print(f"\n{'example input':<22}{'score':>14}   verdict")
    print("-" * 52)
    examples = [("clean wifi", make_normal("wifi", n, fs, rng, cfg)),
                ("clean zigbee", make_normal("zigbee", n, fs, rng, cfg))]
    for atype in atypes:
        examples.append((f"anomaly: {atype}", make_anomaly(atype, n, fs, rng, cfg)))
    for name, iq in examples:
        sp = iq_to_spectrogram(iq, sc)[None]
        score = float(min_class_score(models["flow"], "flow", sp, n_classes, device)[0])
        print(f"{name:<22}{score:>14,.1f}   {'ANOMALY' if score > threshold else 'normal'}")
    print(f"\nwrote comparison figure -> {out}")


if __name__ == "__main__":
    main()
