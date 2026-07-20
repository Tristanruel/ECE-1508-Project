
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from rfgen.data import load_anomalies, load_split
from rfgen.data.dataset import class_names
from rfgen.utils import ensure_dir, load_config, load_json, resolve_path

CMAP = "viridis"
MODEL_COLORS = {"vae": "#1f77b4", "flow": "#d62728", "diffusion": "#2ca02c", "mamba": "#9467bd"}
MODEL_LABELS = {"vae": "Cond. VAE", "flow": "RealNVP flow", "diffusion": "Cond. DDPM",
                "mamba": "Mamba (I/Q)"}
TAGS = ["vae", "flow", "diffusion", "mamba"]


def _imgrid(ax, img):
    ax.imshow(img, origin="lower", aspect="auto", cmap=CMAP, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])


def fig_signal_gallery(cfg, outdir):
    test = load_split(cfg, "test")
    x, y = test.x.numpy(), test.y.numpy()
    names = class_names(cfg)
    xan, ytype, tnames = load_anomalies(cfg)
    rows = names + tnames
    ncol = 6
    fig, ax = plt.subplots(len(rows), ncol, figsize=(1.5 * ncol, 1.5 * len(rows)))
    for r, name in enumerate(rows):
        if r < len(names):
            idx = np.where(y == r)[0][:ncol]
            imgs = x[idx, 0]
        else:
            idx = np.where(ytype == (r - len(names)))[0][:ncol]
            imgs = xan[idx, 0]
        for c in range(ncol):
            _imgrid(ax[r, c], imgs[c])
        ax[r, 0].set_ylabel(name, fontsize=11, rotation=90, labelpad=6)
    ax[0, ncol // 2].set_title("Real protocols (top 3 rows) and anomalies (bottom 4 rows)", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "fig_signal_gallery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_real_vs_generated(cfg, outdir):
    names = class_names(cfg)
    n_cls = len(names)
    per_cls = 4
    test = load_split(cfg, "test")
    xr, yr = test.x.numpy(), test.y.numpy()
    samples = np.load(resolve_path(f"{cfg['results_dir']}/generated_samples.npz"))

    def row_images(x, y):
        imgs = []
        for c in range(n_cls):
            idx = np.where(y == c)[0][:per_cls]
            imgs.append(x[idx, 0])
        return np.concatenate(imgs, 0)

    row_defs = [("Real", row_images(xr, yr))]
    for tag in TAGS:
        row_defs.append((MODEL_LABELS[tag], row_images(samples[f"{tag}_x"], samples[f"{tag}_y"])))

    ncol = n_cls * per_cls
    fig, ax = plt.subplots(len(row_defs), ncol, figsize=(1.2 * ncol, 1.35 * len(row_defs)))
    for r, (label, imgs) in enumerate(row_defs):
        for c in range(ncol):
            _imgrid(ax[r, c], imgs[c])
        ax[r, 0].set_ylabel(label, fontsize=11)
    for c in range(n_cls):
        ax[0, c * per_cls + per_cls // 2].set_title(names[c], fontsize=11)
    fig.suptitle("Real vs. class-conditional generated spectrograms", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "fig_real_vs_generated.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_roc_and_hist(cfg, outdir):
    scores = np.load(resolve_path(f"{cfg['results_dir']}/anomaly_scores.npz"))
    metrics = load_json(f"{cfg['results_dir']}/anomaly_metrics.json")["per_model"]


    fig, ax = plt.subplots(figsize=(5, 4.2))
    for tag in TAGS:
        sn, sa = scores[f"{tag}_normal"], scores[f"{tag}_anom"]
        yt = np.concatenate([np.zeros_like(sn), np.ones_like(sa)])
        ys = np.concatenate([sn, sa])
        fpr, tpr, _ = roc_curve(yt, ys)
        ax.plot(fpr, tpr, color=MODEL_COLORS[tag], lw=2,
                label=f"{MODEL_LABELS[tag]} (AUC={metrics[tag]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Anomaly-detection ROC"); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(outdir / "fig_roc.png", dpi=150); plt.close(fig)


    fig, axes = plt.subplots(1, len(TAGS), figsize=(3.4 * len(TAGS), 3.2))
    for ax, tag in zip(axes, TAGS):
        sn, sa = scores[f"{tag}_normal"], scores[f"{tag}_anom"]
        center = np.median(sn)
        scale = 1.4826 * np.median(np.abs(sn - center)) + 1e-9
        tf = lambda s: np.arcsinh((s - center) / scale)
        tn, ta = tf(sn), tf(sa)
        lo, hi = np.percentile(np.concatenate([tn, ta]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        ax.hist(tn, bins=bins, alpha=0.6, color="#2c7fb8", label="normal", density=True)
        ax.hist(ta, bins=bins, alpha=0.6, color="#d95f0e", label="anomaly", density=True)
        ax.set_title(f"{MODEL_LABELS[tag]}")
        ax.set_xlabel("anomaly score (asinh-scaled)"); ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Anomaly-score distributions: normal vs. anomaly", y=1.02)
    fig.tight_layout(); fig.savefig(outdir / "fig_score_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_pertype_auc(cfg, outdir):
    metrics = load_json(f"{cfg['results_dir']}/anomaly_metrics.json")
    types = metrics["anomaly_types"]
    tags = TAGS
    x = np.arange(len(types)); w = 0.2
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for i, tag in enumerate(tags):
        vals = [metrics["per_model"][tag]["per_type_auc"][t] for t in types]
        ax.bar(x + (i - 1.5) * w, vals, w, color=MODEL_COLORS[tag], label=MODEL_LABELS[tag])
    ax.set_xticks(x); ax.set_xticklabels(types, rotation=15)
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("ROC-AUC")
    ax.set_title("Anomaly-detection AUC by corruption type")
    ax.legend(fontsize=9, ncol=4, loc="lower center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / "fig_pertype_auc.png", dpi=150); plt.close(fig)


def fig_training_curves(cfg, outdir):
    fig, axes = plt.subplots(1, len(TAGS), figsize=(3.2 * len(TAGS), 3.2))
    for ax, tag in zip(axes, TAGS):
        h = load_json(f"{cfg['train']['runs_dir']}/{tag}/history.json")["history"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], color=MODEL_COLORS[tag], label="train")
        ax.plot(ep, h["val_loss"], color=MODEL_COLORS[tag], ls="--", label="val")
        ax.set_title(MODEL_LABELS[tag]); ax.set_xlabel("epoch"); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("loss")
    fig.suptitle("Training curves", y=1.02)
    fig.tight_layout(); fig.savefig(outdir / "fig_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_summary_bars(cfg, outdir):
    an = load_json(f"{cfg['results_dir']}/anomaly_metrics.json")["per_model"]
    gen = load_json(f"{cfg['results_dir']}/generation_metrics.json")["per_model"]
    tags = TAGS
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar([MODEL_LABELS[t] for t in tags], [an[t]["auc"] for t in tags],
                color=[MODEL_COLORS[t] for t in tags])
    axes[0].set_ylim(0.5, 1.0); axes[0].set_title("Anomaly AUC (higher better)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar([MODEL_LABELS[t] for t in tags], [gen[t]["frechet"] for t in tags],
                color=[MODEL_COLORS[t] for t in tags])
    axes[1].set_yscale("log")
    axes[1].set_title("Frechet distance (lower better, log scale)")
    axes[1].grid(axis="y", alpha=0.3, which="both")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(outdir / "fig_summary_bars.png", dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    outdir = ensure_dir(f"{cfg['results_dir']}/figures")
    print(f"[figures] writing to {outdir}")
    fig_signal_gallery(cfg, outdir)
    fig_real_vs_generated(cfg, outdir)
    fig_roc_and_hist(cfg, outdir)
    fig_pertype_auc(cfg, outdir)
    fig_training_curves(cfg, outdir)
    fig_summary_bars(cfg, outdir)
    print("[figures] done:", ", ".join(p.name for p in sorted(outdir.glob("*.png"))))


if __name__ == "__main__":
    main()
