from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.linalg import sqrtm

from rfgen.data import load_split
from rfgen.eval.common import load_all
from rfgen.utils import ensure_dir, get_device, load_config, resolve_path, save_json


@torch.no_grad()
def _frechet(feat_r: np.ndarray, feat_g: np.ndarray) -> float:
    mu_r, mu_g = feat_r.mean(0), feat_g.mean(0)
    cov_r = np.cov(feat_r, rowvar=False) + 1e-6 * np.eye(feat_r.shape[1])
    cov_g = np.cov(feat_g, rowvar=False) + 1e-6 * np.eye(feat_g.shape[1])
    covmean = sqrtm(cov_r @ cov_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_r - mu_g
    return float(diff @ diff + np.trace(cov_r + cov_g - 2 * covmean))


@torch.no_grad()
def _features(clf, x: torch.Tensor, device, batch: int = 256) -> np.ndarray:
    out = []
    for i in range(0, x.shape[0], batch):
        out.append(clf.features(x[i : i + batch].to(device)).cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def _generate(model, tag: str, n_per_class: int, n_classes: int, device):
    xs, ys = [], []
    for c in range(n_classes):
        y = torch.full((n_per_class,), c, device=device, dtype=torch.long)
        xs.append(model.sample(n_per_class, y).cpu())
        ys.append(torch.full((n_per_class,), c))
    return torch.cat(xs, 0), torch.cat(ys, 0)


@torch.no_grad()
def _likelihood_metrics(models, cfg, device) -> dict:
    test = load_split(cfg, "test")
    x = test.x.to(device)
    y = test.y.to(device)
    out = {}


    out["flow"] = {"test_bpd": float(models["flow"].bits_per_dim(x, y)),
                   "test_nll_nats": float(models["flow"].nll(x, y).mean())}

    recon, kl = models["vae"].elbo_terms(x, y)
    d = x[0].numel()
    out["vae"] = {
        "test_bpd_bound": float(((recon + kl) / (d * np.log(2))).mean()),
        "test_recon_nats": float(recon.mean()),
        "test_kl_nats": float(kl.mean()),
        "test_recon_mse": float(models["vae"].recon_error(x, y).mean()),
    }

    out["diffusion"] = {"test_denoise_mse": float(models["diffusion"].denoising_error(x, y).mean())}
    # Mamba: exact discrete bits/dim on raw I/Q with true labels.
    if "mamba" in models:
        from rfgen.data.dataset import load_split_iq
        from rfgen.eval.common import iq_complex_to_channels
        iq, y_iq = load_split_iq(cfg, "test")
        iq_t = torch.from_numpy(iq_complex_to_channels(iq)).float().to(device)
        y_t = torch.from_numpy(y_iq).long().to(device)
        bpd = 0.0
        for i in range(0, iq_t.shape[0], 256):
            bpd += float(models["mamba"].bits_per_dim(iq_t[i:i+256], y_t[i:i+256])) * min(256, iq_t.shape[0]-i)
        out["mamba"] = {"test_bpd": bpd / iq_t.shape[0]}
    return out


def evaluate(cfg: dict, n_per_class: int = 500) -> dict:
    device = get_device()
    n_classes = len(cfg["signal"]["classes"])
    class_names = list(cfg["signal"]["classes"])
    models, clf = load_all(cfg, device)
    from rfgen.eval.common import load_mamba
    models["mamba"] = load_mamba(cfg, device)  
    test = load_split(cfg, "test")
    feat_real = _features(clf, test.x, device)

    results = {"per_model": {}, "class_names": class_names}
    sample_cache = {}
    for tag, model in models.items():
        gx, gy = _generate(model, tag, n_per_class, n_classes, device)
        logits = []
        for i in range(0, gx.shape[0], 256):
            logits.append(clf(gx[i : i + 256].to(device)).cpu())
        pred = torch.cat(logits, 0).argmax(1)
        acc = float((pred == gy).float().mean())
        per_class_acc = {
            class_names[c]: float((pred[gy == c] == c).float().mean()) for c in range(n_classes)
        }
        fid = _frechet(feat_real, _features(clf, gx, device))
        results["per_model"][tag] = {
            "gen_accuracy": acc, "per_class_accuracy": per_class_acc, "frechet": fid,
        }
        sample_cache[tag] = (gx.numpy(), gy.numpy())
        print(f"[gen:{tag:9s}] gen_acc={acc:.4f}  frechet={fid:.3f}  "
              + "  ".join(f"{k}={v:.2f}" for k, v in per_class_acc.items()))

    results["likelihood"] = _likelihood_metrics(models, cfg, device)
    print("[gen] likelihood/reconstruction:")
    for tag, d in results["likelihood"].items():
        print(f"    {tag:9s} " + "  ".join(f"{k}={v:.4f}" for k, v in d.items()))

    ensure_dir(cfg["results_dir"])
    save_json(results, f"{cfg['results_dir']}/generation_metrics.json")


    def _balanced(gx, gy, k=20):
        xs, ys = [], []
        for c in range(n_classes):
            idx = np.where(gy == c)[0][:k]
            xs.append(gx[idx]); ys.append(gy[idx])
        return np.concatenate(xs, 0), np.concatenate(ys, 0)

    npz = resolve_path(f"{cfg['results_dir']}/generated_samples.npz")
    balanced = {t: _balanced(*sample_cache[t]) for t in sample_cache}
    np.savez(npz, **{f"{t}_x": balanced[t][0] for t in balanced},
             **{f"{t}_y": balanced[t][1] for t in balanced})
    print(f"[gen] wrote {cfg['results_dir']}/generation_metrics.json (+generated_samples.npz)")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-per-class", type=int, default=500)
    args = ap.parse_args()
    evaluate(load_config(args.config), n_per_class=args.n_per_class)


if __name__ == "__main__":
    main()
