"""Anomaly detection evaluation
    python -m rfgen.eval.anomaly [--config ...]
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from rfgen.data import load_anomalies, load_split
from rfgen.eval.common import load_all, min_class_score
from rfgen.utils import ensure_dir, get_device, load_config, save_json


def evaluate(cfg: dict) -> dict:
    device = get_device()
    n_classes = len(cfg["signal"]["classes"])
    models, _ = load_all(cfg, device)

    test = load_split(cfg, "test")
    x_normal = test.x.numpy()
    x_anom, y_type, type_names = load_anomalies(cfg)

    def _summarize(tag, s_norm, s_anom):
        y_true = np.concatenate([np.zeros_like(s_norm), np.ones_like(s_anom)])
        y_score = np.concatenate([s_norm, s_anom])
        per_type = {}
        for ti, name in enumerate(type_names):
            mask = y_type == ti
            yt = np.concatenate([np.zeros_like(s_norm), np.ones(mask.sum())])
            ys = np.concatenate([s_norm, s_anom[mask]])
            per_type[name] = float(roc_auc_score(yt, ys))
        results["per_model"][tag] = {
            "auc": float(roc_auc_score(y_true, y_score)),
            "ap": float(average_precision_score(y_true, y_score)),
            "per_type_auc": per_type,
            "normal_score_mean": float(s_norm.mean()),
            "anom_score_mean": float(s_anom.mean()),
        }
        scores_cache[tag] = {"normal": s_norm, "anom": s_anom}
        r = results["per_model"][tag]
        print(f"[anomaly:{tag:9s}] AUC={r['auc']:.4f} AP={r['ap']:.4f}  "
              + "  ".join(f"{k}={v:.3f}" for k, v in per_type.items()))

    results = {"per_model": {}, "anomaly_types": type_names}
    scores_cache = {}
    # Spectrogram-domain models (VAE, flow, diffusion).
    for tag, model in models.items():
        _summarize(tag,
                   min_class_score(model, tag, x_normal, n_classes, device),
                   min_class_score(model, tag, x_anom, n_classes, device))

    # I/Q-domain Mamba autoregressive model (scored on raw I/Q).
    from rfgen.data.dataset import load_anomalies_iq, load_split_iq
    from rfgen.eval.common import iq_complex_to_channels, load_mamba, min_class_score_iq
    mamba = load_mamba(cfg, device)
    iq_norm, _ = load_split_iq(cfg, "test")
    iq_anom, _ = load_anomalies_iq(cfg)
    _summarize("mamba",
               min_class_score_iq(mamba, iq_complex_to_channels(iq_norm), n_classes, device),
               min_class_score_iq(mamba, iq_complex_to_channels(iq_anom), n_classes, device))

    ensure_dir(cfg["results_dir"])
    save_json(results, f"{cfg['results_dir']}/anomaly_metrics.json")

    from rfgen.utils import resolve_path
    npz_path = resolve_path(f"{cfg['results_dir']}/anomaly_scores.npz")
    np.savez(
        npz_path,
        **{f"{t}_normal": scores_cache[t]["normal"] for t in scores_cache},
        **{f"{t}_anom": scores_cache[t]["anom"] for t in scores_cache},
        y_type=y_type,
        type_names=np.array(type_names),
    )
    print(f"[anomaly] wrote {cfg['results_dir']}/anomaly_metrics.json (+scores.npz)")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
