"""dataset builder
Run:  python -m rfgen.data.build_dataset
      python -m rfgen.data.build_dataset --config configs/other.yaml
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
from tqdm import tqdm

from rfgen.dsp.generators import CLASS_NAMES
from rfgen.dsp.impairments import make_anomaly, make_normal
from rfgen.dsp.spectrogram import SpectrogramConfig, iq_to_spectrogram
from rfgen.utils import ensure_dir, load_config, resolve_path, set_seed


def _gen_normal_split(n: int, fs: float, cfg: dict, sc: SpectrogramConfig, rng):
    x = np.empty((n, 1, sc.n_freq, sc.n_time), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)
    iq = np.empty((n, cfg["signal"]["num_samples"]), dtype=np.complex64)
    for i in tqdm(range(n), leave=False):
        cls_idx = i % len(CLASS_NAMES)
        cls = CLASS_NAMES[cls_idx]
        sig = make_normal(cls, cfg["signal"]["num_samples"], fs, rng, cfg)
        x[i] = iq_to_spectrogram(sig, sc)
        y[i] = cls_idx
        iq[i] = sig

    perm = rng.permutation(n)
    return x[perm], y[perm], iq[perm]


def _gen_anomalies(cfg: dict, fs: float, sc: SpectrogramConfig, rng):
    types = cfg["data"]["anomaly_types"]
    per = int(cfg["data"]["n_anom_per_type"])
    total = per * len(types)
    x = np.empty((total, 1, sc.n_freq, sc.n_time), dtype=np.float32)
    y_type = np.empty((total,), dtype=np.int64)
    iq = np.empty((total, cfg["signal"]["num_samples"]), dtype=np.complex64)
    k = 0
    for ti, atype in enumerate(types):
        for _ in tqdm(range(per), desc=atype, leave=False):
            sig = make_anomaly(atype, cfg["signal"]["num_samples"], fs, rng, cfg)
            x[k] = iq_to_spectrogram(sig, sc)
            y_type[k] = ti
            iq[k] = sig
            k += 1
    perm = rng.permutation(total)
    return x[perm], y_type[perm], iq[perm]


def build(config_path: str | None = None) -> str:
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))
    fs = float(cfg["signal"]["sample_rate_hz"])
    sc = SpectrogramConfig.from_config(cfg)


    ss = np.random.SeedSequence(int(cfg["seed"]))
    rngs = [np.random.default_rng(s) for s in ss.spawn(4)]

    out_dir = ensure_dir(cfg["data"]["out_dir"])
    out_path = out_dir / "dataset.h5"

    print(f"[build] fs={fs/1e6:.1f} MHz  N={cfg['signal']['num_samples']}  "
          f"spec={sc.n_freq}x{sc.n_time}  -> {out_path}")

    print("[build] train ...")
    xtr, ytr, iqtr = _gen_normal_split(int(cfg["data"]["n_train"]), fs, cfg, sc, rngs[0])
    print("[build] val ...")
    xva, yva, iqva = _gen_normal_split(int(cfg["data"]["n_val"]), fs, cfg, sc, rngs[1])
    print("[build] test (normal) ...")
    xte, yte, iqte = _gen_normal_split(int(cfg["data"]["n_test"]), fs, cfg, sc, rngs[2])
    print("[build] anomalies ...")
    xan, yan, iqan = _gen_anomalies(cfg, fs, sc, rngs[3])

    with h5py.File(out_path, "w") as f:
        f.attrs["class_names"] = json.dumps(CLASS_NAMES)
        f.attrs["anomaly_types"] = json.dumps(list(cfg["data"]["anomaly_types"]))
        f.attrs["config"] = json.dumps(cfg)
        f.attrs["spectrogram"] = json.dumps(cfg["spectrogram"])
        for name, (xx, yy, iqq) in {
            "train": (xtr, ytr, iqtr),
            "val": (xva, yva, iqva),
            "test": (xte, yte, iqte),
        }.items():
            g = f.create_group(name)
            g.create_dataset("x", data=xx, compression="gzip", compression_opts=4)
            g.create_dataset("y", data=yy)
            g.create_dataset("iq", data=iqq, compression="gzip", compression_opts=4)
        g = f.create_group("anom")
        g.create_dataset("x", data=xan, compression="gzip", compression_opts=4)
        g.create_dataset("y_type", data=yan)
        g.create_dataset("iq", data=iqan, compression="gzip", compression_opts=4)

    print(f"[build] wrote {out_path}  "
          f"(train={len(xtr)} val={len(xva)} test={len(xte)} anom={len(xan)})")
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the RF spectrogram dataset.")
    ap.add_argument("--config", default=None, help="path to YAML config")
    args = ap.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
