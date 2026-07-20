#!/usr/bin/env python
"""End-to-end pipeline: build data -> train all models -> evaluate -> figures.

Reproduces every result and figure in the report with a single command:

    python scripts/run_all.py
    python scripts/run_all.py --skip-data
    python scripts/run_all.py --config configs/quick.yaml

On a modern GPU the whole pipeline finishes in ~2 minutes; on CPU in a few
minutes. Every stage is also runnable on its own as ``python -m rfgen.<stage>``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rfgen.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-data", action="store_true", help="reuse existing dataset")
    args = ap.parse_args()
    cfg = load_config(args.config)

    from rfgen.data.dataset import dataset_path

    t0 = time.time()


    if args.skip_data and dataset_path(cfg).exists():
        print(f"[run_all] reusing dataset at {dataset_path(cfg)}")
    else:
        from rfgen.data.build_dataset import build
        build(args.config)


    from rfgen.models import ConditionalDDPM, ConditionalRealNVP, ConditionalVAE
    from rfgen.train.common import train_generative
    import rfgen.train.train_classifier as tclf

    print("\n=== Train classifier ===")
    sys.argv = ["train_classifier"] + (["--config", args.config] if args.config else [])
    tclf.main()

    n_classes = len(cfg["signal"]["classes"])
    s = cfg["spectrogram"]
    print("\n=== Train VAE ===")
    train_generative(
        ConditionalVAE(n_classes=n_classes, latent_dim=int(cfg["models"]["vae"]["latent_dim"]),
                       base_ch=int(cfg["models"]["vae"]["base_ch"]), beta=float(cfg["models"]["vae"]["beta"])),
        cfg, epochs=int(cfg["train"]["epochs_vae"]), tag="vae")
    print("\n=== Train Flow ===")
    train_generative(
        ConditionalRealNVP(n_classes=n_classes, img_shape=(1, int(s["n_freq"]), int(s["n_time"])),
                           n_coupling=int(cfg["models"]["flow"]["n_coupling"]), hidden=int(cfg["models"]["flow"]["hidden"])),
        cfg, epochs=int(cfg["train"]["epochs_flow"]), tag="flow")
    print("\n=== Train Diffusion ===")
    train_generative(
        ConditionalDDPM(n_classes=n_classes, base_ch=int(cfg["models"]["diffusion"]["base_ch"]),
                        timesteps=int(cfg["models"]["diffusion"]["timesteps"]),
                        beta_start=float(cfg["models"]["diffusion"]["beta_start"]),
                        beta_end=float(cfg["models"]["diffusion"]["beta_end"])),
        cfg, epochs=int(cfg["train"]["epochs_diffusion"]), tag="diffusion")

    print("\n=== Train Mamba (autoregressive, raw I/Q) ===")
    from rfgen.data import make_iq_loaders
    from rfgen.train.train_mamba import build_mamba
    train_generative(build_mamba(cfg), cfg, epochs=int(cfg["train"]["epochs_mamba"]),
                     tag="mamba", loaders=make_iq_loaders(cfg))


    print("\n=== Evaluate: anomaly detection ===")
    from rfgen.eval.anomaly import evaluate as eval_anomaly
    eval_anomaly(cfg)
    print("\n=== Evaluate: generation quality ===")
    from rfgen.eval.generation import evaluate as eval_generation
    eval_generation(cfg)


    print("\n=== Figures ===")
    import rfgen.eval.figures as figs
    sys.argv = ["figures"] + (["--config", args.config] if args.config else [])
    figs.main()
    print("\n=== Summary ===")
    from rfgen.eval.summarize import build_summary
    from rfgen.utils import resolve_path
    md = build_summary(cfg)
    resolve_path(f"{cfg['results_dir']}/summary.md").write_text(md, encoding="utf-8")
    print(md)

    print(f"\n[run_all] complete in {time.time() - t0:.1f}s. See results/ and results/figures/.")


if __name__ == "__main__":
    main()
