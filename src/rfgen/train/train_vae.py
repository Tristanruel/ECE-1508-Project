from __future__ import annotations

import argparse

from rfgen.models import ConditionalVAE
from rfgen.train.common import train_generative
from rfgen.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    m = cfg["models"]["vae"]
    model = ConditionalVAE(
        n_classes=len(cfg["signal"]["classes"]),
        latent_dim=int(m["latent_dim"]),
        base_ch=int(m["base_ch"]),
        beta=float(m["beta"]),
    )
    train_generative(model, cfg, epochs=int(cfg["train"]["epochs_vae"]), tag="vae")


if __name__ == "__main__":
    main()
