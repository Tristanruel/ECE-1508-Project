from __future__ import annotations

import argparse

from rfgen.models import ConditionalDDPM
from rfgen.train.common import train_generative
from rfgen.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    m = cfg["models"]["diffusion"]
    model = ConditionalDDPM(
        n_classes=len(cfg["signal"]["classes"]),
        base_ch=int(m["base_ch"]),
        timesteps=int(m["timesteps"]),
        beta_start=float(m["beta_start"]),
        beta_end=float(m["beta_end"]),
    )
    train_generative(model, cfg, epochs=int(cfg["train"]["epochs_diffusion"]), tag="diffusion")


if __name__ == "__main__":
    main()
