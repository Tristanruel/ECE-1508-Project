from __future__ import annotations

import argparse

from rfgen.data import make_iq_loaders
from rfgen.dsp.spectrogram import SpectrogramConfig
from rfgen.models import ConditionalMambaAR
from rfgen.train.common import train_generative
from rfgen.utils import load_config


def build_mamba(cfg: dict) -> ConditionalMambaAR:
    m = cfg["models"]["mamba"]
    return ConditionalMambaAR(
        n_classes=len(cfg["signal"]["classes"]),
        n_samples=int(cfg["signal"]["num_samples"]),
        levels=int(m.get("levels", 256)),
        d_model=int(m["d_model"]),
        n_layers=int(m["n_layers"]),
        d_state=int(m["d_state"]),
        spec_cfg=SpectrogramConfig.from_config(cfg),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model = build_mamba(cfg)
    loaders = make_iq_loaders(cfg)
    train_generative(model, cfg, epochs=int(cfg["train"]["epochs_mamba"]), tag="mamba", loaders=loaders)


if __name__ == "__main__":
    main()
