from __future__ import annotations

import argparse

from rfgen.models import ConditionalRealNVP
from rfgen.train.common import train_generative
from rfgen.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    m = cfg["models"]["flow"]
    s = cfg["spectrogram"]
    model = ConditionalRealNVP(
        n_classes=len(cfg["signal"]["classes"]),
        img_shape=(1, int(s["n_freq"]), int(s["n_time"])),
        n_coupling=int(m["n_coupling"]),
        hidden=int(m["hidden"]),
    )
    train_generative(model, cfg, epochs=int(cfg["train"]["epochs_flow"]), tag="flow")


if __name__ == "__main__":
    main()
