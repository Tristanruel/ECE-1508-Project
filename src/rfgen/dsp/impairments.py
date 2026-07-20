"""Channel/hardware impairments (for realistic *normal* signals) and the
corruption operators used to synthesise *anomalies* for anomaly detection.

Design intent
-------------
"Normal" signals get mild, bounded nuisance variation (AWGN in a healthy SNR
band, small carrier-frequency offset, small gain) so the generative models must
learn a distribution rather than memorise a single waveform.

"Anomalies" are valid signals pushed *outside* that learned distribution in four
qualitatively different ways -- exactly the failure modes named in the project
brief: heavy noise/corruption, frequency shifting, broken timing structure, and
mixing features from different protocol classes.
"""
from __future__ import annotations

import numpy as np

from rfgen.dsp.generators import CLASS_NAMES, generate_signal, normalize_power


def add_awgn(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_power = float(np.mean(np.abs(iq) ** 2)) + 1e-12
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(iq.shape) + 1j * rng.standard_normal(iq.shape)
    )
    return (iq + noise).astype(np.complex64)


def apply_cfo(iq: np.ndarray, cfo_frac: float, rng: np.random.Generator | None = None) -> np.ndarray:
    n = np.arange(iq.size)
    phase0 = 0.0 if rng is None else rng.uniform(0, 2 * np.pi)
    return (iq * np.exp(1j * (2 * np.pi * cfo_frac * n + phase0))).astype(np.complex64)


def apply_gain(iq: np.ndarray, gain_db: float) -> np.ndarray:
    return (iq * (10.0 ** (gain_db / 20.0))).astype(np.complex64)


def make_normal(
    class_name: str,
    n: int,
    fs: float,
    rng: np.random.Generator,
    cfg: dict,
) -> np.ndarray:
    sig = cfg["signal"]
    iq = generate_signal(class_name, n, fs, rng, params=sig)
    lo, hi = sig["cfo_frac_range"]
    iq = apply_cfo(iq, rng.uniform(lo, hi), rng)
    lo, hi = sig["gain_db_range"]
    iq = apply_gain(iq, rng.uniform(lo, hi))
    lo, hi = sig["snr_db_range"]
    iq = add_awgn(iq, rng.uniform(lo, hi), rng)
    return iq


def anom_heavy_noise(n: int, fs: float, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    cls = rng.choice(CLASS_NAMES)
    iq = generate_signal(cls, n, fs, rng, params=cfg["signal"])
    return add_awgn(iq, rng.uniform(-6.0, 4.0), rng)


def anom_freq_shift(n: int, fs: float, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    cls = rng.choice(CLASS_NAMES)
    iq = generate_signal(cls, n, fs, rng, params=cfg["signal"])
    frac = rng.choice([-1.0, 1.0]) * rng.uniform(0.28, 0.45)
    iq = apply_cfo(iq, frac, rng)
    return add_awgn(iq, rng.uniform(12.0, 25.0), rng)


def anom_timing_corrupt(n: int, fs: float, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    cls = rng.choice(CLASS_NAMES)
    iq = generate_signal(cls, n, fs, rng, params=cfg["signal"]).copy()

    for _ in range(rng.integers(2, 5)):
        g = int(rng.integers(n // 16, n // 6))
        start = int(rng.integers(0, max(1, n - g)))
        iq[start : start + g] = 0.0

    clen = int(rng.integers(n // 8, n // 4))
    src = int(rng.integers(0, n - clen))
    dst = int(rng.integers(0, n - clen))
    iq[dst : dst + clen] = iq[src : src + clen]
    iq = normalize_power(iq)
    return add_awgn(iq, rng.uniform(12.0, 25.0), rng)


def anom_protocol_mix(n: int, fs: float, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    c1, c2 = rng.choice(CLASS_NAMES, size=2, replace=False)
    a = generate_signal(c1, n, fs, rng, params=cfg["signal"])
    b = generate_signal(c2, n, fs, rng, params=cfg["signal"])
    b = apply_cfo(b, rng.uniform(-0.1, 0.1), rng)
    ratio = rng.uniform(0.6, 1.0)
    mix = normalize_power(a + ratio * b)
    return add_awgn(mix, rng.uniform(12.0, 25.0), rng)


ANOMALY_FNS = {
    "heavy_noise": anom_heavy_noise,
    "freq_shift": anom_freq_shift,
    "timing_corrupt": anom_timing_corrupt,
    "protocol_mix": anom_protocol_mix,
}


def make_anomaly(anom_type: str, n: int, fs: float, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    try:
        fn = ANOMALY_FNS[anom_type]
    except KeyError as exc:
        raise ValueError(f"unknown anomaly type '{anom_type}'") from exc
    return fn(n, fs, rng, cfg)
