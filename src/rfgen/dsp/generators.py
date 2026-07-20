"""Synthetic RF waveform generators for three 2.4 GHz-band protocol families.

The goal is protocol-like baseband I/Q that is cheap, fully reproducible from a
seed, and visually/statistically distinct in a spectrogram. The three families are:

* wifi   -- OFDM (64-point FFT, 16-sample cyclic prefix, 52 active carriers,
                QPSK sub-carriers, comb pilots). Wideband, block structure.
* ble    -- GFSK (Gaussian-smoothed 2-FSK, modulation index ~0.5). Narrowband,
                constant-envelope, a wandering tone.
* zigbee -- O-QPSK with 16-chip DSSS spreading and half-symbol Q offset.
                Medium bandwidth, near-constant envelope.
"""
from __future__ import annotations

import numpy as np

CLASS_NAMES: list[str] = ["wifi", "ble", "zigbee"]
CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


def normalize_power(iq: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale to unit average power (RMS = 1)."""
    iq = np.asarray(iq, dtype=np.complex64)
    if iq.size == 0:
        return iq
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
    if rms < eps:
        return iq
    return (iq / rms).astype(np.complex64)


def _fit_length(iq: np.ndarray, n: int) -> np.ndarray:
    iq = np.asarray(iq, dtype=np.complex64)
    if iq.size >= n:
        return iq[:n]
    out = np.zeros(n, dtype=np.complex64)
    out[: iq.size] = iq
    return out


def _rrc_taps(sps: int, span: int, rolloff: float) -> np.ndarray:
    """Root-raised-cosine pulse taps."""
    n = span * sps
    t = (np.arange(-n // 2, n // 2 + 1)) / sps
    beta = rolloff
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            taps[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-6:
            taps[i] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(
                np.pi * ti * (1 + beta)
            )
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den
    taps /= np.sqrt(np.sum(taps ** 2))
    return taps.astype(np.float64)


def _pulse_shape(symbols: np.ndarray, sps: int, n_out: int, rolloff: float = 0.35) -> np.ndarray:
    """Upsample complex symbols by ``sps`` and RRC pulse-shape them."""
    up = np.zeros(len(symbols) * sps, dtype=np.complex64)
    up[::sps] = symbols
    taps = _rrc_taps(sps, span=6, rolloff=rolloff)
    shaped = np.convolve(up, taps, mode="same")
    return _fit_length(shaped, n_out)


def gen_wifi_ofdm(
    n: int, rng: np.random.Generator, nfft: int = 64, cp_len: int = 16, active: int = 52
) -> np.ndarray:
    """OFDM with QPSK data sub-carriers and boosted comb pilots."""
    symbol_len = nfft + cp_len
    n_symbols = int(np.ceil(n / symbol_len)) + 1
    active = min(active, nfft - 2)
    half = active // 2
    carrier_idx = np.r_[np.arange(-half, 0), np.arange(1, active - half + 1)]
    bins = np.mod(carrier_idx, nfft)

    pilot_spacing = max(2, len(bins) // 8)
    pilot_bins = bins[::pilot_spacing]
    pilot_seq = (1 - 2 * (np.arange(len(pilot_bins)) % 2)).astype(np.complex64)
    qpsk = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2)).astype(np.complex64)

    out = []
    for _ in range(n_symbols):
        grid = np.zeros(nfft, dtype=np.complex64)
        grid[bins] = qpsk[rng.integers(0, 4, len(bins))]
        grid[pilot_bins] = pilot_seq * 1.2
        time_sym = np.fft.ifft(grid) * np.sqrt(nfft)
        out.append(np.r_[time_sym[-cp_len:], time_sym])
    return normalize_power(_fit_length(np.concatenate(out), n))


def gen_ble_gfsk(
    n: int, rng: np.random.Generator, fs: float, symbol_rate: float, deviation: float
) -> np.ndarray:
    """GFSK: Gaussian-smoothed binary FSK, constant envelope (BLE-like)."""
    sps = max(2, int(round(fs / symbol_rate)))
    n_symbols = int(np.ceil(n / sps)) + 4
    bits = rng.integers(0, 2, n_symbols)
    freq = np.repeat(2 * bits - 1, sps)[:n].astype(np.float64) * deviation

    span = max(3, sps // 2)
    t = np.linspace(-2.5, 2.5, span)
    kernel = np.exp(-0.5 * t * t)
    kernel /= kernel.sum()
    freq = np.convolve(freq, kernel, mode="same")
    phase = 2.0 * np.pi * np.cumsum(freq) / fs
    return normalize_power(np.exp(1j * phase).astype(np.complex64))


def gen_zigbee_oqpsk(
    n: int, rng: np.random.Generator, fs: float, chip_rate: float
) -> np.ndarray:
    """O-QPSK with 16-chip DSSS spreading and a half-chip Q offset (ZigBee-like)."""
    sps = max(2, int(round(fs / chip_rate)))
    n_chips = int(np.ceil(n / sps)) + 64
    chip_seq = np.array(
        [1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1], dtype=np.float32
    )
    n_symbols = int(np.ceil(n_chips / 16)) + 2
    data_i = 2 * rng.integers(0, 2, n_symbols) - 1
    data_q = 2 * rng.integers(0, 2, n_symbols) - 1
    chips_i = np.outer(data_i, chip_seq).reshape(-1)[:n_chips]
    chips_q = np.outer(data_q, np.roll(chip_seq, 3)).reshape(-1)[:n_chips]
    i_shaped = _pulse_shape(chips_i.astype(np.complex64), sps, n + sps, 0.35).real
    q_shaped = _pulse_shape(chips_q.astype(np.complex64), sps, n + sps, 0.35).real

    off = sps // 2
    i_arm = i_shaped[:n]
    q_arm = q_shaped[off : off + n]
    return normalize_power((i_arm + 1j * q_arm).astype(np.complex64))


def generate_signal(
    class_name: str,
    n: int,
    fs: float,
    rng: np.random.Generator,
    params: dict | None = None,
) -> np.ndarray:
    """Generate a clean, unit-power baseband I/Q signal for ``class_name``."""
    params = params or {}
    fs = float(fs)
    if class_name == "wifi":
        p = params.get("wifi", {})
        return gen_wifi_ofdm(
            n, rng,
            nfft=int(p.get("nfft", 64)),
            cp_len=int(p.get("cp_len", 16)),
            active=int(p.get("active_carriers", 52)),
        )
    if class_name == "ble":
        p = params.get("ble", {})
        return gen_ble_gfsk(
            n, rng, fs,
            symbol_rate=float(p.get("symbol_rate_hz", 1.0e6)),
            deviation=float(p.get("deviation_hz", 250.0e3)),
        )
    if class_name == "zigbee":
        p = params.get("zigbee", {})
        return gen_zigbee_oqpsk(n, rng, fs, chip_rate=float(p.get("chip_rate_hz", 2.0e6)))
    raise ValueError(f"unknown class '{class_name}' (expected one of {CLASS_NAMES})")
