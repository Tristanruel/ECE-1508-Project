from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectrogramConfig:
    nperseg: int = 32
    hop: int = 16
    n_freq: int = 32
    n_time: int = 32
    ref_db: float = 80.0

    @classmethod
    def from_config(cls, cfg: dict) -> "SpectrogramConfig":
        s = cfg["spectrogram"]
        return cls(
            nperseg=int(s["nperseg"]),
            hop=int(s["hop"]),
            n_freq=int(s["n_freq"]),
            n_time=int(s["n_time"]),
            ref_db=float(s["ref_db"]),
        )


def _stft(iq: np.ndarray, nperseg: int, hop: int) -> np.ndarray:
    window = np.hanning(nperseg).astype(np.float32)
    n = iq.size
    n_frames = 1 + max(0, (n - nperseg) // hop)
    frames = np.empty((nperseg, n_frames), dtype=np.complex64)
    for t in range(n_frames):
        seg = iq[t * hop : t * hop + nperseg] * window
        frames[:, t] = np.fft.fftshift(np.fft.fft(seg, n=nperseg))
    return frames


def _fix_shape(mat: np.ndarray, n_freq: int, n_time: int) -> np.ndarray:
    f, t = mat.shape
    out = np.zeros((n_freq, n_time), dtype=mat.dtype)
    out[: min(f, n_freq), : min(t, n_time)] = mat[:n_freq, :n_time]
    return out


def iq_to_spectrogram(iq: np.ndarray, cfg: SpectrogramConfig) -> np.ndarray:
    frames = _stft(iq, cfg.nperseg, cfg.hop)
    mag = np.abs(frames)
    peak = float(mag.max()) + 1e-12
    db = 20.0 * np.log10(mag / peak + 1e-12)
    db = np.clip(db, -cfg.ref_db, 0.0)
    norm = (db + cfg.ref_db) / cfg.ref_db
    norm = _fix_shape(norm.astype(np.float32), cfg.n_freq, cfg.n_time)
    return norm[None, :, :]


def batch_iq_to_spectrogram(iq_batch: np.ndarray, cfg: SpectrogramConfig) -> np.ndarray:
    return np.stack([iq_to_spectrogram(iq, cfg) for iq in iq_batch], axis=0)
