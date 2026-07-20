import numpy as np
import pytest

from rfgen.dsp.generators import CLASS_NAMES, generate_signal, normalize_power
from rfgen.dsp.impairments import make_anomaly, make_normal
from rfgen.dsp.spectrogram import SpectrogramConfig, iq_to_spectrogram
from rfgen.utils import load_config

CFG = load_config()
FS = float(CFG["signal"]["sample_rate_hz"])
N = int(CFG["signal"]["num_samples"])
SC = SpectrogramConfig.from_config(CFG)


@pytest.mark.parametrize("cls", CLASS_NAMES)
def test_generator_shape_and_finiteness(cls):
    rng = np.random.default_rng(0)
    iq = generate_signal(cls, N, FS, rng, params=CFG["signal"])
    assert iq.shape == (N,)
    assert iq.dtype == np.complex64
    assert np.all(np.isfinite(iq.view(np.float32)))


def test_normalize_power_unit_rms():
    rng = np.random.default_rng(1)
    iq = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(np.complex64)
    out = normalize_power(iq)
    assert np.isclose(np.sqrt(np.mean(np.abs(out) ** 2)), 1.0, atol=1e-4)


@pytest.mark.parametrize("cls", CLASS_NAMES)
def test_spectrogram_shape_and_range(cls):
    rng = np.random.default_rng(2)
    sp = iq_to_spectrogram(make_normal(cls, N, FS, rng, CFG), SC)
    assert sp.shape == (1, SC.n_freq, SC.n_time)
    assert sp.dtype == np.float32
    assert sp.min() >= 0.0 and sp.max() <= 1.0
    assert np.isclose(sp.max(), 1.0, atol=1e-5)


@pytest.mark.parametrize("atype", CFG["data"]["anomaly_types"])
def test_anomaly_generation(atype):
    rng = np.random.default_rng(3)
    sp = iq_to_spectrogram(make_anomaly(atype, N, FS, rng, CFG), SC)
    assert sp.shape == (1, SC.n_freq, SC.n_time)
    assert np.all(np.isfinite(sp))


def test_classes_are_statistically_distinct():
    rng = np.random.default_rng(4)
    means = []
    for cls in CLASS_NAMES:
        arr = np.stack([iq_to_spectrogram(make_normal(cls, N, FS, rng, CFG), SC) for _ in range(20)])
        means.append(arr.mean(0))

    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            assert np.linalg.norm(means[i] - means[j]) > 1.0
