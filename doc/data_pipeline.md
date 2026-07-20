# Data pipeline

## Signal synthesis (`rfgen/dsp/generators.py`)

All signals are complex baseband I/Q, unit average power, `num_samples` long
(default 512) at a nominal `sample_rate_hz` (default 20 MHz). Three protocol
families are modelled as compact, *protocol-like* waveforms (not standards-complete
PHYs):

| Class | Modulation | Key structure |
|---|---|---|
| `wifi` | OFDM | 64-pt IFFT, 16-sample cyclic prefix, 52 active QPSK sub-carriers, boosted comb pilots every 8th carrier → **wideband, block-structured** |
| `ble` | GFSK | Gaussian-smoothed 2-FSK, symbol rate 1 MHz, deviation 250 kHz (modulation index ≈ 0.5) → **narrowband constant-envelope tone** |
| `zigbee` | O-QPSK / DSSS | 16-chip spreading sequence, 2 Mchip/s, half-chip Q offset, RRC pulse shaping → **medium-band chip-structured** |

## Nuisance variation (normal signals)

Applied by `make_normal` so the models learn a distribution rather than one
waveform, all bounded to a "healthy" regime:

- AWGN at SNR ∈ `[8, 30]` dB,
- carrier-frequency offset ∈ `[−0.02, 0.02]` cycles/sample,
- gain ∈ `[−3, 3]` dB (removed again by peak normalisation, kept for realism).

## Anomalies (test-only, `make_anomaly`)

Each anomaly is a valid signal pushed **out of distribution**:

| Type | Construction |
|---|---|
| `heavy_noise` | valid signal at SNR ∈ `[−6, 4]` dB (buried in noise) |
| `freq_shift` | large carrier offset `|Δf|` ∈ `[0.28, 0.45]` cycles/sample (energy at the band edge) |
| `timing_corrupt` | 2–4 blanking gaps + a spliced/repeated chunk (broken packet timing) |
| `protocol_mix` | two different protocols summed at comparable power (feature collision) |

## Spectrogram (`rfgen/dsp/spectrogram.py`)

A two-sided STFT (`nperseg=32`, `hop=16`, Hann window, `fftshift` so DC is
centred) → magnitude → dB relative to the per-image peak → clipped to a
`ref_db=45` dB floor → mapped to `[0, 1]`. Output is `1×n_freq×n_time` = `1×32×32`.
Peak (not global) normalisation makes the representation invariant to absolute
gain, so the models focus on spectro-temporal structure.

## HDF5 dataset (`rfgen/data/build_dataset.py`)

`data/dataset.h5` groups:

- `train`, `val`, `test` — normal signals, each with `x` (`[N,1,32,32]` float32
  spectrograms), `y` (`[N]` class index), `iq` (`[N,512]` complex64 raw I/Q);
- `anom` — anomalies with `x`, `iq`, and `y_type` (anomaly-type index).

File attributes store `class_names`, `anomaly_types`, the spectrogram config and
a full config snapshot, so the dataset is self-describing.
