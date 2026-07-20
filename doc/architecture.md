# Architecture

RF-GenLab is a single Python package (`rfgen`) organised as a linear pipeline.
One YAML config (`configs/default.yaml`) parameterises every stage, so a single
edit changes the whole experiment reproducibly.

```
                 configs/default.yaml
                          │
        ┌─────────────────┼───────────────────────────┐
        ▼                 ▼                             ▼
  rfgen.dsp         rfgen.data.build_dataset      rfgen.models
  (synthesis,   ──▶  (HDF5: train/val/test    ──▶  VAE / Flow / DDPM
   STFT)              normal + anomaly set)         + auxiliary CNN
                          │                             │
                          ▼                             ▼
                   rfgen.data.dataset  ─────────▶  rfgen.train.*
                   (PyTorch loaders)               (checkpoints in runs/)
                                                        │
                                                        ▼
                                                  rfgen.eval.*
                              (anomaly AUC · generation quality · figures · summary)
```

## Design decisions

- **One shared representation.** All models consume the same `1×32×32`
  peak-normalised log-STFT spectrogram, so differences in results are due to the
  *model family*, not the input. Spectrograms make the protocols and the
  anomalies visually and statistically distinct while keeping tensors tiny
  (fast, CPU-friendly training).

- **Class-conditional throughout.** Every generative model is conditioned on the
  protocol class. This exercises the course's *conditional generation* component
  and enables the `min_y s(x|y)` anomaly score (match to the closest known
  protocol) even though the true class is unknown at test time.

- **Pure-NumPy synthesis.** The dataset is generated from seedable NumPy code with
  no MATLAB/SDR dependency, so `pip install -r requirements.txt` is enough to
  reproduce every number.

- **Anomalies are test-only.** Generative models train exclusively on *normal*
  signals; the anomaly set is held out and used only to measure detection AUC.
  This is the realistic unsupervised-anomaly-detection setting.

- **Small but real.** The default corpus (6k train / 1.2k val / 1.5k test normal +
  1.5k anomalies) and model sizes were chosen so the full pipeline runs in minutes
  yet produces genuine, non-degenerate results.

## Reproducibility

`rfgen.utils.set_seed` seeds Python, NumPy and Torch. The dataset builder spawns
independent RNG streams per split via `numpy.random.SeedSequence`, so changing
one split's size leaves the others byte-for-byte identical. Training is seeded and
saves both `best.pt` (lowest val loss) and `last.pt` plus a `history.json`.
