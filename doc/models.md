# Models and evaluation

All three generative models are **class-conditional** and consume the shared
`1×32×32` spectrogram. Each provides `loss(x, y)` for training and a natural
per-example anomaly score.

## Conditional VAE (`cvae.py`) — latent-variable

Convolutional encoder `q(z|x,y)` (32→16→8→4, class embedding concatenated) and a
transpose-conv decoder `p(x|z,y)` with a sigmoid output. Trained by maximising the
ELBO with a Bernoulli reconstruction term and `β·KL` to a standard-normal prior.

- **Likelihood metric:** negative ELBO → an upper bound on NLL, reported in bits/dim.
- **Anomaly score:** negative ELBO (reconstruction + KL). A valid signal
  reconstructs cheaply and stays near the prior; an anomaly does not.

## Conditional RealNVP flow (`flow.py`) — explicit likelihood

An exact invertible map from spectrogram to a standard-normal latent:

1. **Logit preprocessing** maps `[0,1]` pixels to `ℝ` (`u = α+(1−2α)x`, then
   `logit(u)`) with its Jacobian tracked (`α = 0.05`).
2. **8 affine coupling layers** with alternating parity masks; each layer's
   scale/translation MLP is conditioned on a class embedding. Log-scales are
   `tanh`-clamped for stability and initialised to identity.

`log p(x|y) = log N(z;0,I) + Σ log|det|`. Exact, so it yields exact bits/dim.

- **Likelihood metric:** exact test bits/dim (see note below).
- **Anomaly score:** negative log-likelihood. Anomalies fall in the tails.

> **Negative bits/dim** is expected here: peak-normalised spectrograms form a
> concentrated *continuous* distribution whose differential entropy can be
> negative. It does not indicate a bug — the *relative* likelihood separates
> normal from anomalous cleanly (AUC 0.983). The coupling layers' exact
> invertibility is unit-tested (`tests/test_models.py::test_coupling_layer_invertible`).

## Conditional DDPM (`diffusion.py`) — denoising

A compact 3-resolution U-Net (GroupNorm/SiLU ResBlocks, sinusoidal timestep
embedding + class embedding) predicts the noise added at a random diffusion step
(`T=400`, linear β schedule). Class-conditional samples come from ancestral
sampling; `[0,1]` pixels are mapped to `[−1,1]` internally.

- **Likelihood metric:** mean denoising MSE on the test set (surrogate for the
  variational bound).
- **Anomaly score:** mean squared denoising error over a fixed grid of noise
  levels. The network learned to denoise valid signals, so it denoises anomalies
  worse.

## Conditional Mamba autoregressive model (`mamba_ar.py`) — autoregressive, sequence domain

Covers the course's *autoregressive explicit-likelihood* component, but in the
**raw-I/Q sequence domain** rather than on spectrograms. Each I and Q sample is
mu-law companded and quantised to 256 levels; a stack of **Mamba** selective
state-space (SSM) blocks models the token sequence causally, factorising
`p(x) = prod_t p(I_t | past) p(Q_t | past, I_t)` with softmax heads. This gives
an exact **discrete** log-likelihood — a clean, non-negative bits/dim — and
constant-memory recurrent sampling.

- **Why Mamba.** An SSM processes a length-`L` sequence in `O(L)` time with a
  fixed-size state, so it scales to long I/Q windows far better than a quadratic
  Transformer while remaining a proper autoregressive likelihood model.
- **Implementation note.** The selective scan is written in portable pure
  PyTorch (no CUDA `mamba-ssm` kernel) so the project reproduces with
  `pip install`. Training uses an **exact chunked scan** (the length-`chunk`
  inner recurrence is run in parallel across all chunks, with only the
  cross-chunk state hand-off sequential) — mathematically identical to the naive
  recurrence (unit-tested) but ~40x faster here. Generation uses the natural
  recurrent `step()` (constant memory).
- **Likelihood metric:** exact discrete test bits/dim.
- **Anomaly score:** per-example negative log-likelihood on the raw I/Q,
  `min_y NLL(x | y)`.

## Unknown-class scoring

At test time the true class is unknown, so the anomaly score is
`s(x) = min_y s(x|y)` — the best match over the known protocols (`rfgen/eval/common.py`).

## Auxiliary classifier (`classifier.py`)

A small CNN trained on **real** spectrograms (100% test accuracy — the protocols
are well separated). Used to (a) measure **generated-sample classification
accuracy** (does a model's class-`c` sample get classified as `c`?) and (b)
provide penultimate features for a **Fréchet distance** between real and generated
sets (an Inception-Score-style fidelity metric).

## Metrics (`rfgen/eval`)

| Metric | Script | Meaning |
|---|---|---|
| ROC-AUC / AP (overall + per type) | `anomaly.py` | anomaly-detection quality |
| Generated-sample accuracy | `generation.py` | class-conditional fidelity (semantic) |
| Fréchet distance | `generation.py` | distributional fidelity (lower better) |
| bits/dim, ELBO, denoise MSE | `generation.py` | family-appropriate likelihood/reconstruction |
