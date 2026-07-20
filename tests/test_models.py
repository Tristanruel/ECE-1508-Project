import torch

from rfgen.models import (
    ConditionalDDPM,
    ConditionalMambaAR,
    ConditionalRealNVP,
    ConditionalVAE,
    SpectrogramCNN,
)
from rfgen.models.flow import CouplingLayer
from rfgen.models.mamba_ar import MambaBlock

B = 8


def _batch():
    torch.manual_seed(0)
    return torch.rand(B, 1, 32, 32), torch.randint(0, 3, (B,))


def test_classifier_forward():
    x, _ = _batch()
    clf = SpectrogramCNN()
    assert clf(x).shape == (B, 3)
    assert clf.features(x).shape == (B, clf.feat_dim)


def test_vae_loss_and_sample():
    x, y = _batch()
    vae = ConditionalVAE()
    loss, logs = vae.loss(x, y)
    assert loss.requires_grad and torch.isfinite(loss)
    assert set(logs) == {"recon", "kl"}
    assert vae.sample(4, y[:4]).shape == (4, 1, 32, 32)
    assert vae.neg_elbo(x, y).shape == (B,)


def test_flow_log_prob_and_sample():
    x, y = _batch()
    flow = ConditionalRealNVP(n_coupling=4)
    lp = flow.log_prob(x, y)
    assert lp.shape == (B,) and torch.all(torch.isfinite(lp))
    s = flow.sample(4, y[:4])
    assert s.shape == (4, 1, 32, 32)
    assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0


def test_coupling_layer_invertible():
    torch.manual_seed(1)
    dim = 64
    mask = (torch.arange(dim) % 2).float()
    layer = CouplingLayer(dim, cond_dim=8, hidden=32, mask=mask)

    for p in layer.parameters():
        p.data = torch.randn_like(p) * 0.1
    x = torch.randn(5, dim)
    cond = torch.randn(5, 8)
    z, log_det = layer(x, cond)
    x_rec = layer.inverse(z, cond)
    assert torch.allclose(x, x_rec, atol=1e-4)
    assert log_det.shape == (5,)


def test_diffusion_loss_and_denoise():
    x, y = _batch()
    ddpm = ConditionalDDPM(timesteps=50)
    loss, logs = ddpm.loss(x, y)
    assert loss.requires_grad and torch.isfinite(loss)
    assert ddpm.denoising_error(x, y).shape == (B,)


def test_diffusion_sample_small():
    _, y = _batch()
    ddpm = ConditionalDDPM(timesteps=20)
    s = ddpm.sample(2, y[:2])
    assert s.shape == (2, 1, 32, 32)
    assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0


def test_mamba_chunked_scan_matches_sequential():
    torch.manual_seed(2)
    Bn, L, E, N = 2, 64, 4, 3
    dA = torch.rand(Bn, L, E, N) * 0.2 + 0.75
    dBx = torch.randn(Bn, L, E, N) * 0.1
    C = torch.randn(Bn, L, N)
    h = torch.zeros(Bn, E, N)
    ref = []
    for t in range(L):
        h = dA[:, t] * h + dBx[:, t]
        ref.append((h * C[:, t].unsqueeze(1)).sum(-1))
    ref = torch.stack(ref, 1)
    got = MambaBlock._scan(dA, dBx, C, chunk=16)
    assert torch.allclose(ref, got, atol=1e-4)


def test_mamba_loss_and_sample():
    torch.manual_seed(0)
    iq = torch.rand(4, 2, 128) * 2 - 1
    y = torch.randint(0, 3, (4,))
    m = ConditionalMambaAR(n_classes=3, n_samples=128, d_model=32, n_layers=2, d_state=4)
    loss, logs = m.loss(iq, y)
    assert loss.requires_grad and torch.isfinite(loss)
    assert "bpd" in logs and 0.0 < logs["bpd"] < 12.0   # finite, ~log2(256) at init
    assert m.nll_iq(iq, y).shape == (4,)
    sp = m.sample(2, y[:2])
    assert sp.shape == (2, 1, 32, 32)
    assert float(sp.min()) >= 0.0 and float(sp.max()) <= 1.0
