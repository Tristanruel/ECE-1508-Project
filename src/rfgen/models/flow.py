from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG2PI = math.log(2.0 * math.pi)
_ALPHA = 0.05


class _CouplingNet(nn.Module):

    def __init__(self, dim: int, cond_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * dim),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_masked, cond):
        h = self.net(torch.cat([x_masked, cond], dim=1))
        s, t = h.chunk(2, dim=1)
        return s, t


class CouplingLayer(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = _CouplingNet(dim, cond_dim, hidden)
        self.log_scale_clamp = 2.0

    def forward(self, x, cond):
        b = self.mask
        x_b = x * b
        s, t = self.net(x_b, cond)
        s = self.log_scale_clamp * torch.tanh(s)
        s = s * (1 - b)
        t = t * (1 - b)
        z = x_b + (1 - b) * ((x - t) * torch.exp(-s))
        log_det = -s.sum(dim=1)
        return z, log_det

    def inverse(self, z, cond):
        b = self.mask
        z_b = z * b
        s, t = self.net(z_b, cond)
        s = self.log_scale_clamp * torch.tanh(s)
        s = s * (1 - b)
        t = t * (1 - b)
        x = z_b + (1 - b) * (z * torch.exp(s) + t)
        return x


class ConditionalRealNVP(nn.Module):
    def __init__(
        self,
        n_classes: int = 3,
        img_shape: tuple[int, int, int] = (1, 32, 32),
        n_coupling: int = 8,
        hidden: int = 256,
        emb_dim: int = 32,
    ):
        super().__init__()
        self.img_shape = img_shape
        self.dim = int(img_shape[0] * img_shape[1] * img_shape[2])
        self.emb = nn.Embedding(n_classes, emb_dim)

        base = torch.arange(self.dim) % 2
        layers = []
        for i in range(n_coupling):
            mask = base if i % 2 == 0 else (1 - base)
            layers.append(CouplingLayer(self.dim, emb_dim, hidden, mask.float()))
        self.layers = nn.ModuleList(layers)


    def _logit_forward(self, x):
        u = _ALPHA + (1 - 2 * _ALPHA) * x
        z = torch.log(u) - torch.log1p(-u)
        log_det = (
            math.log(1 - 2 * _ALPHA) - torch.log(u) - torch.log1p(-u)
        ).sum(dim=1)
        return z, log_det

    def _logit_inverse(self, z):
        u = torch.sigmoid(z)
        x = (u - _ALPHA) / (1 - 2 * _ALPHA)
        return x.clamp(0.0, 1.0)


    def log_prob(self, x, y):
        x = x.flatten(1)
        cond = self.emb(y)
        z, ldj = self._logit_forward(x)
        for layer in self.layers:
            z, ld = layer(z, cond)
            ldj = ldj + ld
        log_pz = (-0.5 * (z ** 2 + _LOG2PI)).sum(dim=1)
        return log_pz + ldj

    def loss(self, x, y):
        lp = self.log_prob(x, y)
        nll = -lp.mean()
        bpd = nll / (self.dim * math.log(2.0))
        return nll, {"bpd": bpd.item()}

    @torch.no_grad()
    def nll(self, x, y):
        return -self.log_prob(x, y)

    def bits_per_dim(self, x, y):
        return (-self.log_prob(x, y)).mean() / (self.dim * math.log(2.0))

    @torch.no_grad()
    def sample(self, n: int, y: torch.Tensor):
        cond = self.emb(y)
        z = torch.randn(n, self.dim, device=y.device)
        for layer in reversed(self.layers):
            z = layer.inverse(z, cond)
        x = self._logit_inverse(z)
        return x.view(n, *self.img_shape)
