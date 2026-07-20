from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        n_classes: int = 3,
        latent_dim: int = 32,
        base_ch: int = 32,
        beta: float = 1.0,
        emb_dim: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.n_classes = n_classes
        c = base_ch
        self.emb = nn.Embedding(n_classes, emb_dim)

        self.enc = nn.Sequential(
            nn.Conv2d(1, c, 4, 2, 1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, 2 * c, 4, 2, 1), nn.BatchNorm2d(2 * c), nn.ReLU(),
            nn.Conv2d(2 * c, 4 * c, 4, 2, 1), nn.BatchNorm2d(4 * c), nn.ReLU(),
        )
        self.enc_flat = 4 * c * 4 * 4
        self.fc_mu = nn.Linear(self.enc_flat + emb_dim, latent_dim)
        self.fc_lv = nn.Linear(self.enc_flat + emb_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim + emb_dim, self.enc_flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(4 * c, 2 * c, 4, 2, 1), nn.BatchNorm2d(2 * c), nn.ReLU(),
            nn.ConvTranspose2d(2 * c, c, 4, 2, 1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.ConvTranspose2d(c, 1, 4, 2, 1),
        )
        self._c = c


    def encode(self, x: torch.Tensor, y: torch.Tensor):
        h = self.enc(x).flatten(1)
        h = torch.cat([h, self.emb(y)], dim=1)
        return self.fc_mu(h), self.fc_lv(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, y: torch.Tensor):
        h = self.fc_dec(torch.cat([z, self.emb(y)], dim=1))
        h = h.view(-1, 4 * self._c, 4, 4)
        return torch.sigmoid(self.dec(h))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar


    def elbo_terms(self, x, y):
        xhat, mu, logvar = self.forward(x, y)
        recon = F.binary_cross_entropy(xhat, x, reduction="none").flatten(1).sum(1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
        return recon, kl

    def loss(self, x, y):
        recon, kl = self.elbo_terms(x, y)
        loss = (recon + self.beta * kl).mean()
        return loss, {"recon": recon.mean().item(), "kl": kl.mean().item()}

    @torch.no_grad()
    def neg_elbo(self, x, y):
        recon, kl = self.elbo_terms(x, y)
        return recon + kl

    @torch.no_grad()
    def recon_error(self, x, y):
        xhat, _, _ = self.forward(x, y)
        return F.mse_loss(xhat, x, reduction="none").flatten(1).mean(1)

    @torch.no_grad()
    def sample(self, n: int, y: torch.Tensor):
        z = torch.randn(n, self.latent_dim, device=y.device)
        return self.decode(z, y)

    def bits_per_dim(self, x, y):
        recon, kl = self.elbo_terms(x, y)
        d = x[0].numel()
        return ((recon + kl) / (d * math.log(2.0))).mean()
