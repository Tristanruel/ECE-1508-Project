from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, groups: int = 8):
        super().__init__()
        g_in = math.gcd(groups, in_ch)
        g_out = math.gcd(groups, out_ch)
        self.norm1 = nn.GroupNorm(g_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(g_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):

    def __init__(self, n_classes: int = 3, base_ch: int = 32):
        super().__init__()
        c = base_ch
        td = 4 * c
        self.temb_mlp = nn.Sequential(nn.Linear(c, td), nn.SiLU(), nn.Linear(td, td))
        self.cls_emb = nn.Embedding(n_classes, td)
        self.base_ch = c

        self.in_conv = nn.Conv2d(1, c, 3, padding=1)
        self.res1 = ResBlock(c, c, td)
        self.down1 = nn.Conv2d(c, 2 * c, 4, 2, 1)
        self.res2 = ResBlock(2 * c, 2 * c, td)
        self.down2 = nn.Conv2d(2 * c, 4 * c, 4, 2, 1)
        self.res_mid = ResBlock(4 * c, 4 * c, td)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 4, 2, 1)
        self.res3 = ResBlock(4 * c, 2 * c, td)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 4, 2, 1)
        self.res4 = ResBlock(2 * c, c, td)
        self.out_norm = nn.GroupNorm(math.gcd(8, c), c)
        self.out_conv = nn.Conv2d(c, 1, 3, padding=1)

    def forward(self, x, t, y):
        temb = self.temb_mlp(timestep_embedding(t, self.base_ch)) + self.cls_emb(y)
        h = self.in_conv(x)
        r1 = self.res1(h, temb)
        r2 = self.res2(self.down1(r1), temb)
        b = self.res_mid(self.down2(r2), temb)
        u2 = self.res3(torch.cat([self.up2(b), r2], dim=1), temb)
        u1 = self.res4(torch.cat([self.up1(u2), r1], dim=1), temb)
        return self.out_conv(F.silu(self.out_norm(u1)))


class ConditionalDDPM(nn.Module):
    def __init__(
        self,
        n_classes: int = 3,
        base_ch: int = 32,
        timesteps: int = 400,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.T = timesteps
        self.net = ConditionalUNet(n_classes, base_ch)
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = F.pad(acp[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("acp", acp)
        self.register_buffer("sqrt_acp", torch.sqrt(acp))
        self.register_buffer("sqrt_one_minus_acp", torch.sqrt(1 - acp))
        self.register_buffer("posterior_var", betas * (1 - acp_prev) / (1 - acp))

    def q_sample(self, x0, t, noise):
        return (
            self.sqrt_acp[t][:, None, None, None] * x0
            + self.sqrt_one_minus_acp[t][:, None, None, None] * noise
        )

    def loss(self, x, y):
        x0 = x * 2.0 - 1.0
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.net(xt, t, y)
        loss = F.mse_loss(pred, noise)
        return loss, {"mse": loss.item()}

    @torch.no_grad()
    def sample(self, n: int, y: torch.Tensor):
        device = y.device
        x = torch.randn(n, 1, 32, 32, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            eps = self.net(x, t, y)
            alpha = self.alphas[i]
            acp = self.acp[i]
            mean = (x - (1 - alpha) / torch.sqrt(1 - acp) * eps) / torch.sqrt(alpha)
            if i > 0:
                x = mean + torch.sqrt(self.posterior_var[i]) * torch.randn_like(x)
            else:
                x = mean
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)

    @torch.no_grad()
    def denoising_error(self, x, y, t_grid: list[int] | None = None, seed: int = 0):
        x0 = x * 2.0 - 1.0
        b = x0.shape[0]
        if t_grid is None:
            t_grid = [int(self.T * f) for f in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)]
        g = torch.Generator(device=x0.device).manual_seed(seed)
        errs = torch.zeros(b, device=x0.device)
        for tv in t_grid:
            t = torch.full((b,), tv, device=x0.device, dtype=torch.long)
            noise = torch.randn(x0.shape, generator=g, device=x0.device)
            xt = self.q_sample(x0, t, noise)
            pred = self.net(xt, t, y)
            errs += (pred - noise).pow(2).flatten(1).mean(1)
        return errs / len(t_grid)
