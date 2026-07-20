from __future__ import annotations

import torch


def mu_law_encode(x: torch.Tensor, mu: float = 255.0) -> torch.Tensor:

    return torch.sign(x) * torch.log1p(mu * x.abs()) / torch.log1p(torch.tensor(mu))


def mu_law_decode(y: torch.Tensor, mu: float = 255.0) -> torch.Tensor:

    return torch.sign(y) * torch.expm1(y.abs() * torch.log1p(torch.tensor(mu))) / mu


def tokenize_iq(iq2: torch.Tensor, levels: int = 256, eps: float = 1e-8) -> torch.Tensor:

    m = iq2.abs().amax(dim=(1, 2), keepdim=True) + eps
    y = mu_law_encode(iq2 / m)
    q = torch.round((y + 1.0) * 0.5 * (levels - 1))
    return q.clamp(0, levels - 1).long()


def detokenize_iq(tok: torch.Tensor, levels: int = 256) -> torch.Tensor:

    y = tok.float() / (levels - 1) * 2.0 - 1.0
    return mu_law_decode(y)
