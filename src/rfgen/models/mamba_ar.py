from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rfgen.dsp.quantize import detokenize_iq, tokenize_iq
from rfgen.dsp.spectrogram import SpectrogramConfig, iq_to_spectrogram


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def _ssm_params(self, xc: torch.Tensor):
        """xc [B, L, d_inner] (post-conv, post-SiLU) -> dt, B, C."""
        dbl = self.x_proj(xc)
        dt, B, C = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                    # [B, L, d_inner]
        return dt, B, C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, L, _ = x.shape
        x_ssm, z = self.in_proj(x).chunk(2, dim=-1)          # each [B, L, d_inner]
        xc = self.conv1d(x_ssm.transpose(1, 2))[..., :L].transpose(1, 2)
        xc = F.silu(xc)
        dt, Bm, Cm = self._ssm_params(xc)
        A = -torch.exp(self.A_log)                           # [d_inner, d_state]
        dA = torch.exp(dt.unsqueeze(-1) * A)                 # [B, L, d_inner, d_state]
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * xc.unsqueeze(-1)
        y = self._scan(dA, dBx, Cm)                          # [B, L, d_inner]
        y = y + self.D * xc
        y = y * F.silu(z)
        return self.out_proj(y)

    @staticmethod
    def _scan(dA: torch.Tensor, dBx: torch.Tensor, C: torch.Tensor, chunk: int = 16) -> torch.Tensor:
        B_, L, E, N = dA.shape
        if L % chunk != 0:
            chunk = L
        nc = L // chunk
        dA_c = dA.view(B_, nc, chunk, E, N)
        dBx_c = dBx.view(B_, nc, chunk, E, N)

        h = torch.zeros(B_, nc, E, N, device=dA.device, dtype=dA.dtype)
        P = torch.ones(B_, nc, E, N, device=dA.device, dtype=dA.dtype)
        h0, Pc = [], []
        for t in range(chunk):
            h = dA_c[:, :, t] * h + dBx_c[:, :, t]
            P = P * dA_c[:, :, t]
            h0.append(h); Pc.append(P)
        h0 = torch.stack(h0, dim=2)            # [B, nc, chunk, E, N]
        Pc = torch.stack(Pc, dim=2)

        # 2) sequential state hand-off between chunks (nc steps).
        h_end, P_end = h0[:, :, -1], Pc[:, :, -1]   # [B, nc, E, N]
        hin = torch.zeros(B_, E, N, device=dA.device, dtype=dA.dtype)
        hins = []
        for c in range(nc):
            hins.append(hin)
            hin = h_end[:, c] + P_end[:, c] * hin
        hin = torch.stack(hins, dim=1)              # [B, nc, E, N]

        # 3) combine and read out.
        h_full = (h0 + Pc * hin.unsqueeze(2)).view(B_, L, E, N)
        return (h_full * C.unsqueeze(2)).sum(-1)

    def init_state(self, B_: int, device):
        return (torch.zeros(B_, self.d_inner, self.d_conv, device=device),
                torch.zeros(B_, self.d_inner, self.d_state, device=device))

    def step(self, x_t: torch.Tensor, state):
        conv_buf, h = state
        x_ssm, z = self.in_proj(x_t).chunk(2, dim=-1)
        conv_buf = torch.cat([conv_buf[:, :, 1:], x_ssm.unsqueeze(-1)], dim=-1)
        w = self.conv1d.weight.squeeze(1)
        xc = (conv_buf * w).sum(-1) + self.conv1d.bias
        xc = F.silu(xc)
        dbl = self.x_proj(xc)
        dt, Bm, Cm = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(1) * xc.unsqueeze(-1)
        h = dA * h + dBx
        y = (h * Cm.unsqueeze(1)).sum(-1) + self.D * xc
        y = y * F.silu(z)
        return self.out_proj(y), (conv_buf, h)


class ConditionalMambaAR(nn.Module):
    def __init__(
        self,
        n_classes: int = 3,
        n_samples: int = 512,
        levels: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        spec_cfg: SpectrogramConfig | None = None,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.levels = levels
        self.spec_cfg = spec_cfg or SpectrogramConfig()
        self.emb_i = nn.Embedding(levels, d_model)
        self.emb_q = nn.Embedding(levels, d_model)
        self.cls_emb = nn.Embedding(n_classes, d_model)
        self.start = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.head_i = nn.Linear(d_model, levels)
        self.head_q = nn.Linear(d_model, levels)
        self.d_model = d_model

    def _backbone(self, inp: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        h = inp
        for blk, norm in zip(self.blocks, self.norms):
            if use_checkpoint and self.training:
                h = h + checkpoint(blk, norm(h), use_reentrant=False)
            else:
                h = h + blk(norm(h))
        return self.norm_f(h)

    def _logits(self, iq2: torch.Tensor, y: torch.Tensor):
        """iq2 [B, 2, N] real -> (logits_i, logits_q, tok_i, tok_q)."""
        tok = tokenize_iq(iq2, self.levels)
        tok_i, tok_q = tok[:, 0], tok[:, 1]
        emb_seq = self.emb_i(tok_i) + self.emb_q(tok_q)
        start = self.start.expand(iq2.shape[0], -1, -1)
        inp = torch.cat([start, emb_seq[:, :-1]], dim=1)
        inp = inp + self.cls_emb(y).unsqueeze(1)
        h = self._backbone(inp, use_checkpoint=False)
        logits_i = self.head_i(h)
        logits_q = self.head_q(h + self.emb_i(tok_i))
        return logits_i, logits_q, tok_i, tok_q

    def _nll_per_example(self, iq2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits_i, logits_q, tok_i, tok_q = self._logits(iq2, y)
        li = F.cross_entropy(logits_i.transpose(1, 2), tok_i, reduction="none").sum(1)
        lq = F.cross_entropy(logits_q.transpose(1, 2), tok_q, reduction="none").sum(1)
        return li + lq

    def loss(self, iq2: torch.Tensor, y: torch.Tensor):
        nll = self._nll_per_example(iq2, y).mean()
        bpd = nll / (2 * self.n_samples * math.log(2.0))
        return nll, {"bpd": bpd.item()}

    @torch.no_grad()
    def nll_iq(self, iq2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._nll_per_example(iq2, y)

    def bits_per_dim(self, iq2: torch.Tensor, y: torch.Tensor):
        return self._nll_per_example(iq2, y).mean() / (2 * self.n_samples * math.log(2.0))

    @torch.no_grad()
    def sample_iq(self, n: int, y: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        device = y.device
        states = [blk.init_state(n, device) for blk in self.blocks]
        cls = self.cls_emb(y)
        x_t = self.start.expand(n, -1, -1).squeeze(1) + cls  # [n, D]
        toks_i, toks_q = [], []
        for _ in range(self.n_samples):
            h = x_t
            for j, (blk, norm) in enumerate(zip(self.blocks, self.norms)):
                out, states[j] = blk.step(norm(h), states[j])
                h = h + out
            h = self.norm_f(h)
            pi = F.softmax(self.head_i(h) / temperature, dim=-1)
            ti = torch.multinomial(pi, 1).squeeze(1)
            pq = F.softmax(self.head_q(h + self.emb_i(ti)) / temperature, dim=-1)
            tq = torch.multinomial(pq, 1).squeeze(1)
            toks_i.append(ti); toks_q.append(tq)
            x_t = self.emb_i(ti) + self.emb_q(tq) + cls
        tok = torch.stack([torch.stack(toks_i, 1), torch.stack(toks_q, 1)], dim=1)
        return detokenize_iq(tok, self.levels)
    @torch.no_grad()
    def sample(self, n: int, y: torch.Tensor):
        iq2 = self.sample_iq(n, y).cpu().numpy()
        specs = [iq_to_spectrogram(iq2[k, 0] + 1j * iq2[k, 1], self.spec_cfg) for k in range(n)]
        import numpy as np
        return torch.from_numpy(np.stack(specs, 0)).float()
