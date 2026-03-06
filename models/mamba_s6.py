import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = self.d_state

        A = -torch.exp(self.A_log.float())
        x_proj = self.x_proj(x)
        delta_raw, B_ssm, C_ssm = x_proj.split([self.dt_rank, N, N], dim=-1)
        delta = F.softplus(self.dt_proj(delta_raw))

        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            dt_t = delta[:, t, :]
            B_t = B_ssm[:, t, :]
            C_t = C_ssm[:, t, :]
            x_t = x[:, t, :]

            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            h = dA * h + dB * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1) + self.D * x_t
            ys.append(y_t.unsqueeze(1))

        return torch.cat(ys, dim=1)


class MambaBlock(nn.Module):

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=4, padding=3, groups=self.d_inner, bias=True,
        )
        self.ssm = SelectiveSSM(d_model=self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :x.shape[1]]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        x_branch = self.ssm(x_branch)
        z_branch = F.silu(z_branch)
        x_branch = x_branch * z_branch

        return self.out_proj(x_branch) + residual