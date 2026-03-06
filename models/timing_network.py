from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_s6 import MambaBlock


class MambaTimingNet(nn.Module):

    def __init__(self, input_dim=8, d_model=32, d_state=16,
                 n_layers=2, seq_len=30, n_regimes=3):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_regimes = n_regimes

        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(d_model)

        self.exposure_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, n_regimes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        h_last = h[:, -1, :]

        exposure = 2.0 * torch.sigmoid(self.exposure_head(h_last))
        regime_probs = F.softmax(self.regime_head(h_last), dim=-1)
        return exposure, regime_probs


def create_mamba_model(config):
    return MambaTimingNet(
        input_dim=config.get('input_dim', 8),
        d_model=config.get('d_model', 32),
        d_state=config.get('d_state', 16),
        n_layers=config.get('n_layers', 2),
        seq_len=config.get('seq_len', 30),
        n_regimes=config.get('n_regimes', 3),
    )