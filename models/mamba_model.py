"""Mamba S6 timing model used by the daily allocation engine."""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.gpu_utils import autocast_context, dataloader_kwargs


FEATURE_ORDER = [
    'factor_skewness',
    'factor_kurtosis',
    'dispersion',
    'absorption_ratio',
    'portfolio_return_5d',
    'portfolio_volatility_20d',
    'portfolio_drawdown_60d',
    'btc_momentum_20d',
    'n_positions_norm',
    'turnover_20d',
]


def resolve_feature_order(feature_order=None):
    return list(feature_order or FEATURE_ORDER)


class SelectiveSSM(nn.Module):
    """Selective State Space Model (S6) core implementation."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(a_init))
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
        batch_size, seq_len, _ = x.shape
        a_matrix = -torch.exp(self.A_log.float())
        delta_raw, b_ssm, c_ssm = self.x_proj(x).split(
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta_raw))

        state = torch.zeros(batch_size, self.d_model, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for step in range(seq_len):
            dt_t = delta[:, step, :]
            b_t = b_ssm[:, step, :]
            c_t = c_ssm[:, step, :]
            x_t = x[:, step, :]

            d_a = torch.exp(dt_t.unsqueeze(-1) * a_matrix.unsqueeze(0))
            d_b = dt_t.unsqueeze(-1) * b_t.unsqueeze(1)
            state = d_a * state + d_b * x_t.unsqueeze(-1)
            y_t = (state * c_t.unsqueeze(1)).sum(dim=-1) + self.D * x_t
            outputs.append(y_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class MambaBlock(nn.Module):
    """Residual Mamba block with depthwise convolution."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2) -> None:
        super().__init__()
        self.d_inner = d_model * expand
        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=4,
            padding=3,
            groups=self.d_inner,
            bias=True,
        )
        self.ssm = SelectiveSSM(d_model=self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x_branch, z_branch = self.in_proj(x).chunk(2, dim=-1)
        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        x_branch = F.silu(x_branch)
        x_branch = self.ssm(x_branch)
        out = self.out_proj(x_branch * F.silu(z_branch))
        return out + residual


class MambaTimingNet(nn.Module):
    """Mamba S6 timing network for exposure control."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_ORDER),
        d_model: int = 48,
        d_state: int = 16,
        n_layers: int = 2,
        seq_len: int = 30,
        n_regimes: int = 3,
        max_exposure: float = 1.25,
    ) -> None:
        super().__init__()
        self.max_exposure = max_exposure
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model=d_model, d_state=d_state) for _ in range(n_layers)])
        self.final_norm = nn.RMSNorm(d_model)
        self.exposure_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
        )
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_regimes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_proj(x)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_norm(hidden)[:, -1, :]
        exposure = self.max_exposure * torch.sigmoid(self.exposure_head(hidden))
        regime_probs = F.softmax(self.regime_head(hidden), dim=-1)
        return exposure, regime_probs


def _future_path_statistics(future_window: np.ndarray) -> dict:
    """Compute the future-path statistics defined in the thesis."""
    mean_ret = float(np.mean(future_window))
    vol_ret = float(np.std(future_window))
    path = np.concatenate(([1.0], np.cumprod(1.0 + future_window)))
    peak = np.maximum.accumulate(path)
    max_drawdown = float(np.min(path / np.maximum(peak, 1e-8) - 1.0))
    quality = mean_ret / (vol_ret + 1e-6) + 1.5 * mean_ret - 1.2 * abs(min(max_drawdown, 0.0))
    return {
        'mean_ret': mean_ret,
        'vol_ret': vol_ret,
        'max_drawdown': max_drawdown,
        'quality': float(quality),
    }


def build_training_data(
    feature_history: list,
    baseline_returns: list,
    feature_order=None,
    seq_len: int = 30,
    horizon: int = 10,
    min_exposure: float = 0.55,
    max_exposure: float = 1.15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build samples from historical features and future baseline returns."""
    feature_order = resolve_feature_order(feature_order)
    n_obs = min(len(feature_history), len(baseline_returns))
    if n_obs < seq_len + horizon + 20:
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

    x_list = []
    sample_stats = []

    for idx in range(seq_len, n_obs - horizon):
        seq = []
        for hist_idx in range(idx - seq_len, idx):
            row = feature_history[hist_idx]
            seq.append([float(row.get(col, 0.0)) for col in feature_order])
        x_list.append(seq)

        future_window = np.asarray(baseline_returns[idx:idx + horizon], dtype=float)
        sample_stats.append(_future_path_statistics(future_window))

    y_exposure_list = []
    y_regime_list = []
    y_mdd_aux_list = []
    for sample_idx, stats in enumerate(sample_stats):
        quality = stats['quality']
        max_drawdown = stats['max_drawdown']
        vol_ret = stats['vol_ret']
        mean_ret = stats['mean_ret']

        exposure_label = 0.85 + 0.30 * np.tanh(2.5 * quality)
        if max_drawdown < -0.08:
            exposure_label = min(exposure_label, 0.65)
        y_exposure_list.append([float(np.clip(exposure_label, min_exposure, max_exposure))])
        y_mdd_aux_list.append([0.65 if max_drawdown < -0.08 else 1.25])

        current = feature_history[seq_len + sample_idx - 1]
        absorption_ratio = float(current.get('absorption_ratio', 0.5))
        trailing_vol = float(current.get('portfolio_volatility_20d', 0.0))
        if (
            max_drawdown < -0.08
            or (
                absorption_ratio > 0.78
                and (vol_ret > 1.8 * trailing_vol or vol_ret > 0.04)
            )
        ):
            regime = 2
        elif (
            mean_ret > 0.003
            and max_drawdown > -0.03
            and vol_ret < 1.2 * trailing_vol
        ):
            regime = 0
        else:
            regime = 1
        y_regime_list.append(regime)

    x_tensor = torch.tensor(x_list, dtype=torch.float32)
    y_exposure_tensor = torch.tensor(y_exposure_list, dtype=torch.float32)
    y_regime_tensor = torch.tensor(y_regime_list, dtype=torch.long)
    y_mdd_aux_tensor = torch.tensor(y_mdd_aux_list, dtype=torch.float32)
    return x_tensor, y_exposure_tensor, y_regime_tensor, y_mdd_aux_tensor


def differentiable_mdd_penalty(
    exposures: torch.Tensor,
    target_exposures: torch.Tensor,
    lambda_mdd: float = 0.1,
    lambda_smooth: float = 0.05,
) -> torch.Tensor:
    """Penalize excessive leverage in poor future states and noisy exposure paths."""
    downside_penalty = lambda_mdd * torch.mean(torch.relu(exposures - target_exposures) ** 2)
    if exposures.shape[0] > 1:
        smoothness_penalty = lambda_smooth * torch.mean(torch.abs(torch.diff(exposures.squeeze(-1))))
    else:
        smoothness_penalty = torch.tensor(0.0, device=exposures.device)
    return downside_penalty + smoothness_penalty


def train_mamba_net_enhanced(
    net: MambaTimingNet,
    x_data: torch.Tensor,
    y_exposure: torch.Tensor,
    y_regime: torch.Tensor,
    aux_targets: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    lambda_mdd: float = 0.1,
    lambda_smooth: float = 0.05,
    lambda_regime: float = 0.2,
    batch_size: int = 256,
    use_amp: bool = True,
    verbose: bool = True,
) -> MambaTimingNet:
    """Train the Mamba timing model."""
    device = next(net.parameters()).device
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=1e-5,
    )
    use_amp = use_amp and device.type == 'cuda'
    try:
        scaler = torch.amp.GradScaler(device='cuda', enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    dataset = TensorDataset(x_data, y_exposure, y_regime, aux_targets)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, max(len(dataset), 1)),
        shuffle=True,
        drop_last=False,
        **dataloader_kwargs(device),
    )

    net.train()
    for epoch in range(epochs):
        epoch_exposure = 0.0
        epoch_regime = 0.0
        epoch_regularizer = 0.0
        epoch_batches = 0

        for x_batch, y_exp_batch, y_reg_batch, aux_batch in loader:
            x_batch = x_batch.to(device, non_blocking=device.type == 'cuda')
            y_exp_batch = y_exp_batch.to(device, non_blocking=device.type == 'cuda')
            y_reg_batch = y_reg_batch.to(device, non_blocking=device.type == 'cuda')
            aux_batch = aux_batch.to(device, non_blocking=device.type == 'cuda')

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=use_amp):
                pred_exposure, pred_regime = net(x_batch)
                exposure_loss = F.smooth_l1_loss(pred_exposure, y_exp_batch)
                regime_loss = F.cross_entropy(pred_regime, y_reg_batch)
                regularizer = differentiable_mdd_penalty(
                    pred_exposure,
                    aux_batch,
                    lambda_mdd=lambda_mdd,
                    lambda_smooth=lambda_smooth,
                )
                total_loss = exposure_loss + lambda_regime * regime_loss + regularizer

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_exposure += float(exposure_loss.detach().item())
            epoch_regime += float(regime_loss.detach().item())
            epoch_regularizer += float(regularizer.detach().item())
            epoch_batches += 1

        scheduler.step()

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"    [Mamba Net] Epoch {epoch + 1}/{epochs} "
                f"| exposure={epoch_exposure / max(epoch_batches, 1):.5f} "
                f"| regime={epoch_regime / max(epoch_batches, 1):.5f} "
                f"| reg={epoch_regularizer / max(epoch_batches, 1):.5f}"
            )

    net.eval()
    return net


def train_mamba_net(
    net: MambaTimingNet,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    verbose: bool = True,
) -> MambaTimingNet:
    """Backward-compatible lightweight training wrapper."""
    return train_mamba_net_enhanced(
        net=net,
        x_data=x_data,
        y_exposure=y_data,
        y_regime=torch.zeros(x_data.shape[0], dtype=torch.long, device=x_data.device),
        aux_targets=y_data,
        epochs=epochs,
        lr=lr,
        lambda_regime=0.0,
        batch_size=min(256, max(x_data.shape[0], 1)),
        verbose=verbose,
    )


def create_mamba_model(config):
    """Create a Mamba timing model from config."""
    return MambaTimingNet(
        input_dim=config.get('input_dim', len(FEATURE_ORDER)),
        d_model=config.get('d_model', 48),
        d_state=config.get('d_state', 16),
        n_layers=config.get('n_layers', 2),
        seq_len=config.get('seq_len', 30),
        n_regimes=config.get('n_regimes', 3),
        max_exposure=config.get('max_exposure', 1.25),
    )


def train_mamba_model(model, x_data, y_exposure, y_regime, aux_targets, config):
    """Train the Mamba model using a config dictionary."""
    return train_mamba_net_enhanced(
        net=model,
        x_data=x_data,
        y_exposure=y_exposure,
        y_regime=y_regime,
        aux_targets=aux_targets,
        epochs=config.get('train_epochs', 100),
        lr=config.get('learning_rate', 1e-3),
        lambda_mdd=config.get('lambda_mdd', 0.1),
        lambda_smooth=config.get('lambda_smooth', 0.05),
        lambda_regime=config.get('lambda_regime', 0.2),
        batch_size=config.get('batch_size', 256),
        use_amp=config.get('use_amp', True),
        verbose=config.get('verbose', True),
    )
