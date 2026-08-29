import numpy as np
import torch

from models.mamba_model import FEATURE_ORDER, MambaTimingNet, build_training_data
from strategies.strategy_backtest import calculate_dynamic_hysteresis


def _feature_rows(n: int, trailing_vol: float = 0.02, absorption_ratio: float = 0.50):
    rows = []
    for _ in range(n):
        row = {feature: 0.0 for feature in FEATURE_ORDER}
        row["portfolio_volatility_20d"] = trailing_vol
        row["absorption_ratio"] = absorption_ratio
        rows.append(row)
    return rows


def test_mamba_network_exposure_and_regime_shapes():
    model = MambaTimingNet(input_dim=len(FEATURE_ORDER), max_exposure=1.25)
    exposure, regimes = model(torch.zeros(3, 40, len(FEATURE_ORDER)))

    assert exposure.shape == (3, 1)
    assert regimes.shape == (3, 3)
    assert torch.all(exposure >= 0.0)
    assert torch.all(exposure <= 1.25)
    assert torch.allclose(regimes.sum(dim=1), torch.ones(3), atol=1e-6)


def test_positive_path_uses_thesis_label_formula_and_trend_regime():
    seq_len, horizon, n = 4, 3, 30
    returns = np.zeros(n)
    returns[seq_len : seq_len + horizon] = 0.01

    _, exposure, regime, mdd_aux = build_training_data(
        _feature_rows(n), returns.tolist(), seq_len=seq_len, horizon=horizon
    )

    assert np.isclose(exposure[0].item(), 1.15)
    assert regime[0].item() == 0
    assert mdd_aux[0].item() == 1.25


def test_crash_path_caps_exposure_and_sets_risk_regime():
    seq_len, horizon, n = 4, 3, 30
    returns = np.zeros(n)
    returns[seq_len : seq_len + horizon] = [-0.10, 0.0, 0.0]

    _, exposure, regime, mdd_aux = build_training_data(
        _feature_rows(n), returns.tolist(), seq_len=seq_len, horizon=horizon
    )

    assert exposure[0].item() <= 0.65 + 1e-6
    assert regime[0].item() == 2
    assert np.isclose(mdd_aux[0].item(), 0.65)


def test_dynamic_rebalance_band_is_bounded():
    assert calculate_dynamic_hysteresis([0.1] * 10, base_threshold=0.03) == 0.03
    high_recent = [0.01] * 30 + [0.20] * 20
    threshold = calculate_dynamic_hysteresis(high_recent, base_threshold=0.03)
    assert 0.021 <= threshold <= 0.045
