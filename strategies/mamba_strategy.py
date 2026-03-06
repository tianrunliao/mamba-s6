import pandas as pd
import torch
import numpy as np
from strategies.base_strategy import BaseStrategy
from models.timing_network import create_mamba_model
from models.training import build_training_data, train_mamba_model
from backtesting.engine import BacktestingEngine


class MambaStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__(config)
        self.model_config = config['model_config']
        self.train_every_n_days = config['train_every_n_days']
        self.rolling_window_days = config['rolling_window_days']
        self.model = None
        self.last_train_date = None

    def get_position_weights(self, date, telemetry, telemetry_history=None, prev_weights=None):
        base_weights = {sym: telemetry[f'weight_{sym}'] for sym in self.symbols if f'weight_{sym}' in telemetry and not pd.isna(telemetry[f'weight_{sym}'])}
        multiplier = self._get_dynamic_exposure(date, telemetry, telemetry_history)
        
        # Neutralize with multiplier target to keep scales consistent for hysteresis
        scaled = self.enforce_market_neutrality(base_weights, target_gross_exposure=multiplier)
        risk_adjusted = self.apply_risk_management(scaled, telemetry, prev_weights=prev_weights, date=date)
        final_weights = self.enforce_market_neutrality(risk_adjusted, target_gross_exposure=multiplier)
        
        return final_weights

    def _get_dynamic_exposure(self, date, telemetry, telemetry_history):
        if self.model is None or not telemetry_history:
            return 1.0

        seq_len = self.model_config['seq_len']
        if len(telemetry_history) < seq_len:
            return 1.0

        history_slice = telemetry_history[-seq_len:]
        seq_features = []
        for t in history_slice:
            feat = [
                t['factor_skewness'],
                t['factor_kurtosis'],
                t['dispersion'],
                t['absorption_ratio'],
                t['portfolio_return_5d'],
                t['portfolio_volatility_5d'],
                t['btc_momentum'],
                t['n_positions_norm'],
            ]
            fw_keys = sorted([k for k in t.keys() if k.startswith('feat_fw_')])
            feat.extend([t[k] for k in fw_keys])
            seq_features.append(feat)

        input_seq = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0)
        actual_dim = input_seq.shape[-1]

        if self.model.input_dim != actual_dim:
            self.model_config['input_dim'] = actual_dim
            self.model = create_mamba_model(self.model_config)
            return 1.0

        with torch.no_grad():
            exposure, regime_probs = self.model(input_seq)
            exposure_val = exposure.item()

        regime = torch.argmax(regime_probs, dim=1).item()
        if regime == 2:
            exposure_val = min(exposure_val, 0.3)
        elif regime == 0:
            exposure_val = min(exposure_val, 1.5)

        return exposure_val

    def get_strategy_name(self):
        return "Mamba Enhanced Strategy"

    def update_model(self, date, telemetry_history, nav_history=None):
        if self.last_train_date is not None and (date - self.last_train_date).days < self.train_every_n_days:
            return

        end_idx = len(telemetry_history)
        start_idx = max(0, end_idx - self.rolling_window_days)
        recent_telemetry = telemetry_history[start_idx:end_idx]

        # nav_history is ignored to remove feedback loops. CIR will be estimated from market factors in training.py
        X, y_cir, y_regime = build_training_data(recent_telemetry, seq_len=self.model_config['seq_len'])

        if X.shape[0] == 0:
            return

        if self.model is None:
            self.model_config['input_dim'] = X.shape[-1]
            self.model = create_mamba_model(self.model_config)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        X, y_cir, y_regime = X.to(device), y_cir.to(device), y_regime.to(device)

        self.model = train_mamba_model(self.model, X, y_cir, y_regime, self.model_config)
        self.last_train_date = date


def run_mamba_strategy(config):
    engine = BacktestingEngine(config)
    strategy = MambaStrategy(config)
    return engine.run_backtest(strategy)
