import os
import random
import sys

import numpy as np
import torch


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from data_processing.data_processing import process_data
from strategies.strategy import run_strategies
from visualization.visualization import generate_report


def main():
    config = {
        'symbols': [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT',
            'DOGEUSDT', 'TRXUSDT', 'DOTUSDT', 'LTCUSDT', 'LINKUSDT', 'AVAXUSDT'
        ],
        'train_end': '2021-12-31',
        'backtest_start': '2022-04-01',
        'start_date': '2021-01-01',
        'end_date': '2026-04-03',
        'data_dir': os.path.join(ROOT_DIR, 'data_integrated'),
        'spot_dir': os.path.join(ROOT_DIR, 'data_integrated', 'spot'),
        'futures_dir': os.path.join(ROOT_DIR, 'data_integrated', 'futures'),
        'factors_dir': os.path.join(ROOT_DIR, 'data_integrated', 'factors'),
        'results_dir': os.path.join(ROOT_DIR, 'results_integrated'),
        'download_data': False,
        'calculate_factors': True,
        'factor_config_path': os.path.join(ROOT_DIR, 'config', 'factor_sets', 'crypto_core.yaml'),
        'portfolio_config': {
            'ic_lookback': 120,
            'min_ic_observations': 40,
            'top_n': 3,
            'max_weight': 0.18,
        },
        'trading_config': {
            'rebalance_band': 0.03,
            'max_weight': 0.18,
            'maker_fee': 0.00005,
            'taker_fee': 0.0004,
            'slippage': 0.0003,
            'maker_offset_min': 0.0008,
            'maker_offset_max': 0.0060,
            'maker_vol_multiplier': 0.35,
            'execution_mode': 'adaptive_maker',
            'max_daily_turnover': 1.20,
            'max_gross_exposure': 1.0,
            'max_net_exposure': 0.06,
            'min_trade_notional': 0.01,
        },
        'baseline_config': {
            'signal_smoothing_window': 9,
            'rebalance_every': 7,
            'drift_threshold': 0.45,
            'momentum_scale': 0.20,
            'momentum_strength': 0.60,
            'vol_floor': 0.015,
            'turnover_blend': 0.55,
            'risk_override_vol': 0.28,
            'risk_override_drawdown': -0.10,
        },
        'equal_weight_config': {
            'rebalance_every': 7,
            'drift_threshold': 0.12,
            'min_momentum_filter': -0.35,
            'max_weight': 0.20,
            'max_gross_exposure': 1.0,
            'max_net_exposure': 1.0,
            'mamba_exposure_floor': 0.25,
            'mamba_exposure_cap': 1.0,
        },
        'mamba_overlay_config': {},
        'model_config': {
            'seq_len': 40,
            'horizon': 10,
            'd_model': 48,
            'd_state': 16,
            'n_layers': 2,
            'train_epochs': 40,
            'fine_tune_epochs': 3,
            'fine_tune_window': 126,
            'fine_tune_freq': 42,
            'learning_rate': 1e-3,
            'fine_tune_lr': 1e-4,
            'lambda_mdd': 0.10,
            'lambda_smooth': 0.05,
            'lambda_regime': 0.2,
            'label_min_exposure': 0.55,
            'label_max_exposure': 1.15,
            'model_max_exposure': 1.25,
            'execution_feature_lag': 2,
        },
    }

    print('Initialising data processor...')
    process_data(config)

    print('\nRunning backtest strategies...')
    results, positions, prices = run_strategies(config)

    print('\nGenerating evaluation report...')
    generate_report(config, results, positions, prices)


if __name__ == '__main__':
    main()
