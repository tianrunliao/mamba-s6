"""Strategy wrappers for the enhanced baseline, Mamba, and equal-weight variants."""

import os
from glob import glob

import pandas as pd

from .strategy_backtest import run_all_strategies


OUTPUT_FILE_MAP = {
    'baseline': 'baseline',
    'mamba': 'mamba',
    'equal_weight': 'equal_weight',
    'mamba_equal_weight': 'mamba_equal_weight',
}


def run_strategies(config):
    """Run all unified strategies and persist their outputs."""
    results, telemetry, positions, prices = run_all_strategies(config)

    results_dir = config['results_dir']
    for strategy_name, file_stub in OUTPUT_FILE_MAP.items():
        res_df = results.get(strategy_name, pd.DataFrame())
        telem_df = telemetry.get(strategy_name, pd.DataFrame())
        pos_df = positions.get(strategy_name, pd.DataFrame())
        if not res_df.empty:
            res_df.to_csv(os.path.join(results_dir, f'{file_stub}_nav.csv'))
        if not telem_df.empty:
            telem_df.to_csv(os.path.join(results_dir, f'{file_stub}_telemetry.csv'))
        if not pos_df.empty:
            pos_df.to_csv(os.path.join(results_dir, f'{file_stub}_positions.csv'))

    if results.get('baseline', pd.DataFrame()).empty:
        print(f'  No strategy results were saved to {results_dir}')
    else:
        print(f'  Strategy results saved to {results_dir}')

    if not prices:
        for file_path in glob(os.path.join(config['futures_dir'], '*.csv')):
            symbol = os.path.basename(file_path).replace('.csv', '')
            price_df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            prices[symbol] = price_df['close'].resample('D').last()

    return results, positions, prices
