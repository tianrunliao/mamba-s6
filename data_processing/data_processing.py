"""Data Processing Module Wrapper.

This module provides the process_data function that wraps the data_processor functionality.
"""

import os
import argparse
from .data_processor import BinanceDownloader, FactorEngine, configure_data_paths


def process_data(config):
    """Process data according to configuration.
    
    Args:
        config (dict): Configuration dictionary with keys:
            - symbols: list of symbols to process
            - start_date: start date for data
            - end_date: end date for data  
            - data_dir: base data directory
            - results_dir: results directory
            - download_data: whether to download data
            - calculate_factors: whether to calculate factors
    """
    data_dir = config.get('data_dir', 'data_integrated')
    configure_data_paths(
        data_dir=data_dir,
        spot_dir=config.get('spot_dir'),
        futures_dir=config.get('futures_dir'),
        factors_dir=config.get('factors_dir'),
    )

    # Create data directories if they don't exist
    spot_dir = os.path.join(data_dir, 'spot')
    futures_dir = os.path.join(data_dir, 'futures')
    factors_dir = config.get('factors_dir', os.path.join(data_dir, 'factors'))
    
    for d in [data_dir, spot_dir, futures_dir, factors_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # Download data if requested
    if config.get('download_data', False):
        print("  Downloading data...")
        BinanceDownloader().run(
            start_date=config.get('start_date', '2021-01-01'),
            end_date=config.get('end_date'),
        )
    
    # Calculate factors if requested
    if config.get('calculate_factors', False):
        print("  Calculating factors...")
        FactorEngine(config.get('factor_config_path')).run()


if __name__ == "__main__":
    # For backward compatibility, support command line arguments
    parser = argparse.ArgumentParser(description="Data Processing Module")
    parser.add_argument('--download', action='store_true', help='Download Data')
    parser.add_argument('--recalc', action='store_true', help='Recalculate Factors')
    args = parser.parse_args()
    
    # Create a config dict
    config = {
        'symbols': [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT',
            'DOGEUSDT', 'TRXUSDT', 'DOTUSDT', 'LTCUSDT', 'LINKUSDT', 'AVAXUSDT'
        ],
        'start_date': '2021-01-01',
        'end_date': '2022-12-31',
        'data_dir': 'data_integrated',
        'results_dir': 'results_integrated',
        'download_data': args.download,
        'calculate_factors': args.recalc
    }
    
    process_data(config)
