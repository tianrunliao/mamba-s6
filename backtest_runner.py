import os
import sys
import argparse
import yaml

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.entropy_strategy import run_entropy_only_strategy
from strategies.mamba_strategy import run_mamba_strategy
from visualization.report_generator import generate_report


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    return {
        'symbols': ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI', 'ATOM'],
        'data_dir': 'data_integrated',
        'results_dir': 'results',
        'realistic_fee_rate': 0.0002,
        'slippage_rate': 0.0003,
        'cooling_period': 48,
        'hysteresis_threshold': 0.08,
        'ar_stop_threshold': 0.8,
        'model_config': {
            'd_model': 64,
            'n_layers': 4,
            'seq_len': 30,
            'dropout': 0.1,
            'train_epochs': 100,
            'learning_rate': 1e-3,
            'lambda_regime': 0.2,
            'verbose': True
        },
        'train_every_n_days': 7,
        'rolling_window_days': 252
    }


def main():
    parser = argparse.ArgumentParser(description='Mamba Crypto Strategy')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--data-dir', type=str, default='data_integrated', help='Directory containing processed data')
    parser.add_argument('--symbols', type=str, nargs='+', help='List of symbols to trade')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.results_dir != 'results':
        config['results_dir'] = args.results_dir
    if args.data_dir != 'data_integrated':
        config['data_dir'] = args.data_dir
    if args.symbols:
        config['symbols'] = args.symbols
        
    print(f"Starting Backtest. Config: {config}")
    
    print("\n1. Running Entropy-Only Baseline Strategy...")
    entropy_results = run_entropy_only_strategy(config)
    entropy_end_nav = entropy_results['nav'][-1] if entropy_results['nav'] else 1.0
    print(f"Entropy strategy completed! Final NAV: {entropy_end_nav:.4f}")
    
    print("\n2. Running Enhanced Mamba Strategy...")
    mamba_results = run_mamba_strategy(config)
    mamba_end_nav = mamba_results['nav'][-1] if mamba_results['nav'] else 1.0
    print(f"Mamba strategy completed! Final NAV: {mamba_end_nav:.4f}")
    
    print("\n3. Generating Comprehensive Report...")
    generate_report(config, mamba_results, entropy_results)
    print(f"Results saved to: {config['results_dir']}")


if __name__ == "__main__":
    main()
