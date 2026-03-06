# Mamba Crypto Strategy

Cryptocurrency multi-factor strategy with Mamba S6 dynamic timing.

## Structure

```
mamba_crypto_strategy/
├── backtest_runner.py
├── config.yaml
├── data/
│   ├── downloader.py
│   ├── processor.py
│   ├── cross_section.py
│   ├── configurable_factor_engine.py
│   ├── factor_config.yaml
│   └── example_custom_factors.yaml
├── models/
│   ├── mamba_s6.py
│   ├── timing_network.py
│   └── training.py
├── strategies/
│   ├── base_strategy.py
│   ├── entropy_strategy.py
│   └── mamba_strategy.py
├── backtesting/
│   ├── engine.py
│   └── metrics.py
└── visualization/
    ├── plots.py
    └── report_generator.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Download Data

```python
from data.downloader import download_binance_data

download_binance_data({
    'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', ...],
    'start_date': '2021-01-01',
    'data_dir': 'data_integrated',
})
```

### Calculate Factors

```python
from data.processor import process_factors

process_factors({
    'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', ...],
    'data_dir': 'data_integrated',
})
```

### Run Backtest

```bash
python backtest_runner.py --config config.yaml
```

Override via CLI:

```bash
python backtest_runner.py --data-dir path/to/data --results-dir path/to/results
```

### Custom Factors (YAML)

See `data/example_custom_factors.yaml` for examples. Enable with:

```python
process_factors({
    'symbols': [...],
    'data_dir': 'data_integrated',
    'use_configurable_factors': True,
    'factor_config_path': 'my_factors.yaml',
})
```

## License

Research and educational use only.