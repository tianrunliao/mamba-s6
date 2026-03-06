import os
import ast
import yaml
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import skew, kurtosis
from typing import Dict, List
from .cross_section import compute_weights_and_features, save_weight_results

_SAFE_NAMES = {
    'np': np, 'pd': pd, 'scipy': scipy,
    'abs': abs, 'len': len, 'max': max, 'min': min,
    'sum': sum, 'float': float, 'int': int, 'round': round,
    'True': True, 'False': False, 'None': None,
}

_BLOCKED_NODES = (ast.Import, ast.ImportFrom)


def _validate_ast(node):
    for child in ast.walk(node):
        if isinstance(child, tuple(_BLOCKED_NODES)):
            raise ValueError(f"Disallowed AST node: {type(child).__name__}")
        if isinstance(child, ast.Attribute) and child.attr.startswith('__'):
            raise ValueError(f"Dunder attribute access disallowed: {child.attr}")
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name) and func.id in (
                'exec', 'eval', 'compile', '__import__',
                'globals', 'locals', 'getattr', 'setattr', 'delattr',
                'type', 'vars', 'dir', 'open',
            ):
                raise ValueError(f"Disallowed function call: {func.id}")


def safe_execute(code_str, local_namespace):
    try:
        tree = ast.parse(code_str, mode='exec')
    except SyntaxError as e:
        raise ValueError(f"Syntax error in factor code: {e}") from e
    _validate_ast(tree)
    restricted_globals = {'__builtins__': _SAFE_NAMES}
    exec(compile(tree, '<factor>', 'exec'), restricted_globals, local_namespace)
    return local_namespace.get('result', 0.0)


class ConfigurableFactorEngine:

    def __init__(self, config_path='data/factor_config.yaml', data_dir='data_integrated'):
        self.config_path = config_path
        self.data_dir = data_dir
        self.factors_dir = os.path.join(data_dir, 'factors')
        if not os.path.exists(self.factors_dir):
            os.makedirs(self.factors_dir)
        self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Factor config not found: {self.config_path}")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        for i, factor in enumerate(self.config.get('factors', [])):
            for key in ('name', 'calculation', 'direction'):
                if key not in factor:
                    raise ValueError(f"Factor {i} missing '{key}'")
        self.factors = self.config.get('factors', [])
        self.processing_params = self.config.get('processing', {})

    def _calculate_factor(self, factor_def, group):
        try:
            local_ns = {
                'group': group, 'ret': group['ret'],
                'close': group['close'], 'open': group['open'],
                'volume': group['volume'], 'amount': group['amount'],
                'result': 0.0,
            }
            code = factor_def['calculation'].strip()
            if '\n' not in code and not code.startswith('result'):
                code = f'result = {code}'
            result = safe_execute(code, local_ns)
            return float(result) if not pd.isna(result) else 0.0
        except ValueError:
            return 0.0
        except Exception:
            return 0.0

    def calc_factors(self, df):
        df = df.copy()
        df['ret'] = df['close'].pct_change()
        df['amount'] = df['close'] * df['volume']

        results = []
        grouped = df.groupby(pd.Grouper(freq='D'))
        min_pts = self.processing_params.get('min_data_points', 30)

        for date, group in grouped:
            if len(group) < 60:
                continue
            g = group.dropna().copy()
            if len(g) < min_pts:
                continue
            row = {'date': date}
            for factor_def in self.factors:
                name = f"factor_{factor_def['name']}"
                row[name] = self._calculate_factor(factor_def, g)
            results.append(row)

        return pd.DataFrame(results).set_index('date')

    def get_factor_directions(self):
        return {f"factor_{f['name']}": f['direction'] for f in self.factors}

    def run(self, symbols, data_dir='data_integrated'):
        all_dfs = []
        prices_dict = {}
        futures_dir = os.path.join(data_dir, 'futures')
        factors_dir = os.path.join(data_dir, 'factors')

        for sym in symbols:
            price_path = os.path.join(futures_dir, f"{sym}.csv")
            if os.path.exists(price_path):
                price_df = pd.read_csv(price_path, parse_dates=['timestamp'], index_col='timestamp')
                prices_dict[sym] = price_df['close'].resample('D').last()

            factor_path = os.path.join(futures_dir, f"{sym}.csv")
            if not os.path.exists(factor_path):
                continue

            df = pd.read_csv(factor_path, parse_dates=['timestamp'], index_col='timestamp')
            factors = self.calc_factors(df)
            if not factors.empty:
                factors['symbol'] = sym
                all_dfs.append(factors.reset_index())

        if not all_dfs:
            return

        combined_factors = pd.concat(all_dfs, ignore_index=True)
        combined_factors.to_csv(os.path.join(factors_dir, 'all_factors.csv'), index=False)

        dir_map = self.get_factor_directions()
        daily_results = compute_weights_and_features(
            combined_factors, prices_dict, symbols, dir_map,
        )
        save_weight_results(daily_results, factors_dir)


def process_factors_configurable(config):
    symbols = config.get('symbols', [])
    if not symbols:
        raise ValueError("No symbols specified in config")
    engine = ConfigurableFactorEngine(
        config_path=config.get('factor_config_path', 'data/factor_config.yaml'),
        data_dir=config.get('data_dir', 'data_integrated'),
    )
    engine.run(symbols=symbols, data_dir=config.get('data_dir', 'data_integrated'))