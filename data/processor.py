import os
import logging
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from .downloader import BinanceDownloader
from .configurable_factor_engine import process_factors_configurable
from .cross_section import compute_weights_and_features, save_weight_results

logger = logging.getLogger(__name__)

DIRECTIONS = {
    'factor_1': 1,
    'factor_2': 1,
    'factor_3': -1,
    'factor_4': 1,
    'factor_5': 1,
    'factor_6': 1,
    'factor_7': -1,
    'factor_8': 1,
    'factor_9': -1,
    'factor_10': -1,
}


class FactorEngine:

    def __init__(self, data_dir='data_integrated'):
        self.data_dir = data_dir
        self.factors_dir = os.path.join(data_dir, 'factors')
        if not os.path.exists(self.factors_dir):
            os.makedirs(self.factors_dir)

    def calc_factors(self, df):
        df = df.copy()
        df['ret'] = df['close'].pct_change()
        df['amount'] = df['close'] * df['volume']

        results = []
        grouped = df.groupby(pd.Grouper(freq='D'))

        for date, group in grouped:
            if len(group) < 60:
                continue
            g = group.dropna().copy()
            if len(g) < 30:
                continue

            row = {'date': date}

            try:
                s_t = g['ret'].abs() / np.sqrt(g['volume'])
                s_t = s_t.replace([np.inf, -np.inf], 0).fillna(0)
                sorted_idx = s_t.sort_values(ascending=False).index
                sorted_vol = g.loc[sorted_idx, 'volume']
                cum_vol = sorted_vol.cumsum()
                smart_mask = cum_vol <= (sorted_vol.sum() * 0.2)
                smart_idx = smart_mask[smart_mask].index
                if len(smart_idx) > 0:
                    vwap_smart = g.loc[smart_idx, 'amount'].sum() / g.loc[smart_idx, 'volume'].sum()
                    vwap_all = g['amount'].sum() / g['volume'].sum()
                    row['factor_1'] = vwap_smart / vwap_all if vwap_all > 0 else 1.0
                else:
                    row['factor_1'] = 1.0
            except Exception:
                logger.debug("factor_1 calc failed for %s", date)
                row['factor_1'] = 1.0

            try:
                hist, _ = np.histogram(g['volume'], bins=10)
                prob = hist / hist.sum()
                prob = prob[prob > 0]
                row['factor_2'] = -np.sum(prob * np.log(prob))
            except Exception:
                row['factor_2'] = 0.0

            row['factor_3'] = kurtosis(g['ret'])

            try:
                thresh = g['volume'].quantile(0.75)
                large = g[g['volume'] > thresh]
                if len(large) > 0:
                    up = large[large['ret'] > 0]['volume'].sum()
                    down = large[large['ret'] < 0]['volume'].sum()
                    total = up + down
                    row['factor_4'] = (up - down) / total if total > 0 else 0.0
                else:
                    row['factor_4'] = 0.0
            except Exception:
                row['factor_4'] = 0.0

            try:
                mid = len(g) // 2
                if mid > 0:
                    row['factor_5'] = g['close'].iloc[mid] / g['close'].iloc[0] - 1
                else:
                    row['factor_5'] = 0.0
            except Exception:
                row['factor_5'] = 0.0

            row['factor_6'] = g['close'].corr(g['volume'])
            row['factor_7'] = g['volume'].std() / (g['volume'].mean() + 1e-8)

            if len(g) > 60:
                row['factor_8'] = g['close'].iloc[-30:].mean() / g['close'].iloc[:30].mean() - 1
            else:
                row['factor_8'] = 0.0

            try:
                r_vol = g['ret'].rolling(15).std().dropna()
                row['factor_9'] = skew(r_vol)
            except Exception:
                row['factor_9'] = 0.0

            try:
                thresh = g['volume'].quantile(0.9)
                big_idx = g[g['volume'] > thresh].index
                follows = []
                for idx in big_idx:
                    loc = g.index.get_loc(idx)
                    if loc + 5 < len(g):
                        big_v = g['volume'].iloc[loc]
                        follow_v = g['volume'].iloc[loc + 1:loc + 6].sum()
                        if big_v > 0:
                            follows.append(follow_v / big_v)
                row['factor_10'] = np.mean(follows) if follows else 0.0
            except Exception:
                row['factor_10'] = 0.0

            results.append(row)

        return pd.DataFrame(results).set_index('date')

    def run(self, symbols, data_dir='data_integrated'):
        logger.info("Calculating factors and cross-sectional weights")

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
            logger.warning("No factors calculated")
            return

        combined_factors = pd.concat(all_dfs, ignore_index=True)
        combined_factors.to_csv(os.path.join(factors_dir, 'all_factors.csv'), index=False)

        daily_results = compute_weights_and_features(
            combined_factors, prices_dict, symbols, DIRECTIONS,
        )
        save_weight_results(daily_results, factors_dir)


def process_factors(config):
    symbols = config.get('symbols', [])
    if not symbols:
        raise ValueError("No symbols specified in config")

    if config.get('use_configurable_factors', False):
        return process_factors_configurable(config)

    engine = FactorEngine(data_dir=config.get('data_dir', 'data_integrated'))
    engine.run(symbols=symbols, data_dir=config.get('data_dir', 'data_integrated'))
