"""Data download, factor generation, and portfolio feature construction."""

import argparse
import glob
import json
import os
import time
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.stats._warnings_errors import ConstantInputWarning
import warnings
import yaml


SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data_integrated')
SPOT_DIR = os.path.join(DATA_DIR, 'spot')
FUTURES_DIR = os.path.join(DATA_DIR, 'futures')
FACTORS_DIR = os.path.join(DATA_DIR, 'factors')
CONFIG_DIR = os.path.join(SCRIPT_DIR, 'config', 'factor_sets')
DOWNLOAD_PROGRESS_PATH = os.path.join(DATA_DIR, 'download_progress.json')

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT',
    'DOGEUSDT', 'TRXUSDT', 'DOTUSDT', 'LTCUSDT', 'LINKUSDT', 'AVAXUSDT'
]


def ensure_directories():
    for directory in [DATA_DIR, SPOT_DIR, FUTURES_DIR, FACTORS_DIR]:
        os.makedirs(directory, exist_ok=True)


def configure_data_paths(data_dir=None, spot_dir=None, futures_dir=None, factors_dir=None):
    """Point data IO at caller-provided market-data and artifact directories."""
    global DATA_DIR, SPOT_DIR, FUTURES_DIR, FACTORS_DIR, DOWNLOAD_PROGRESS_PATH
    if not data_dir:
        return
    DATA_DIR = os.path.abspath(data_dir)
    SPOT_DIR = os.path.abspath(spot_dir) if spot_dir else os.path.join(DATA_DIR, 'spot')
    FUTURES_DIR = os.path.abspath(futures_dir) if futures_dir else os.path.join(DATA_DIR, 'futures')
    FACTORS_DIR = os.path.abspath(factors_dir) if factors_dir else os.path.join(DATA_DIR, 'factors')
    DOWNLOAD_PROGRESS_PATH = os.path.join(DATA_DIR, 'download_progress.json')
    ensure_directories()


class BinanceDownloader:
    BASE_URL_SPOT = 'https://api.binance.com/api/v3/klines'
    BASE_URL_FUTURES = 'https://fapi.binance.com/fapi/v1/klines'

    def __init__(self, flush_interval_batches: int = 5, progress_save_interval: float = 5.0):
        ensure_directories()
        self.flush_interval_batches = max(1, int(flush_interval_batches))
        self.progress_save_interval = max(1.0, float(progress_save_interval))
        self.progress_state = self._load_progress_state()
        self._last_progress_save = 0.0

    def _load_progress_state(self):
        if not os.path.exists(DOWNLOAD_PROGRESS_PATH):
            return {}
        try:
            with open(DOWNLOAD_PROGRESS_PATH, 'r', encoding='utf-8') as file_obj:
                payload = json.load(file_obj)
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            print(f'  Warning: failed to read download progress file: {exc}')
        return {}

    def _save_progress_state(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_progress_save) < self.progress_save_interval:
            return
        temp_path = DOWNLOAD_PROGRESS_PATH + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as file_obj:
            json.dump(self.progress_state, file_obj, ensure_ascii=False, indent=2)
        os.replace(temp_path, DOWNLOAD_PROGRESS_PATH)
        self._last_progress_save = now

    def _progress_key(self, symbol, data_type):
        return f'{data_type}:{symbol}'

    def _inspect_existing_file(self, filepath):
        if not os.path.exists(filepath):
            return None
        try:
            current = pd.read_csv(filepath, usecols=['timestamp'])
            if current.empty:
                return None
            current['timestamp'] = pd.to_datetime(current['timestamp'])
            last_dt = current['timestamp'].iloc[-1]
            return {
                'rows': int(len(current)),
                'last_dt': last_dt,
                'last_ts': int(last_dt.timestamp() * 1000),
            }
        except Exception as exc:
            print(f'    Warning: failed to inspect existing file {filepath}: {exc}')
            return None

    def _describe_resume_point(self, symbol, data_type, existing_info, start_ts):
        if existing_info is None:
            print(f'  {symbol} {data_type}: no local file, starting from {datetime.fromtimestamp(start_ts / 1000):%Y-%m-%d %H:%M}')
            return
        print(
            f"  {symbol} {data_type}: detected existing file with {existing_info['rows']} rows, "
            f"resuming from {existing_info['last_dt']:%Y-%m-%d %H:%M}"
        )

    def _update_progress(self, symbol, data_type, filepath, start_ts, end_ts, current_ts, rows_saved, status):
        key = self._progress_key(symbol, data_type)
        self.progress_state[key] = {
            'symbol': symbol,
            'data_type': data_type,
            'file_path': filepath,
            'start_ts': int(start_ts),
            'end_ts': int(end_ts),
            'last_ts': int(current_ts),
            'rows_saved': int(rows_saved),
            'updated_at': datetime.now().isoformat(timespec='seconds'),
            'status': status,
        }
        self._save_progress_state(force=False)

    def _print_chunk_progress(self, symbol, data_type, start_ts, current_ts, end_ts, rows_saved, batch_rows):
        covered = max(0, min(current_ts, end_ts) - start_ts)
        total = max(end_ts - start_ts, 1)
        progress = covered / total
        print(
            f'    {symbol} {data_type}: +{batch_rows} rows saved | '
            f'through {datetime.fromtimestamp(min(current_ts, end_ts) / 1000):%Y-%m-%d %H:%M} | '
            f'total_rows={rows_saved} | progress={progress:.2%}'
        )

    def fetch_data(self, symbol, start_ts, end_ts, data_type='futures'):
        url_base = self.BASE_URL_FUTURES if data_type == 'futures' else self.BASE_URL_SPOT
        current_start = start_ts
        interval = '1m'
        limit = 1000

        while current_start < end_ts:
            params = f'?symbol={symbol}&interval={interval}&startTime={current_start}&endTime={end_ts}&limit={limit}'
            url = url_base + params
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    if response.status != 200:
                        time.sleep(1)
                        continue
                    payload = json.loads(response.read().decode())
            except Exception as exc:
                print(f'    Error downloading {symbol} {data_type}: {exc}')
                time.sleep(2)
                continue

            if not payload:
                break

            current_start = payload[-1][0] + 60000
            yield payload
            time.sleep(0.05)

    def process_and_save(self, raw_data, filepath, mode='w'):
        if not raw_data:
            return 0, None
        df = pd.DataFrame(raw_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        for column in ['open', 'high', 'low', 'close', 'volume']:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        final_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].set_index('timestamp')
        final_df.to_csv(filepath, mode=mode, header=(mode == 'w'))
        return int(len(final_df)), final_df.index[-1]

    def _download_symbol(self, symbol, data_type, base_dir, start_ts, end_ts):
        path = os.path.join(base_dir, f'{symbol}.csv')
        existing_info = self._inspect_existing_file(path)
        self._describe_resume_point(symbol, data_type, existing_info, start_ts)

        current_start_ts = start_ts
        rows_saved = 0
        write_mode = 'w'
        if existing_info is not None:
            current_start_ts = existing_info['last_ts'] + 60000
            rows_saved = existing_info['rows']
            write_mode = 'a'

        if current_start_ts >= end_ts - 60000:
            print(f'  {symbol} {data_type}: already up to date.')
            self._update_progress(symbol, data_type, path, start_ts, end_ts, end_ts, rows_saved, 'completed')
            return

        pending_batches = []
        last_saved_ts = current_start_ts
        for batch_idx, payload in enumerate(self.fetch_data(symbol, current_start_ts, end_ts, data_type), start=1):
            pending_batches.extend(payload)
            last_saved_ts = payload[-1][0]
            should_flush = batch_idx % self.flush_interval_batches == 0
            if should_flush:
                saved_rows, saved_dt = self.process_and_save(pending_batches, path, mode=write_mode)
                if saved_rows > 0:
                    rows_saved += saved_rows
                    write_mode = 'a'
                    pending_batches = []
                    saved_ts = int(saved_dt.timestamp() * 1000)
                    self._print_chunk_progress(symbol, data_type, start_ts, saved_ts, end_ts, rows_saved, saved_rows)
                    self._update_progress(symbol, data_type, path, start_ts, end_ts, saved_ts, rows_saved, 'downloading')

        if pending_batches:
            saved_rows, saved_dt = self.process_and_save(pending_batches, path, mode=write_mode)
            if saved_rows > 0:
                rows_saved += saved_rows
                saved_ts = int(saved_dt.timestamp() * 1000)
                self._print_chunk_progress(symbol, data_type, start_ts, saved_ts, end_ts, rows_saved, saved_rows)
                self._update_progress(symbol, data_type, path, start_ts, end_ts, saved_ts, rows_saved, 'downloading')
                last_saved_ts = saved_ts

        final_ts = max(last_saved_ts, current_start_ts)
        self._update_progress(symbol, data_type, path, start_ts, end_ts, final_ts, rows_saved, 'completed')
        print(f'  {symbol} {data_type}: completed, total_rows={rows_saved}')

    def run(self, start_date='2021-01-01', end_date=None):
        start_dt = pd.Timestamp(start_date).to_pydatetime()
        if end_date is None:
            end_dt = datetime.now()
        else:
            end_dt = min(pd.Timestamp(end_date).to_pydatetime(), datetime.now())
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        print('\n=== [1/2] Downloading Data ===')
        for symbol in SYMBOLS:
            for data_type, base_dir in [('spot', SPOT_DIR), ('futures', FUTURES_DIR)]:
                self._download_symbol(symbol, data_type, base_dir, start_ts, end_ts)
        self._save_progress_state(force=True)


class FactorEngine:
    """Build cross-sectional signals from historical intraday factors."""

    ALPHA_PRIORS = {
        'f_price_trend',
        'f_intra_mom',
        'f_return_acceleration',
        'f_smart_money',
        'f_large_order_imbalance',
        'f_volume_synergy',
        'f_pv_corr',
    }

    DEFAULT_DIRECTIONS = {
        'f_smart_money': 1.0,
        'f_volume_entropy': 1.0,
        'f_kurt': -1.0,
        'f_large_order_imbalance': 1.0,
        'f_volume_synergy': 1.0,
        'f_max_dd': -1.0,
        'f_overnight_gap': 1.0,
        'f_head_95': -1.0,
        'f_pv_corr': 1.0,
        'f_vol_cv': -1.0,
        'f_range_pos': 1.0,
        'f_return_acceleration': 1.0,
        'f_morning_ret_reversal': -1.0,
        'f_herding_follow': -1.0,
        'f_intra_mom': 1.0,
        'f_vol_trend': -1.0,
        'f_price_trend': 1.0,
        'f_daily_vol': -1.0,
        'f_vol_roc_skew': -1.0,
    }

    IMPLEMENTED_IMPORTED_FACTORS = {
        'f_imported_fuzzy_amount_ratio',
        'f_imported_smart_money_factor_original',
        'f_imported_volume_peak_count',
        'f_imported_real_var_positive',
        'f_imported_uret_prime',
        'f_imported_smart_money_factor_rank',
        'f_imported_fuzzy_price_spread',
        'f_imported_smart_money_factor_volume',
        'f_imported_volume_synergy',
        'f_imported_real_var',
        'f_imported_amihud_illiq',
        'f_imported_vol_entropy',
    }

    def __init__(self, factor_config_path=None):
        ensure_directories()
        warnings.filterwarnings('ignore', category=ConstantInputWarning)
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        self.factor_config_path = factor_config_path or os.path.join(CONFIG_DIR, 'crypto_core.yaml')
        self.factor_config = self._load_factor_config()

    def _load_factor_config(self):
        config_path = self.factor_config_path
        if not os.path.exists(config_path):
            return {}
        with open(config_path, 'r', encoding='utf-8') as file_obj:
            return yaml.safe_load(file_obj) or {}

    def _get_selection_config(self):
        selection = self.factor_config.get('selection', {})
        return {
            'ic_lookback': int(selection.get('ic_lookback', 120)),
            'min_ic_observations': int(selection.get('min_ic_observations', 40)),
            'top_n': int(selection.get('top_n', 2)),
            'candidate_pool_size': int(selection.get('candidate_pool_size', max(4, selection.get('top_n', 2) * 2))),
            'strong_bucket_size': int(selection.get('strong_bucket_size', selection.get('top_n', 2))),
            'max_weight': float(selection.get('max_weight', 0.18)),
            'max_active_factors': int(selection.get('max_active_factors', 6)),
            'require_positive_icir': bool(selection.get('require_positive_icir', False)),
            'hard_disable_non_positive_ic': bool(selection.get('hard_disable_non_positive_ic', False)),
            'max_factors_per_group': int(selection.get('max_factors_per_group', 2)),
            'prior_strength': float(selection.get('prior_strength', 0.35)),
            'min_factor_score': float(selection.get('min_factor_score', 0.0)),
            'min_names_per_side': int(selection.get('min_names_per_side', 2)),
            'score_power': float(selection.get('score_power', 1.25)),
            'max_crowding_hhi': float(selection.get('max_crowding_hhi', 0.18)),
        }

    def _get_enabled_factor_metadata(self):
        factor_meta = {}
        for row in self.factor_config.get('native_factors', []):
            name = row.get('name')
            if not name:
                continue
            factor_meta[name] = {
                'enabled': bool(row.get('enabled', True)),
                'direction': float(row.get('direction', self.DEFAULT_DIRECTIONS.get(name, 1.0))),
                'group_tag': row.get('group_tag', 'misc'),
                'source': row.get('source', 'native'),
            }
        imported_cfg = self.factor_config.get('imported_factors', {})
        imported_path = imported_cfg.get('config_path')
        if imported_path and not os.path.isabs(imported_path):
            imported_path = os.path.join(os.path.dirname(self.factor_config_path), imported_path)
        if imported_cfg.get('integrate_when_implemented', False) and imported_path and os.path.exists(imported_path):
            with open(imported_path, 'r', encoding='utf-8') as file_obj:
                imported_data = yaml.safe_load(file_obj) or {}
            for row in imported_data.get('factors', []):
                raw_name = str(row.get('signal_name') or row.get('signal_name_raw') or '').strip()
                if not raw_name:
                    continue
                factor_name = f"f_imported_{raw_name.lower()}"
                factor_meta.setdefault(factor_name, {
                    'enabled': True,
                    'direction': float(row.get('direction', 1.0)),
                    'group_tag': row.get('group_tag', 'imported'),
                    'source': 'imported_ashare',
                })
        return factor_meta

    def _select_factor_universe(self, combined_factors: pd.DataFrame, factor_meta: dict):
        available_factor_cols = [col for col in combined_factors.columns if col.startswith('f_')]
        configured_factors = [name for name, meta in factor_meta.items() if meta.get('enabled', False)]
        factor_cols = [col for col in configured_factors if col in available_factor_cols]
        if factor_cols:
            return factor_cols
        return available_factor_cols

    def _cap_group_members(self, strength: pd.Series, factor_meta: dict, max_per_group: int):
        if max_per_group <= 0 or strength.empty:
            return strength

        kept = []
        group_counts = {}
        ordered = strength.sort_values(ascending=False)
        for factor in ordered.index:
            group_tag = factor_meta.get(factor, {}).get('group_tag', 'misc')
            count = group_counts.get(group_tag, 0)
            if count >= max_per_group:
                continue
            kept.append(factor)
            group_counts[group_tag] = count + 1

        capped = strength.copy()
        capped.loc[~capped.index.isin(kept)] = 0.0
        return capped

    def _winsorize(self, series, stds=3.0):
        mean_val = series.mean()
        std_val = series.std()
        if pd.isna(std_val) or std_val <= 1e-8:
            return series.fillna(mean_val)
        return series.clip(mean_val - stds * std_val, mean_val + stds * std_val)

    def _standardize(self, series):
        std_val = series.std()
        if pd.isna(std_val) or std_val <= 1e-8:
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std_val

    def _load_daily_bars(self):
        close_dict = {}
        open_dict = {}
        for file_path in glob.glob(os.path.join(FUTURES_DIR, '*.csv')):
            symbol = os.path.basename(file_path).replace('.csv', '')
            df = pd.read_csv(file_path, parse_dates=['timestamp']).set_index('timestamp').sort_index()
            daily = df.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
            }).dropna(how='all').ffill()
            close_dict[symbol] = daily['close']
            open_dict[symbol] = daily['open']
        close_df = pd.DataFrame(close_dict).sort_index()
        open_df = pd.DataFrame(open_dict).sort_index()
        return close_df, open_df

    def _calc_absorption_ratio(self, close_df: pd.DataFrame, symbols: list[str], date, window=20):
        available = close_df.reindex(columns=symbols).loc[:date].tail(window)
        available = available.dropna(axis=1, how='any')
        if available.shape[1] < 3 or available.shape[0] < window:
            return 0.5
        returns = available.pct_change().dropna()
        if len(returns) < window * 0.8:
            return 0.5
        cov = returns.cov()
        eigenvals = np.linalg.eigvalsh(cov)
        if eigenvals.sum() <= 1e-8:
            return 0.5
        return float(eigenvals[-1] / eigenvals.sum())

    def calc_factors(self, df):
        """Compute daily factors from minute bars."""
        local = df.copy()
        local['ret'] = local['close'].pct_change()
        local['amount'] = local['close'] * local['volume']
        results = []

        for date, group in local.groupby(pd.Grouper(freq='D')):
            if len(group) < 60:
                continue
            g = group.dropna().copy()
            if len(g) < 30:
                continue

            row = {'date': date}
            volume = g['volume'].replace(0.0, np.nan)
            abs_ret = g['ret'].abs().fillna(0.0)
            vwap_all = g['amount'].sum() / max(g['volume'].sum(), 1e-8)
            s_t = (abs_ret / np.sqrt(volume)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            order = s_t.sort_values(ascending=False).index
            sorted_vol = g.loc[order, 'volume']
            smart_cutoff = sorted_vol.cumsum() <= sorted_vol.sum() * 0.2
            smart_idx = smart_cutoff[smart_cutoff].index
            if len(smart_idx) > 0 and g.loc[smart_idx, 'volume'].sum() > 0:
                vwap_smart = g.loc[smart_idx, 'amount'].sum() / g.loc[smart_idx, 'volume'].sum()
                row['f_smart_money'] = vwap_smart / max(vwap_all, 1e-8)
                row['f_imported_smart_money_factor_original'] = row['f_smart_money']
            else:
                row['f_smart_money'] = 1.0
                row['f_imported_smart_money_factor_original'] = 1.0

            hist, _ = np.histogram(g['volume'], bins=10)
            prob = hist / max(hist.sum(), 1)
            prob = prob[prob > 0]
            row['f_volume_entropy'] = float(-np.sum(prob * np.log(prob)))
            row['_daily_vol_entropy'] = row['f_volume_entropy']
            row['f_kurt'] = float(kurtosis(g['ret'].dropna())) if g['ret'].notna().sum() > 5 else 0.0

            large_threshold = g['volume'].quantile(0.75)
            large = g[g['volume'] > large_threshold]
            up = large.loc[large['ret'] > 0, 'volume'].sum()
            down = large.loc[large['ret'] < 0, 'volume'].sum()
            row['f_large_order_imbalance'] = float((up - down) / max(up + down, 1e-8))

            row['f_volume_synergy'] = float(g['ret'].corr(g['volume'].pct_change())) if g['volume'].pct_change().notna().sum() > 5 else 0.0
            row['f_imported_volume_synergy'] = row['f_volume_synergy']
            intraday_curve = (1.0 + g['ret'].fillna(0.0)).cumprod()
            peak = intraday_curve.cummax()
            row['f_max_dd'] = float((intraday_curve / np.maximum(peak, 1e-8) - 1.0).min())

            open_price = g['open'].iloc[0]
            prev_close_proxy = g['close'].iloc[0]
            row['f_overnight_gap'] = float((open_price - prev_close_proxy) / max(prev_close_proxy, 1e-8))
            row['f_head_95'] = float(np.quantile(abs_ret, 0.95))
            row['f_pv_corr'] = float(g['close'].corr(g['volume'])) if g['volume'].nunique() > 1 else 0.0
            row['f_vol_cv'] = float(g['volume'].std() / max(g['volume'].mean(), 1e-8))

            daily_high = g['high'].max()
            daily_low = g['low'].min()
            row['f_range_pos'] = float((g['close'].iloc[-1] - daily_low) / max(daily_high - daily_low, 1e-8))
            row['f_return_acceleration'] = float(g['ret'].diff().mean())
            morning = g.iloc[: max(5, len(g) // 3)]
            afternoon = g.iloc[-max(5, len(g) // 3):]
            row['f_morning_ret_reversal'] = float(-morning['ret'].mean() * afternoon['ret'].mean())

            threshold_90 = g['volume'].quantile(0.9)
            big_idx = g[g['volume'] > threshold_90].index
            follow_ratios = []
            for idx in big_idx:
                loc = g.index.get_loc(idx)
                if loc + 5 < len(g):
                    follow_vol = g['volume'].iloc[loc + 1:loc + 6].sum()
                    base_vol = g['volume'].iloc[loc]
                    follow_ratios.append(follow_vol / max(base_vol, 1e-8))
            row['f_herding_follow'] = float(np.mean(follow_ratios)) if follow_ratios else 0.0

            row['f_intra_mom'] = float(g['close'].iloc[-1] / max(g['close'].iloc[0], 1e-8) - 1.0)
            rolling_vol = g['ret'].rolling(30).std().dropna()
            row['f_vol_trend'] = float(rolling_vol.iloc[-1] - rolling_vol.iloc[0]) if len(rolling_vol) > 1 else 0.0
            row['f_price_trend'] = float(np.polyfit(np.arange(len(g)), g['close'].values, 1)[0] / max(g['close'].mean(), 1e-8))
            row['f_daily_vol'] = float(g['ret'].std())
            row['f_vol_roc_skew'] = float(skew(rolling_vol)) if len(rolling_vol) > 5 else 0.0

            minute_rank_abs_ret = abs_ret.rank(pct=True, method='average')
            minute_rank_vol = g['volume'].rank(pct=True, method='average')
            smart_rank_score = minute_rank_abs_ret + minute_rank_vol
            smart_rank_order = smart_rank_score.sort_values(ascending=False).index
            smart_rank_sorted_vol = g.loc[smart_rank_order, 'volume']
            smart_rank_cutoff = smart_rank_sorted_vol.cumsum() <= smart_rank_sorted_vol.sum() * 0.2
            smart_rank_idx = smart_rank_cutoff[smart_rank_cutoff].index
            if len(smart_rank_idx) > 0 and g.loc[smart_rank_idx, 'volume'].sum() > 0:
                vwap_smart_rank = g.loc[smart_rank_idx, 'amount'].sum() / g.loc[smart_rank_idx, 'volume'].sum()
                row['f_imported_smart_money_factor_rank'] = float(vwap_smart_rank / max(vwap_all, 1e-8))
            else:
                row['f_imported_smart_money_factor_rank'] = 1.0

            vol_order = g['volume'].sort_values(ascending=False).index
            vol_sorted = g.loc[vol_order, 'volume']
            vol_cutoff = vol_sorted.cumsum() <= vol_sorted.sum() * 0.2
            vol_idx = vol_cutoff[vol_cutoff].index
            if len(vol_idx) > 0 and g.loc[vol_idx, 'volume'].sum() > 0:
                vwap_smart_vol = g.loc[vol_idx, 'amount'].sum() / g.loc[vol_idx, 'volume'].sum()
                row['f_imported_smart_money_factor_volume'] = float(vwap_smart_vol / max(vwap_all, 1e-8))
            else:
                row['f_imported_smart_money_factor_volume'] = 1.0

            fuzzy_score = s_t.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            fuzzy_mask = fuzzy_score > fuzzy_score.mean()
            fuzzy_amount_mean = float(g.loc[fuzzy_mask, 'amount'].mean()) if fuzzy_mask.any() else 0.0
            fuzzy_volume_mean = float(g.loc[fuzzy_mask, 'volume'].mean()) if fuzzy_mask.any() else 0.0
            total_amount_mean = float(g['amount'].mean())
            total_volume_mean = float(g['volume'].mean())
            row['f_imported_fuzzy_amount_ratio'] = fuzzy_amount_mean / max(total_amount_mean, 1e-8)
            fuzzy_volume_ratio = fuzzy_volume_mean / max(total_volume_mean, 1e-8)
            row['_daily_fuzzy_volume_ratio'] = fuzzy_volume_ratio
            row['f_imported_fuzzy_price_spread'] = row['f_imported_fuzzy_amount_ratio'] - fuzzy_volume_ratio

            row['f_imported_real_var'] = float(np.nanmean(np.square(g['ret'].fillna(0.0))))
            positive_rets = g.loc[g['ret'] > 0, 'ret']
            row['f_imported_real_var_positive'] = float(np.nanmean(np.square(positive_rets))) if len(positive_rets) > 0 else 0.0
            row['f_imported_amihud_illiq'] = float((abs_ret / g['amount'].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).mean())

            vol_threshold = g['volume'].mean() + g['volume'].std()
            vol_peaks = g['volume'] > vol_threshold
            peak_count = 0
            last_peak_loc = -10
            for idx in np.where(vol_peaks.values)[0]:
                if idx - last_peak_loc > 1:
                    peak_count += 1
                    last_peak_loc = idx
            row['_daily_volume_peak_count'] = float(peak_count)

            ret_weights = abs_ret / max(abs_ret.sum(), 1e-8)
            z_uniformity = float(-np.sum((ret_weights[ret_weights > 0]) * np.log(ret_weights[ret_weights > 0] + 1e-12)))
            row['_daily_info_uniformity'] = z_uniformity
            row['_daily_day_return'] = row['f_intra_mom']
            results.append(row)

        if not results:
            return pd.DataFrame()

        factor_df = pd.DataFrame(results).set_index('date').sort_index()
        factor_df['f_imported_volume_peak_count'] = factor_df['_daily_volume_peak_count'].rolling(20, min_periods=5).mean()
        factor_df['f_imported_vol_entropy'] = factor_df['_daily_vol_entropy'].rolling(20, min_periods=5).std()

        def _uret_prime(window_df):
            if len(window_df) < 8:
                return 0.0
            ordered = window_df.sort_values('_daily_info_uniformity')
            bucket = max(2, len(ordered) // 5)
            low_avg = ordered['_daily_day_return'].head(bucket).mean()
            high_avg = ordered['_daily_day_return'].tail(bucket).mean()
            return float(high_avg - low_avg)

        uret_values = []
        for idx in range(len(factor_df)):
            window = factor_df.iloc[max(0, idx - 19):idx + 1][['_daily_info_uniformity', '_daily_day_return']].dropna()
            uret_values.append(_uret_prime(window))
        factor_df['f_imported_uret_prime'] = uret_values

        factor_df = factor_df.drop(columns=[
            '_daily_vol_entropy',
            '_daily_fuzzy_volume_ratio',
            '_daily_volume_peak_count',
            '_daily_info_uniformity',
            '_daily_day_return',
        ], errors='ignore')
        return factor_df

    def _load_or_build_raw_factors(self):
        factor_out_path = os.path.join(FACTORS_DIR, 'all_factors.csv')
        if os.path.exists(factor_out_path):
            factors = pd.read_csv(factor_out_path, parse_dates=['date'])
            missing_cols = [col for col in self.IMPLEMENTED_IMPORTED_FACTORS if col not in factors.columns]
            if not missing_cols:
                print(f'  [OK] Using existing raw factors from {factor_out_path}')
                return factors
            print(f'  Existing raw factors missing {len(missing_cols)} implemented imported columns; rebuilding...')

        all_rows = []
        for symbol in SYMBOLS:
            file_path = os.path.join(FUTURES_DIR, f'{symbol}.csv')
            if not os.path.exists(file_path):
                continue
            print(f'  Calculating factors for {symbol}...')
            df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            fac = self.calc_factors(df)
            if fac.empty:
                continue
            fac['symbol'] = symbol
            all_rows.append(fac.reset_index())

        if not all_rows:
            raise FileNotFoundError('No factor data available. Download market data first.')

        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(factor_out_path, index=False)
        print(f'  [OK] Saved raw factors to {factor_out_path}')
        return combined

    def _rolling_factor_signals(
        self,
        factor_ic_history,
        factor_cols,
        factor_meta,
        date_idx,
        lookback,
        min_obs,
        max_active_factors,
        max_factors_per_group,
        require_positive_icir,
        hard_disable_non_positive_ic,
        prior_strength,
        min_factor_score,
    ):
        factor_scores = {}
        factor_signs = {}
        for factor in factor_cols:
            history = factor_ic_history.get(factor, [])
            start_idx = max(0, date_idx - lookback)
            ic_values = np.asarray([val for val in history[start_idx:date_idx] if pd.notna(val)], dtype=float)
            default_sign = self.DEFAULT_DIRECTIONS.get(factor, 1.0)
            if len(ic_values) >= min_obs:
                mean_ic = float(np.mean(ic_values))
                std_ic = float(np.std(ic_values))
                icir = mean_ic / (std_ic + 1e-4)
                if hard_disable_non_positive_ic and (mean_ic <= 0 or icir <= 0):
                    factor_scores[factor] = 0.0
                elif require_positive_icir and icir <= 0:
                    factor_scores[factor] = 0.0
                else:
                    factor_scores[factor] = max(abs(icir), 0.0)
                factor_signs[factor] = np.sign(mean_ic) if abs(mean_ic) > 1e-5 else default_sign
            else:
                factor_scores[factor] = prior_strength if factor in self.ALPHA_PRIORS else 0.0
                factor_signs[factor] = default_sign

        strength = pd.Series(factor_scores).clip(lower=min_factor_score)
        strength = self._cap_group_members(strength, factor_meta=factor_meta, max_per_group=max_factors_per_group)
        selected = strength.sort_values(ascending=False).head(max_active_factors).index
        strength.loc[~strength.index.isin(selected)] = 0.0
        if strength.sum() <= 1e-8:
            strength.loc[list(self.ALPHA_PRIORS & set(factor_cols))] = 1.0
        strength = strength / strength.sum()
        return strength, pd.Series(factor_signs)

    def _blend_toward_equal_weight(self, weights: pd.Series, max_crowding_hhi: float):
        if weights.abs().sum() <= 1e-8:
            return weights, 0.0, 0.0

        normalized_abs = weights.abs() / max(weights.abs().sum(), 1e-8)
        current_hhi = float((normalized_abs ** 2).sum())
        if current_hhi <= max_crowding_hhi:
            return weights, current_hhi, 0.0

        long_names = weights[weights > 0].index
        short_names = weights[weights < 0].index
        equal = pd.Series(0.0, index=weights.index)
        if len(long_names) > 0:
            equal.loc[long_names] = 0.5 / len(long_names)
        if len(short_names) > 0:
            equal.loc[short_names] = -0.5 / len(short_names)

        best = weights.copy()
        blend_used = 0.0
        for blend in np.linspace(0.05, 1.0, 20):
            candidate = (1.0 - blend) * weights + blend * equal
            candidate_abs = candidate.abs() / max(candidate.abs().sum(), 1e-8)
            candidate_hhi = float((candidate_abs ** 2).sum())
            best = candidate
            blend_used = float(blend)
            if candidate_hhi <= max_crowding_hhi:
                return candidate, candidate_hhi, blend_used
        best_abs = best.abs() / max(best.abs().sum(), 1e-8)
        return best, float((best_abs ** 2).sum()), blend_used

    def _compose_target_weights(
        self,
        scores: pd.Series,
        top_n: int,
        candidate_pool_size: int,
        strong_bucket_size: int,
        max_weight: float,
        min_names_per_side: int,
        score_power: float,
        max_crowding_hhi: float,
    ):
        scores = scores.sort_values()
        candidate_pool_size = max(min_names_per_side, min(candidate_pool_size, len(scores) // 2))
        strong_bucket_size = max(min_names_per_side, min(strong_bucket_size, candidate_pool_size, len(scores) // 2, top_n))

        candidate_long_names = scores.nlargest(candidate_pool_size).index
        candidate_short_names = scores.nsmallest(candidate_pool_size).index
        long_names = scores.loc[candidate_long_names].nlargest(strong_bucket_size).index
        short_names = scores.loc[candidate_short_names].nsmallest(strong_bucket_size).index

        long_strength = scores.loc[long_names].clip(lower=0.0).pow(score_power)
        short_strength = (-scores.loc[short_names]).clip(lower=0.0).pow(score_power)
        if long_strength.sum() <= 1e-8 or short_strength.sum() <= 1e-8:
            centered = scores - scores.mean()
            weights = centered / max(centered.abs().sum(), 1e-8)
            weights = weights.clip(lower=-max_weight, upper=max_weight)
            weights, crowding_hhi, crowding_blend = self._blend_toward_equal_weight(weights, max_crowding_hhi)
            return weights, {
                'candidate_pool_size': int(candidate_pool_size),
                'strong_bucket_size': int(strong_bucket_size),
                'long_count': int((weights > 0).sum()),
                'short_count': int((weights < 0).sum()),
                'crowding_hhi': crowding_hhi,
                'crowding_blend': crowding_blend,
            }

        weights = pd.Series(0.0, index=scores.index)
        weights.loc[long_names] = long_strength / long_strength.sum() * 0.5
        weights.loc[short_names] = -short_strength / short_strength.sum() * 0.5
        weights = weights.clip(lower=-max_weight, upper=max_weight)

        long_part = weights.clip(lower=0.0)
        short_part = (-weights.clip(upper=0.0))
        if long_part.sum() > 0:
            weights[weights > 0] *= 0.5 / long_part.sum()
        if short_part.sum() > 0:
            weights[weights < 0] *= 0.5 / short_part.sum()
        weights, crowding_hhi, crowding_blend = self._blend_toward_equal_weight(weights, max_crowding_hhi)
        return weights, {
            'candidate_pool_size': int(candidate_pool_size),
            'strong_bucket_size': int(strong_bucket_size),
            'long_count': int(len(long_names)),
            'short_count': int(len(short_names)),
            'crowding_hhi': crowding_hhi,
            'crowding_blend': crowding_blend,
        }

    def run(self):
        print('\n=== [2/2] Building Factors and Portfolio Inputs ===')
        combined_factors = self._load_or_build_raw_factors()
        combined_factors['date'] = pd.to_datetime(combined_factors['date'])
        combined_factors = combined_factors.sort_values(['date', 'symbol']).reset_index(drop=True)

        close_df, open_df = self._load_daily_bars()
        forward_returns = open_df.shift(-2).div(open_df.shift(-1)).sub(1.0)

        all_dates = sorted(d for d in combined_factors['date'].unique() if d >= pd.Timestamp('2021-01-01'))
        factor_meta = self._get_enabled_factor_metadata()
        factor_cols = self._select_factor_universe(combined_factors, factor_meta)
        factor_ic_history = {factor: [] for factor in factor_cols}
        daily_rows = []
        portfolio_returns = []
        turnover_history = []
        nav_history = [1.0]
        prev_weights = pd.Series(0.0, index=SYMBOLS)

        selection_cfg = self._get_selection_config()
        lookback = selection_cfg['ic_lookback']
        min_obs = selection_cfg['min_ic_observations']
        top_n = selection_cfg['top_n']
        candidate_pool_size = selection_cfg['candidate_pool_size']
        strong_bucket_size = selection_cfg['strong_bucket_size']
        max_weight = selection_cfg['max_weight']
        max_active_factors = selection_cfg['max_active_factors']
        max_factors_per_group = selection_cfg['max_factors_per_group']
        require_positive_icir = selection_cfg['require_positive_icir']
        hard_disable_non_positive_ic = selection_cfg['hard_disable_non_positive_ic']
        prior_strength = selection_cfg['prior_strength']
        min_factor_score = selection_cfg['min_factor_score']
        min_names_per_side = selection_cfg['min_names_per_side']
        score_power = selection_cfg['score_power']
        max_crowding_hhi = selection_cfg['max_crowding_hhi']
        factor_diag_rows = []

        for date_idx, date in enumerate(all_dates):
            day_frame = combined_factors[combined_factors['date'] == date].set_index('symbol')
            if len(day_frame) < 6:
                continue

            xs_df = day_frame[factor_cols].copy()
            for col in factor_cols:
                median_val = xs_df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                xs_df[col] = self._standardize(self._winsorize(xs_df[col].fillna(median_val)))
            realized = forward_returns.loc[date].reindex(xs_df.index) if date in forward_returns.index else pd.Series(index=xs_df.index, dtype=float)
            for factor in factor_cols:
                joined = pd.concat([xs_df[factor].rename('factor'), realized.rename('ret')], axis=1).dropna()
                if len(joined) >= 5 and joined['factor'].nunique() >= 2 and joined['ret'].nunique() >= 2:
                    factor_ic_history[factor].append(joined['factor'].corr(joined['ret'], method='spearman'))
                else:
                    factor_ic_history[factor].append(np.nan)

            factor_strength, factor_signs = self._rolling_factor_signals(
                factor_ic_history=factor_ic_history,
                factor_cols=factor_cols,
                factor_meta=factor_meta,
                date_idx=date_idx,
                lookback=lookback,
                min_obs=min_obs,
                max_active_factors=max_active_factors,
                max_factors_per_group=max_factors_per_group,
                require_positive_icir=require_positive_icir,
                hard_disable_non_positive_ic=hard_disable_non_positive_ic,
                prior_strength=prior_strength,
                min_factor_score=min_factor_score,
            )
            configured_signs = pd.Series({
                factor: factor_meta.get(factor, {}).get('direction', self.DEFAULT_DIRECTIONS.get(factor, 1.0))
                for factor in factor_cols
            })
            effective_signs = factor_signs.reindex(xs_df.columns).fillna(1.0) * configured_signs.reindex(xs_df.columns).fillna(1.0)
            composite_scores = (xs_df * effective_signs).mul(
                factor_strength.reindex(xs_df.columns).fillna(0.0),
                axis=1,
            ).sum(axis=1)
            target_weights, bucket_diag = self._compose_target_weights(
                composite_scores,
                top_n=top_n,
                candidate_pool_size=candidate_pool_size,
                strong_bucket_size=strong_bucket_size,
                max_weight=max_weight,
                min_names_per_side=min_names_per_side,
                score_power=score_power,
                max_crowding_hhi=max_crowding_hhi,
            )
            target_weights = target_weights.reindex(SYMBOLS).fillna(0.0)

            active_factors = factor_strength[factor_strength > 0].sort_values(ascending=False)
            group_summary = {}
            source_summary = {}
            for factor_name in active_factors.index:
                meta = factor_meta.get(factor_name, {})
                group_tag = meta.get('group_tag', 'misc')
                source_tag = meta.get('source', 'native')
                group_summary[group_tag] = group_summary.get(group_tag, 0) + 1
                source_summary[source_tag] = source_summary.get(source_tag, 0) + 1
            factor_diag_rows.append({
                'date': date,
                'active_factor_count': int(len(active_factors)),
                'signal_dispersion': float(composite_scores.std()),
                'signal_concentration': float(active_factors.iloc[0]) if len(active_factors) > 0 else 0.0,
                'factor_crowding_hhi': float(bucket_diag.get('crowding_hhi', 0.0)),
                'crowding_blend': float(bucket_diag.get('crowding_blend', 0.0)),
                'candidate_name_count': int(bucket_diag.get('candidate_pool_size', 0) * 2),
                'strong_name_count': int(bucket_diag.get('strong_bucket_size', 0) * 2),
                'long_bucket_count': int(bucket_diag.get('long_count', 0)),
                'short_bucket_count': int(bucket_diag.get('short_count', 0)),
                'native_factor_count': int(source_summary.get('native', 0)),
                'imported_factor_count': int(source_summary.get('imported_ashare', 0)),
                'trend_factor_count': int(group_summary.get('trend', 0)),
                'flow_factor_count': int(group_summary.get('flow', 0)),
                'volatility_factor_count': int(group_summary.get('volatility', 0)),
                'liquidity_factor_count': int(group_summary.get('liquidity', 0)),
                'behavior_factor_count': int(group_summary.get('behavior', 0)),
                'top_factor_1': active_factors.index[0] if len(active_factors) > 0 else '',
                'top_factor_2': active_factors.index[1] if len(active_factors) > 1 else '',
                'top_factor_3': active_factors.index[2] if len(active_factors) > 2 else '',
            })

            ar = self._calc_absorption_ratio(close_df, day_frame.index.tolist(), date)
            btc_momentum = 0.0
            if 'BTCUSDT' in close_df.columns and date in close_df.index:
                btc_hist = close_df.loc[:date, 'BTCUSDT'].dropna()
                if len(btc_hist) > 20:
                    btc_momentum = float(btc_hist.iloc[-1] / btc_hist.iloc[-21] - 1.0)

            recent_returns = portfolio_returns[-5:]
            recent_vol20 = portfolio_returns[-20:]
            recent_turnover = turnover_history[-20:]
            recent_nav60 = nav_history[-60:]
            if len(recent_nav60) >= 2:
                nav_arr = np.asarray(recent_nav60, dtype=float)
                drawdown_60d = float((nav_arr / np.maximum.accumulate(nav_arr) - 1.0).min())
            else:
                drawdown_60d = 0.0

            row = {
                'date': date,
                'factor_skewness': float(skew(composite_scores)) if len(composite_scores) > 3 else 0.0,
                'factor_kurtosis': float(kurtosis(composite_scores)) if len(composite_scores) > 3 else 0.0,
                'dispersion': float(composite_scores.std()),
                'absorption_ratio': ar,
                'portfolio_return_5d': float(np.mean(recent_returns)) if recent_returns else 0.0,
                'portfolio_volatility_20d': float(np.std(recent_vol20)) if len(recent_vol20) > 1 else 0.0,
                'portfolio_drawdown_60d': drawdown_60d,
                'btc_momentum_20d': float(np.clip(btc_momentum, -1.0, 1.0)),
                'n_positions_norm': float((target_weights.abs() > 1e-6).sum() / len(SYMBOLS)),
                'turnover_20d': float(np.mean(recent_turnover)) if recent_turnover else 0.0,
                'crowding_hhi': float(bucket_diag.get('crowding_hhi', 0.0)),
                'crowding_blend': float(bucket_diag.get('crowding_blend', 0.0)),
            }
            for symbol in SYMBOLS:
                row[f'weight_{symbol}'] = float(target_weights.get(symbol, 0.0))
            daily_rows.append(row)

            realized = forward_returns.loc[date].reindex(SYMBOLS).fillna(0.0) if date in forward_returns.index else pd.Series(0.0, index=SYMBOLS)
            portfolio_ret = float((target_weights * realized).sum())
            turnover = float((target_weights - prev_weights).abs().sum())
            portfolio_returns.append(portfolio_ret)
            turnover_history.append(turnover)
            nav_history.append(nav_history[-1] * (1.0 + portfolio_ret))
            prev_weights = target_weights

            if (date_idx + 1) % 250 == 0:
                print(f'  Processed {date_idx + 1}/{len(all_dates)} signal dates...')

        output_df = pd.DataFrame(daily_rows).sort_values('date')
        output_path = os.path.join(FACTORS_DIR, 'cross_sectional_weights_and_features.csv')
        output_df.to_csv(output_path, index=False)
        weights_only = output_df[['date'] + [f'weight_{symbol}' for symbol in SYMBOLS]]
        weights_only.to_csv(os.path.join(FACTORS_DIR, 'daily_weights.csv'), index=False)
        pd.DataFrame(factor_diag_rows).sort_values('date').to_csv(
            os.path.join(FACTORS_DIR, 'factor_selection_diagnostics.csv'),
            index=False,
        )
        print(f'  [OK] Saved portfolio inputs to {output_path} ({len(output_df)} rows)')


def main():
    parser = argparse.ArgumentParser(description='Mamba Data Processor')
    parser.add_argument('--download', action='store_true', help='Download data from Binance')
    parser.add_argument('--recalc', action='store_true', help='Rebuild factors and portfolio inputs')
    args = parser.parse_args()

    if not args.download and not args.recalc:
        args.recalc = True

    ensure_directories()
    if args.download:
        BinanceDownloader().run()
    if args.recalc:
        FactorEngine().run()

    print('\n[OK] Data processing completed successfully!')


if __name__ == '__main__':
    main()
