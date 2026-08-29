"""Unified backtesting engine for enhanced baseline and Mamba variants."""

import glob
import hashlib
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch

from models.mamba_model import FEATURE_ORDER, build_training_data, create_mamba_model, train_mamba_model
from utils.gpu_utils import autocast_context, describe_device


DEVICE, DEVICE_INFO = describe_device()
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT',
    'DOGEUSDT', 'TRXUSDT', 'DOTUSDT', 'LTCUSDT', 'LINKUSDT', 'AVAXUSDT'
]


def _ensure_directory_exists(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def calculate_dynamic_hysteresis(turnover_history: list, base_threshold: float = 0.03, window: int = 20) -> float:
    """Adapt the rebalance band to recent trading intensity."""
    if len(turnover_history) < window:
        return base_threshold
    recent = np.mean(turnover_history[-window:])
    long_term = np.mean(turnover_history)
    if long_term <= 1e-8:
        return base_threshold
    scale = max(0.7, min(1.5, recent / long_term))
    return base_threshold * scale


@dataclass
class UnifiedDataBundle:
    weights: pd.DataFrame
    features: pd.DataFrame
    prices_close: dict
    prices_ohlc: dict
    forward_returns: pd.DataFrame
    close_df: pd.DataFrame
    open_df: pd.DataFrame
    close_returns: pd.DataFrame
    asset_vol_20d: pd.DataFrame
    asset_momentum_20d: pd.DataFrame


@dataclass
class StrategySpec:
    name: str
    allow_short: bool = True
    long_budget: float = 0.5
    short_budget: float = 0.5
    max_weight: float = 0.18
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 0.06


@dataclass
class StrategyDecision:
    raw_weights: pd.Series
    exposure: float
    metrics: dict = field(default_factory=dict)
    spec: Optional[StrategySpec] = None


def _normalize_symbol_weights(raw_weights: pd.Series, symbols: list[str], spec: StrategySpec) -> pd.Series:
    weights = raw_weights.copy()
    weights.index = [str(col).replace('weight_', '') for col in weights.index]
    weights = weights.reindex(symbols).fillna(0.0).astype(float)

    if spec.allow_short:
        weights = weights.clip(lower=-spec.max_weight, upper=spec.max_weight)
        long_part = weights.clip(lower=0.0)
        short_part = (-weights.clip(upper=0.0))
        if long_part.sum() > 0:
            long_part = long_part / long_part.sum() * spec.long_budget
        if short_part.sum() > 0:
            short_part = short_part / short_part.sum() * spec.short_budget
        target = long_part - short_part
    else:
        weights = weights.clip(lower=0.0, upper=spec.max_weight)
        if weights.sum() > 0:
            target = weights / weights.sum() * spec.long_budget
        else:
            target = pd.Series(0.0, index=symbols)

    gross = float(target.abs().sum())
    if gross > spec.max_gross_exposure and gross > 1e-8:
        target = target * (spec.max_gross_exposure / gross)

    net = float(target.sum())
    if spec.allow_short and spec.short_budget > 0 and abs(net) > spec.max_net_exposure:
        target = target - net / max(len(target), 1)
        target = target.clip(lower=-spec.max_weight, upper=spec.max_weight)
        gross = float(target.abs().sum())
        if gross > 1e-8:
            long_part = target.clip(lower=0.0)
            short_part = (-target.clip(upper=0.0))
            if long_part.sum() > 0:
                long_part = long_part / long_part.sum() * spec.long_budget
            if short_part.sum() > 0:
                short_part = short_part / short_part.sum() * spec.short_budget
            target = long_part - short_part
        gross = float(target.abs().sum())
        if gross > spec.max_gross_exposure and gross > 1e-8:
            target = target * (spec.max_gross_exposure / gross)
    elif not spec.allow_short and net > spec.max_net_exposure and net > 1e-8:
        target = target * (spec.max_net_exposure / net)

    return target.reindex(symbols).fillna(0.0)


class UnifiedStrategyData:
    """Load the common signal, feature, and pricing inputs."""

    def __init__(self, symbols=None):
        self.symbols = symbols or SYMBOLS

    def load(self, factors_dir, futures_dir) -> UnifiedDataBundle:
        combined_path = os.path.join(factors_dir, 'cross_sectional_weights_and_features.csv')
        if not os.path.exists(combined_path):
            raise FileNotFoundError(
                'Missing cross_sectional_weights_and_features.csv. '
                'Run the factor engine first so portfolio inputs are generated.'
            )

        combined_df = pd.read_csv(combined_path, parse_dates=['date']).set_index('date').sort_index()
        weight_cols = [col for col in combined_df.columns if col.startswith('weight_')]
        feature_cols = [col for col in combined_df.columns if col in FEATURE_ORDER]
        weights_df = combined_df[weight_cols].copy()
        features_df = combined_df[feature_cols].copy()

        prices_close = {}
        prices_ohlc = {}
        open_prices = {}
        close_prices = {}
        for file_path in glob.glob(os.path.join(futures_dir, '*.csv')):
            symbol = os.path.basename(file_path).replace('.csv', '')
            raw = pd.read_csv(file_path, parse_dates=['timestamp']).set_index('timestamp').sort_index()
            daily = raw.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
            }).dropna(how='all').ffill()
            prices_ohlc[symbol] = daily
            prices_close[symbol] = daily['close']
            open_prices[symbol] = daily['open']
            close_prices[symbol] = daily['close']

        open_df = pd.DataFrame(open_prices).sort_index()
        close_df = pd.DataFrame(close_prices).sort_index()
        close_returns = close_df.pct_change()
        asset_vol_20d = close_returns.rolling(20, min_periods=5).std()
        asset_momentum_20d = close_df.div(close_df.shift(20)).sub(1.0)
        forward_returns = open_df.shift(-2).div(open_df.shift(-1)).sub(1.0)
        forward_returns = forward_returns.reindex(features_df.index)

        valid_mask = forward_returns.notna().sum(axis=1) >= max(3, len(self.symbols) // 3)
        valid_dates = features_df.index[valid_mask]
        return UnifiedDataBundle(
            weights=weights_df.loc[valid_dates],
            features=features_df.loc[valid_dates],
            prices_close=prices_close,
            prices_ohlc=prices_ohlc,
            forward_returns=forward_returns.loc[valid_dates],
            close_df=close_df.reindex(valid_dates),
            open_df=open_df.reindex(valid_dates),
            close_returns=close_returns.reindex(valid_dates),
            asset_vol_20d=asset_vol_20d.reindex(valid_dates),
            asset_momentum_20d=asset_momentum_20d.reindex(valid_dates),
        )


class SharedBacktestEngine:
    """Conservative open-to-open backtester shared by all strategy variants."""

    def __init__(self, data_bundle: UnifiedDataBundle, config: dict, symbols=None):
        self.data = data_bundle
        self.config = config
        self.symbols = symbols or SYMBOLS
        self.trade_cfg = config.get('trading_config', {})
        self.rebalance_band = self.trade_cfg.get('rebalance_band', 0.03)
        self.taker_fee = self.trade_cfg.get('taker_fee', 0.0004)
        self.maker_fee = self.trade_cfg.get('maker_fee', 0.00005)
        self.slippage = self.trade_cfg.get('slippage', 0.0003)
        self.execution_mode = self.trade_cfg.get('execution_mode', 'adaptive_maker')
        self.maker_offset_min = self.trade_cfg.get('maker_offset_min', 0.0008)
        self.maker_offset_max = self.trade_cfg.get('maker_offset_max', 0.0060)
        self.maker_vol_multiplier = self.trade_cfg.get('maker_vol_multiplier', 0.35)
        self.max_daily_turnover = self.trade_cfg.get('max_daily_turnover', 1.20)
        self.min_trade_notional = self.trade_cfg.get('min_trade_notional', 0.01)
        self.trade_cost = self.taker_fee + self.slippage

    def run(self, strategy, start_date: str):
        signal_dates = [date for date in self.data.features.index if date >= pd.Timestamp(start_date)]
        if not signal_dates:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        if hasattr(strategy, 'set_backtest_dates'):
            strategy.set_backtest_dates(signal_dates)

        results = []
        telemetry = []
        positions = []
        prev_weights = pd.Series(0.0, index=self.symbols)
        turnover_history = []
        nav = 1.0

        for day_idx, date in enumerate(signal_dates):
            dynamic_band = calculate_dynamic_hysteresis(turnover_history, self.rebalance_band)
            decision = strategy.decide(
                date=date,
                day_idx=day_idx,
                prev_weights=prev_weights,
                turnover_history=turnover_history,
                dynamic_band=dynamic_band,
            )
            spec = decision.spec or StrategySpec(name='default')
            target_weights = _normalize_symbol_weights(decision.raw_weights, self.symbols, spec)
            turnover_hint = float((target_weights - prev_weights).abs().sum())
            desired_weights = target_weights * float(decision.exposure)

            raw_turnover = float((desired_weights - prev_weights).abs().sum())
            if raw_turnover > self.max_daily_turnover and raw_turnover > 1e-8:
                scale = self.max_daily_turnover / raw_turnover
                desired_weights = prev_weights + (desired_weights - prev_weights) * scale
            else:
                scale = 1.0

            simulated = self._simulate_execution(
                date=date,
                prev_weights=prev_weights,
                desired_weights=desired_weights,
                dynamic_band=dynamic_band,
            )
            final_weights = simulated['final_weights']
            turnover = simulated['turnover']
            turnover_history.append(turnover)
            gross_return = simulated['gross_return']
            net_return = simulated['net_return']
            nav *= (1.0 + net_return)

            telemetry_row = {
                'date': date,
                'strategy': spec.name,
                'exp': float(decision.exposure),
                'turnover': turnover,
                'gross_ret': gross_return,
                'net_ret': net_return,
                'rebalance_band': dynamic_band,
                'raw_turnover': raw_turnover,
                'turnover_scale': scale,
                'turnover_hint': turnover_hint,
                'maker_fill_rate': simulated['maker_fill_rate'],
                'maker_fills': simulated['maker_fills'],
                'taker_fallbacks': simulated['taker_fallbacks'],
                'trade_count': simulated['trade_count'],
            }
            telemetry_row.update(decision.metrics)
            telemetry.append(telemetry_row)

            result_row = {
                'date': date,
                'ret': net_return,
                'nav': nav,
                'turnover': turnover,
                'gross_ret': gross_return,
                'cost_bps': simulated['fees_paid'] * 10000.0,
                'maker_fill_rate': simulated['maker_fill_rate'],
                'maker_fills': simulated['maker_fills'],
                'taker_fallbacks': simulated['taker_fallbacks'],
                'trade_count': simulated['trade_count'],
                'exposure': float(decision.exposure),
            }
            results.append(result_row)

            pos_row = {'date': date}
            for symbol in self.symbols:
                pos_row[symbol] = float(final_weights.get(symbol, 0.0))
            positions.append(pos_row)
            prev_weights = final_weights

            if (day_idx + 1) % 250 == 0:
                print(
                    f"  Backtest progress {day_idx + 1}/{len(signal_dates)} "
                    f"| {spec.name} | date={date.date()} | nav={nav:.4f} | exp={decision.exposure:.3f}"
                )

        return (
            pd.DataFrame(results).set_index('date'),
            pd.DataFrame(telemetry).set_index('date'),
            pd.DataFrame(positions).set_index('date').fillna(0.0),
        )

    def _simulate_execution(self, date, prev_weights: pd.Series, desired_weights: pd.Series, dynamic_band: float):
        gross_return = 0.0
        fees_paid = 0.0
        maker_fills = 0
        taker_fallbacks = 0
        trade_count = 0
        turnover = 0.0
        end_notionals = pd.Series(0.0, index=self.symbols)

        for symbol in self.symbols:
            current_weight = float(prev_weights.get(symbol, 0.0))
            desired_weight = float(desired_weights.get(symbol, 0.0))
            delta = desired_weight - current_weight

            bars = self.data.prices_ohlc.get(symbol)
            if bars is None or date not in bars.index:
                end_notionals[symbol] = current_weight
                continue

            loc = bars.index.get_loc(date)
            if loc >= len(bars) - 2:
                end_notionals[symbol] = current_weight
                continue

            open_next = bars['open'].iloc[loc + 1]
            open_after = bars['open'].iloc[loc + 2]

            if any(pd.isna(val) or val <= 0 for val in [open_next, open_after]):
                end_notionals[symbol] = current_weight
                continue

            base_notional_end = current_weight * (open_after / open_next)
            base_pnl = base_notional_end - current_weight
            trade_pnl = 0.0
            fee = 0.0
            trade_notional_end = 0.0

            if abs(delta) > max(dynamic_band, self.min_trade_notional):
                trade_count += 1
                turnover += abs(delta)
                exec_price = open_next
                if self.execution_mode in {'adaptive_maker', 'limit_maker', 'open_maker'}:
                    filled, exec_price = self._simulate_limit_order_fill(
                        symbol=symbol,
                        date=date,
                        bars=bars,
                        loc=loc,
                        open_next=open_next,
                        delta=delta,
                    )
                    if filled:
                        fee = abs(delta) * self.maker_fee
                        maker_fills += 1
                    else:
                        exec_price = open_next
                        fee = abs(delta) * self.trade_cost
                        taker_fallbacks += 1
                else:
                    fee = abs(delta) * self.trade_cost
                    taker_fallbacks += 1

                trade_notional_end = delta * (open_after / exec_price)
                trade_pnl = trade_notional_end - delta

            ending_notional = base_notional_end + trade_notional_end
            end_notionals[symbol] = ending_notional
            gross_return += base_pnl + trade_pnl
            fees_paid += fee

        net_return = gross_return - fees_paid
        nav_after_cost = max(1.0 + net_return, 1e-8)
        final_weights = end_notionals / nav_after_cost
        maker_fill_rate = maker_fills / max(trade_count, 1)
        return {
            'final_weights': final_weights.reindex(self.symbols).fillna(0.0),
            'turnover': float(turnover),
            'gross_return': float(gross_return),
            'net_return': float(net_return),
            'fees_paid': float(fees_paid),
            'maker_fills': int(maker_fills),
            'taker_fallbacks': int(taker_fallbacks),
            'maker_fill_rate': float(maker_fill_rate),
            'trade_count': int(trade_count),
        }

    def _simulate_limit_order_fill(self, symbol: str, date, bars: pd.DataFrame, loc: int, open_next: float, delta: float):
        """Deterministic adaptive maker simulation with taker fallback."""
        next_bar = bars.iloc[loc + 1]
        day_high = float(next_bar.get('high', np.nan))
        day_low = float(next_bar.get('low', np.nan))
        day_close = float(next_bar.get('close', np.nan))
        if any(pd.isna(val) or val <= 0 for val in [day_high, day_low, day_close]):
            return False, open_next

        vol_window = bars['close'].pct_change().iloc[max(0, loc - 14):loc + 1].dropna()
        daily_vol = float(vol_window.std()) if len(vol_window) > 1 else 0.0
        offset = float(np.clip(
            self.maker_vol_multiplier * daily_vol,
            self.maker_offset_min,
            self.maker_offset_max,
        ))

        if delta > 0:
            limit_price = open_next * (1.0 - offset)
            if day_low <= limit_price * 0.999:
                fill_prob = 1.0
            elif day_low <= limit_price:
                fill_prob = 0.5
            else:
                fill_prob = 0.0
        else:
            limit_price = open_next * (1.0 + offset)
            if day_high >= limit_price * 1.001:
                fill_prob = 1.0
            elif day_high >= limit_price:
                fill_prob = 0.5
            else:
                fill_prob = 0.0

        if fill_prob <= 0.0:
            return False, open_next
        if fill_prob < 1.0 and self._deterministic_uniform(symbol, date) >= fill_prob:
            return False, open_next

        vwap_proxy = (day_high + day_low + day_close) / 3.0
        return True, max(vwap_proxy, 1e-8)

    @staticmethod
    def _deterministic_uniform(symbol: str, date) -> float:
        key = f'{SEED}:{symbol}:{pd.Timestamp(date).date()}'.encode('utf-8')
        digest = hashlib.blake2b(key, digest_size=8).digest()
        return int.from_bytes(digest, byteorder='big') / float(2**64)


class MambaTimingPolicy:
    """Mamba S6 exposure policy trained on portfolio state features."""

    def __init__(
        self,
        data_bundle: UnifiedDataBundle,
        model_config: dict,
        train_end: str,
        backtest_start: str,
        target_return_mode: str = 'long_short',
        training_returns: Optional[pd.Series] = None,
        strategy_positions: Optional[pd.DataFrame] = None,
        strategy_telemetry: Optional[pd.DataFrame] = None,
    ):
        self.data = data_bundle
        self.model_config = model_config or {}
        self.train_end = pd.Timestamp(train_end)
        self.backtest_start = pd.Timestamp(backtest_start)
        self.target_return_mode = target_return_mode
        self.seq_len = self.model_config.get('seq_len', 40)
        self.horizon = self.model_config.get('horizon', 10)
        self.fine_tune_window = self.model_config.get('fine_tune_window', 126)
        self.fine_tune_freq = self.model_config.get('fine_tune_freq', 42)
        self.min_exposure = self.model_config.get('label_min_exposure', 0.55)
        self.max_exposure = self.model_config.get('label_max_exposure', 1.15)
        self.feature_lag = int(self.model_config.get('execution_feature_lag', 2))
        self.exposure_smoothing = float(self.model_config.get('exposure_smoothing', 0.35))
        self.max_exposure_step = float(self.model_config.get('max_exposure_step', 0.22))
        self.trend_leverage_bonus = float(self.model_config.get('trend_leverage_bonus', 0.35))
        self.risk_deleveraging_strength = float(self.model_config.get('risk_deleveraging_strength', 0.45))
        self.raw_exposure_floor = self.model_config.get('raw_exposure_floor')
        self.strategy_positions = strategy_positions.copy() if strategy_positions is not None else pd.DataFrame()
        self.strategy_telemetry = strategy_telemetry.copy() if strategy_telemetry is not None else pd.DataFrame()
        self.feature_order = self._resolve_feature_order()
        self.net = None
        self.feature_mean = None
        self.feature_std = None
        self.backtest_dates = None
        self.last_fine_tune_idx = -9999
        self.last_exposure = 1.0
        self.training_returns = training_returns.sort_index() if training_returns is not None else self._build_training_returns()
        self.enriched_features = self._build_enriched_feature_frame()
        self._train_base_model()

    def _build_training_returns(self) -> pd.Series:
        if self.target_return_mode == 'equal_weight':
            eq = self.data.forward_returns.reindex(columns=self.data.open_df.columns)
            return eq.mean(axis=1).fillna(0.0)

        weights = self.data.weights.copy()
        weights.columns = [col.replace('weight_', '') for col in weights.columns]
        weights = weights.reindex(columns=self.data.forward_returns.columns).fillna(0.0)
        long_part = weights.clip(lower=0.0)
        short_part = (-weights.clip(upper=0.0))
        long_norm = long_part.div(long_part.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0) * 0.5
        short_norm = short_part.div(short_part.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0) * 0.5
        market_neutral = (long_norm - short_norm).clip(lower=-0.18, upper=0.18)
        return (market_neutral * self.data.forward_returns.reindex(market_neutral.index)).sum(axis=1).fillna(0.0)

    def _resolve_feature_order(self):
        extra = [
            'base_ret_1d',
            'base_ret_5d',
            'base_ret_20d',
            'base_nav_drawdown_60d',
            'base_turnover_20d',
            'base_position_hhi',
            'base_net_exposure',
            'base_gross_exposure',
            'base_long_exposure',
            'base_short_exposure',
            'base_active_frac',
            'base_realized_exposure',
        ]
        extra.extend([f'base_weight_{symbol}' for symbol in self.data.open_df.columns])
        return FEATURE_ORDER + extra

    def _build_enriched_feature_frame(self) -> pd.DataFrame:
        features = self.data.features.copy()
        if self.strategy_positions.empty:
            return features.reindex(columns=self.feature_order).fillna(0.0)

        pos = self.strategy_positions.reindex(features.index).shift(self.feature_lag).fillna(0.0)
        pos = pos.reindex(columns=self.data.open_df.columns).fillna(0.0)
        abs_pos = pos.abs()

        features['base_net_exposure'] = pos.sum(axis=1)
        features['base_gross_exposure'] = abs_pos.sum(axis=1)
        features['base_long_exposure'] = pos.clip(lower=0.0).sum(axis=1)
        features['base_short_exposure'] = (-pos.clip(upper=0.0)).sum(axis=1)
        features['base_active_frac'] = (abs_pos > 1e-6).sum(axis=1) / max(len(pos.columns), 1)
        features['base_position_hhi'] = (
            abs_pos.div(abs_pos.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0) ** 2
        ).sum(axis=1)

        returns = self.training_returns.reindex(features.index).shift(self.feature_lag).fillna(0.0)
        nav = (1.0 + returns).cumprod()
        features['base_ret_1d'] = returns
        features['base_ret_5d'] = nav.pct_change(5).fillna(0.0)
        features['base_ret_20d'] = nav.pct_change(20).fillna(0.0)
        features['base_nav_drawdown_60d'] = (nav / nav.rolling(60, min_periods=2).max() - 1.0).fillna(0.0)

        if not self.strategy_telemetry.empty and 'turnover' in self.strategy_telemetry.columns:
            turnover = self.strategy_telemetry['turnover'].reindex(features.index).shift(self.feature_lag).fillna(0.0)
        else:
            turnover = pos.diff().abs().sum(axis=1).fillna(abs_pos.sum(axis=1))
        features['base_turnover_20d'] = turnover.rolling(20, min_periods=1).mean()
        if not self.strategy_telemetry.empty and 'exp' in self.strategy_telemetry.columns:
            features['base_realized_exposure'] = (
                self.strategy_telemetry['exp'].reindex(features.index).shift(self.feature_lag).fillna(1.0)
            )
        else:
            features['base_realized_exposure'] = features['base_gross_exposure'].clip(lower=0.0)

        for symbol in pos.columns:
            features[f'base_weight_{symbol}'] = pos[symbol]

        return features.reindex(columns=self.feature_order).fillna(0.0)

    def _fit_feature_scaler(self, train_frame: pd.DataFrame) -> None:
        self.feature_mean = train_frame[self.feature_order].mean()
        self.feature_std = train_frame[self.feature_order].std().replace(0.0, 1.0).fillna(1.0)

    def _normalized_records(self, frame: pd.DataFrame) -> list[dict]:
        normalized = (frame[self.feature_order] - self.feature_mean) / self.feature_std
        return [{col: float(row[col]) for col in self.feature_order} for _, row in normalized.iterrows()]

    def _train_base_model(self) -> None:
        train_mask = self.data.features.index <= self.train_end
        train_frame = self.enriched_features.loc[train_mask, self.feature_order].dropna()
        train_returns = self.training_returns.reindex(train_frame.index).fillna(0.0)
        if len(train_frame) < self.seq_len + self.horizon + 20:
            print('  [Mamba Net] Insufficient training history; using fallback exposure.')
            return

        self._fit_feature_scaler(train_frame)
        records = self._normalized_records(train_frame)
        x_data, y_exposure, y_regime, y_mdd_aux = build_training_data(
            feature_history=records,
            baseline_returns=train_returns.tolist(),
            feature_order=self.feature_order,
            seq_len=self.seq_len,
            horizon=self.horizon,
            min_exposure=self.min_exposure,
            max_exposure=self.max_exposure,
        )
        if x_data.numel() == 0:
            print('  [Mamba Net] Empty training tensors; using fallback exposure.')
            return

        model_cfg = {
            'input_dim': len(self.feature_order),
            'd_model': self.model_config.get('d_model', 48),
            'd_state': self.model_config.get('d_state', 16),
            'n_layers': self.model_config.get('n_layers', 2),
            'seq_len': self.seq_len,
            'n_regimes': 3,
            'max_exposure': self.model_config.get('model_max_exposure', 1.25),
        }
        self.net = create_mamba_model(model_cfg).to(DEVICE)
        train_cfg = {
            'train_epochs': self.model_config.get('train_epochs', 40),
            'learning_rate': self.model_config.get('learning_rate', 1e-3),
            'lambda_mdd': self.model_config.get('lambda_mdd', 0.10),
            'lambda_smooth': self.model_config.get('lambda_smooth', 0.05),
            'lambda_regime': self.model_config.get('lambda_regime', 0.2),
            'batch_size': self.model_config.get('batch_size', 256),
            'use_amp': self.model_config.get('use_amp', True),
            'verbose': True,
        }
        self.net = train_mamba_model(
            self.net,
            x_data,
            y_exposure,
            y_regime,
            y_mdd_aux,
            train_cfg,
        )
        self.net.eval()
        print(
            f"  [Mamba Net] Base model trained for {self.target_return_mode} "
            f"with {DEVICE_INFO['device']}."
        )

    def set_backtest_dates(self, backtest_dates) -> None:
        self.backtest_dates = list(backtest_dates)

    def _fine_tune(self, current_idx: int) -> None:
        if self.net is None or self.backtest_dates is None:
            return
        if current_idx - self.last_fine_tune_idx < self.fine_tune_freq:
            return

        current_date = self.backtest_dates[current_idx]
        hist_frame = self.enriched_features.loc[:current_date, self.feature_order].iloc[-self.fine_tune_window:]
        hist_returns = self.training_returns.reindex(hist_frame.index).fillna(0.0)
        if len(hist_frame) < self.seq_len + self.horizon + 10:
            return

        records = self._normalized_records(hist_frame)
        x_data, y_exposure, y_regime, y_mdd_aux = build_training_data(
            feature_history=records,
            baseline_returns=hist_returns.tolist(),
            feature_order=self.feature_order,
            seq_len=self.seq_len,
            horizon=self.horizon,
            min_exposure=self.min_exposure,
            max_exposure=self.max_exposure,
        )
        if x_data.numel() == 0:
            return

        fine_tune_cfg = {
            'train_epochs': self.model_config.get('fine_tune_epochs', 3),
            'learning_rate': self.model_config.get('fine_tune_lr', 1e-4),
            'lambda_mdd': self.model_config.get('lambda_mdd', 0.10),
            'lambda_smooth': self.model_config.get('lambda_smooth', 0.05),
            'lambda_regime': self.model_config.get('lambda_regime', 0.2),
            'batch_size': self.model_config.get('batch_size', 256),
            'use_amp': self.model_config.get('use_amp', True),
            'verbose': False,
        }
        self.net = train_mamba_model(
            self.net,
            x_data,
            y_exposure,
            y_regime,
            y_mdd_aux,
            fine_tune_cfg,
        )
        self.net.eval()
        self.last_fine_tune_idx = current_idx

    def _infer(self, date: pd.Timestamp) -> tuple[float, str, np.ndarray]:
        if self.net is None or self.feature_mean is None:
            return 1.0, 'fallback', np.asarray([0.0, 1.0, 0.0], dtype=float)
        recent = self.enriched_features.loc[:date, self.feature_order].tail(self.seq_len)
        if len(recent) < self.seq_len:
            return 1.0, 'warmup', np.asarray([0.0, 1.0, 0.0], dtype=float)

        normalized = (recent - self.feature_mean) / self.feature_std
        x_tensor = torch.tensor([normalized.values.tolist()], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            context = autocast_context(DEVICE, enabled=self.model_config.get('use_amp', True))
            with context:
                exposure, regime_probs = self.net(x_tensor)
            exposure_val = float(exposure.item())
            regime_idx = int(torch.argmax(regime_probs, dim=-1).item())
            regime_prob_values = regime_probs.detach().cpu().numpy()[0]
        if regime_idx == 2:
            return exposure_val, 'risk_off', regime_prob_values
        if regime_idx == 0:
            return exposure_val, 'trend', regime_prob_values
        return exposure_val, 'balanced', regime_prob_values

    def _causal_exposure_multiplier(self, date: pd.Timestamp, mode: str, regime_probs: np.ndarray) -> tuple[float, dict]:
        """Translate current observable state into a leverage multiplier."""
        row = self.enriched_features.loc[date]
        base_ret_5d = float(row.get('base_ret_5d', 0.0))
        base_ret_20d = float(row.get('base_ret_20d', 0.0))
        btc_momentum = float(row.get('btc_momentum_20d', 0.0))
        drawdown = float(row.get('base_nav_drawdown_60d', row.get('portfolio_drawdown_60d', 0.0)))
        vol_ann = float(row.get('portfolio_volatility_20d', 0.0)) * np.sqrt(365.0)
        absorption_ratio = float(row.get('absorption_ratio', 0.0))
        dispersion = float(row.get('dispersion', 0.0))

        momentum_score = (
            0.50 * np.tanh(base_ret_20d / 0.08)
            + 0.30 * np.tanh(base_ret_5d / 0.035)
            + 0.20 * np.tanh(btc_momentum / 0.18)
        )
        structure_score = (
            0.35 * np.tanh((dispersion - 0.45) / 0.20)
            - 0.35 * np.tanh((absorption_ratio - 0.72) / 0.10)
            - 0.30 * np.tanh((vol_ann - 0.22) / 0.08)
        )
        trend_probability = float(regime_probs[0]) if len(regime_probs) > 0 else 0.0
        risk_probability = float(regime_probs[2]) if len(regime_probs) > 2 else 0.0
        opportunity_score = float(np.clip(momentum_score + structure_score + 0.35 * trend_probability, -1.5, 1.5))

        leverage_multiplier = 1.0 + self.trend_leverage_bonus * max(opportunity_score, 0.0)
        risk_score = 0.0
        risk_score += max(vol_ann - 0.28, 0.0) / 0.12
        risk_score += max(absorption_ratio - 0.78, 0.0) / 0.12
        risk_score += max(-drawdown - 0.12, 0.0) / 0.12
        risk_score += 0.65 * risk_probability
        if mode == 'risk_off':
            risk_score += 0.35
        deleveraging = 1.0 / (1.0 + self.risk_deleveraging_strength * max(risk_score, 0.0))

        multiplier = float(np.clip(leverage_multiplier * deleveraging, 0.35, 1.55))
        metrics = {
            'base_ret_5d': base_ret_5d,
            'base_ret_20d': base_ret_20d,
            'base_nav_drawdown_60d': drawdown,
            'trend_probability': trend_probability,
            'risk_probability': risk_probability,
            'opportunity_score': opportunity_score,
            'exposure_multiplier': multiplier,
        }
        return multiplier, metrics

    def __call__(self, date, day_idx, turnover_hint):
        self._fine_tune(day_idx)
        raw_exposure, mode, regime_probs = self._infer(date)
        if self.raw_exposure_floor is not None:
            raw_exposure = max(raw_exposure, float(self.raw_exposure_floor))
        row = self.data.features.loc[date]
        vol = float(row.get('portfolio_volatility_20d', 0.0))
        vol_ann = vol * np.sqrt(365.0)
        drawdown = float(row.get('portfolio_drawdown_60d', 0.0))
        final_exposure = float(np.clip(raw_exposure, 0.0, self.model_config.get('model_max_exposure', 1.25)))
        self.last_exposure = final_exposure
        metrics = {
            'mode': mode,
            'mamba_regime': mode,
            'raw_mamba_exposure': raw_exposure,
            'absorption_ratio': float(row.get('absorption_ratio', 0.0)),
            'dispersion': float(row.get('dispersion', 0.0)),
            'portfolio_volatility_20d': vol,
            'portfolio_volatility_20d_ann': vol_ann,
            'portfolio_drawdown_60d': drawdown,
            'turnover_hint': turnover_hint,
        }
        return final_exposure, metrics


class EnhancedLongShortAllocator:
    """Signal smoothing, momentum confirmation, and inverse-vol scaling."""

    def __init__(self, data_bundle: UnifiedDataBundle, config: dict, symbols=None):
        self.data = data_bundle
        self.config = config or {}
        self.symbols = symbols or SYMBOLS
        self.smoothing_window = int(self.config.get('signal_smoothing_window', 5))
        self.rebalance_every = int(self.config.get('rebalance_every', 3))
        self.drift_threshold = float(self.config.get('drift_threshold', 0.20))
        self.momentum_scale = float(self.config.get('momentum_scale', 0.20))
        self.momentum_strength = float(self.config.get('momentum_strength', 0.45))
        self.vol_floor = float(self.config.get('vol_floor', 0.015))
        self.turnover_blend = float(self.config.get('turnover_blend', 0.25))
        self.risk_override_vol = float(self.config.get('risk_override_vol', 0.22))
        self.risk_override_drawdown = float(self.config.get('risk_override_drawdown', -0.14))
        trade_cfg = self.config.get('trading_fallback', {})
        self.spec = StrategySpec(
            name='baseline',
            allow_short=True,
            long_budget=0.5,
            short_budget=0.5,
            max_weight=float(self.config.get('max_weight', trade_cfg.get('max_weight', 0.18))),
            max_gross_exposure=float(self.config.get('max_gross_exposure', trade_cfg.get('max_gross_exposure', 1.0))),
            max_net_exposure=float(self.config.get('max_net_exposure', trade_cfg.get('max_net_exposure', 0.06))),
        )
        self.last_target = pd.Series(0.0, index=self.symbols)
        self.last_rebalance_day = -10**9

    def set_name(self, name: str) -> None:
        self.spec.name = name

    def _smoothed_signal(self, date: pd.Timestamp) -> pd.Series:
        hist = self.data.weights.loc[:date].tail(self.smoothing_window)
        if hist.empty:
            return pd.Series(0.0, index=self.symbols)
        return hist.mean(axis=0)

    def candidate_weights(self, date: pd.Timestamp) -> tuple[pd.Series, dict]:
        raw = self._smoothed_signal(date)
        raw.index = [str(col).replace('weight_', '') for col in raw.index]
        raw = raw.reindex(self.symbols).fillna(0.0)

        momentum = self.data.asset_momentum_20d.loc[date].reindex(self.symbols).fillna(0.0)
        vol = self.data.asset_vol_20d.loc[date].reindex(self.symbols)
        vol = vol.replace(0.0, np.nan).fillna(vol.median()).fillna(self.vol_floor).clip(lower=self.vol_floor)

        momentum_multiplier = 1.0 + self.momentum_strength * np.tanh(np.sign(raw) * momentum / self.momentum_scale)
        momentum_multiplier = pd.Series(momentum_multiplier, index=self.symbols).clip(lower=0.35, upper=1.65)
        inv_vol_multiplier = (vol.median() / vol).clip(lower=0.60, upper=1.55)

        adjusted = raw * momentum_multiplier * inv_vol_multiplier
        normalized = _normalize_symbol_weights(adjusted, self.symbols, self.spec)
        metrics = {
            'signal_strength': float(normalized.abs().sum()),
            'signal_dispersion': float(adjusted.std()),
            'mean_abs_momentum_20d': float(momentum.abs().mean()),
            'mean_asset_vol_20d': float(vol.mean()),
        }
        return normalized, metrics

    def resolve_target(self, date: pd.Timestamp, day_idx: int, prev_weights: pd.Series, exposure: float, metrics: dict):
        candidate, alloc_metrics = self.candidate_weights(date)
        row = self.data.features.loc[date]
        signal_drift = float((candidate - self.last_target).abs().sum())
        due_rebalance = (day_idx - self.last_rebalance_day) >= self.rebalance_every
        vol_ann = float(row.get('portfolio_volatility_20d', 0.0)) * np.sqrt(365.0)
        risk_override = (
            vol_ann > self.risk_override_vol
            or float(row.get('portfolio_drawdown_60d', 0.0)) < self.risk_override_drawdown
            or exposure < 0.55
        )

        if day_idx == 0 or due_rebalance or risk_override or signal_drift > self.drift_threshold:
            blended = (1.0 - self.turnover_blend) * candidate + self.turnover_blend * self.last_target
            target = _normalize_symbol_weights(blended, self.symbols, self.spec)
            self.last_target = target
            self.last_rebalance_day = day_idx
            rebalance_state = 'refresh'
        else:
            target = self.last_target.copy()
            rebalance_state = 'hold'

        metrics.update(alloc_metrics)
        metrics.update({
            'rebalance_state': rebalance_state,
            'signal_drift': signal_drift,
            'portfolio_volatility_20d_ann': vol_ann,
            'allocator_due_rebalance': int(due_rebalance),
            'allocator_risk_override': int(risk_override),
        })
        return target, metrics


class EnhancedBaselineStrategy:
    def __init__(self, data_bundle: UnifiedDataBundle, config: dict, symbols=None):
        self.data = data_bundle
        self.symbols = symbols or SYMBOLS
        alloc_cfg = dict(config.get('baseline_config', {}))
        alloc_cfg['trading_fallback'] = config.get('trading_config', {})
        self.allocator = EnhancedLongShortAllocator(data_bundle, alloc_cfg, self.symbols)
        self.allocator.set_name('baseline')

    def decide(self, date, day_idx, prev_weights, turnover_history, dynamic_band):
        metrics = {
            'mode': 'baseline_core',
            'turnover_hint': float((self.allocator.last_target - prev_weights).abs().sum()),
            'portfolio_volatility_20d': float(self.data.features.loc[date].get('portfolio_volatility_20d', 0.0)),
            'portfolio_drawdown_60d': float(self.data.features.loc[date].get('portfolio_drawdown_60d', 0.0)),
            'absorption_ratio': float(self.data.features.loc[date].get('absorption_ratio', 0.0)),
            'dispersion': float(self.data.features.loc[date].get('dispersion', 0.0)),
            'crowding_hhi': float(self.data.features.loc[date].get('crowding_hhi', 0.0)),
        }
        target, metrics = self.allocator.resolve_target(date, day_idx, prev_weights, 1.0, metrics)
        return StrategyDecision(raw_weights=target, exposure=1.0, metrics=metrics, spec=self.allocator.spec)


class OverlayMambaStrategy:
    def __init__(
        self,
        base_strategy,
        data_bundle: UnifiedDataBundle,
        config: dict,
        strategy_name: str,
        training_returns: Optional[pd.Series] = None,
        strategy_positions: Optional[pd.DataFrame] = None,
        strategy_telemetry: Optional[pd.DataFrame] = None,
    ):
        self.base_strategy = base_strategy
        self.data = data_bundle
        self.strategy_name = strategy_name
        target_return_mode = 'equal_weight' if strategy_name == 'mamba_equal_weight' else 'long_short'
        overlay_cfg = config.get('mamba_overlay_config', {})
        strategy_cfg = overlay_cfg.get(strategy_name, {})
        policy_cfg = dict(config.get('model_config', {}))
        policy_cfg.update(strategy_cfg)
        self.exposure_policy = MambaTimingPolicy(
            data_bundle=data_bundle,
            model_config=policy_cfg,
            train_end=config.get('train_end', '2021-12-31'),
            backtest_start=config.get('backtest_start', '2022-04-01'),
            target_return_mode=target_return_mode,
            training_returns=training_returns,
            strategy_positions=strategy_positions,
            strategy_telemetry=strategy_telemetry,
        )

    def set_backtest_dates(self, backtest_dates):
        self.exposure_policy.set_backtest_dates(backtest_dates)

    def decide(self, date, day_idx, prev_weights, turnover_history, dynamic_band):
        base_decision = self.base_strategy.decide(date, day_idx, prev_weights, turnover_history, dynamic_band)
        turnover_hint = float((base_decision.raw_weights - prev_weights).abs().sum())
        exposure, metrics = self.exposure_policy(date=date, day_idx=day_idx, turnover_hint=turnover_hint)
        metrics.update(base_decision.metrics)
        metrics['mode'] = self.strategy_name
        metrics['base_mode'] = base_decision.metrics.get('mode', '')
        base_spec = base_decision.spec or StrategySpec(name=self.strategy_name)
        overlay_spec = StrategySpec(
            name=self.strategy_name,
            allow_short=base_spec.allow_short,
            long_budget=base_spec.long_budget,
            short_budget=base_spec.short_budget,
            max_weight=base_spec.max_weight,
            max_gross_exposure=base_spec.max_gross_exposure,
            max_net_exposure=base_spec.max_net_exposure,
        )
        return StrategyDecision(raw_weights=base_decision.raw_weights, exposure=exposure, metrics=metrics, spec=overlay_spec)


class EqualWeightStrategy:
    def __init__(self, data_bundle: UnifiedDataBundle, config: dict, symbols=None):
        self.data = data_bundle
        self.symbols = symbols or SYMBOLS
        eq_cfg = config.get('equal_weight_config', {})
        self.rebalance_every = int(eq_cfg.get('rebalance_every', 7))
        self.drift_threshold = float(eq_cfg.get('drift_threshold', 0.12))
        self.min_momentum_filter = float(eq_cfg.get('min_momentum_filter', -1.0))
        self.spec = StrategySpec(
            name='equal_weight',
            allow_short=False,
            long_budget=1.0,
            short_budget=0.0,
            max_weight=float(eq_cfg.get('max_weight', 0.20)),
            max_gross_exposure=float(eq_cfg.get('max_gross_exposure', 1.0)),
            max_net_exposure=float(eq_cfg.get('max_net_exposure', 1.0)),
        )
        self.last_target = pd.Series(0.0, index=self.symbols)
        self.last_rebalance_day = -10**9

    def _candidate(self, date: pd.Timestamp) -> pd.Series:
        momentum = self.data.asset_momentum_20d.loc[date].reindex(self.symbols).fillna(0.0)
        eligible = momentum[momentum >= self.min_momentum_filter].index.tolist()
        if not eligible:
            eligible = list(self.symbols)
        raw = pd.Series(0.0, index=self.symbols)
        raw.loc[eligible] = 1.0
        return _normalize_symbol_weights(raw, self.symbols, self.spec)

    def decide(self, date, day_idx, prev_weights, turnover_history, dynamic_band):
        candidate = self._candidate(date)
        signal_drift = float((candidate - self.last_target).abs().sum())
        due_rebalance = (day_idx - self.last_rebalance_day) >= self.rebalance_every
        if day_idx == 0 or due_rebalance or signal_drift > self.drift_threshold:
            target = candidate
            self.last_target = target
            self.last_rebalance_day = day_idx
            state = 'refresh'
        else:
            target = self.last_target.copy()
            state = 'hold'

        metrics = {
            'mode': 'equal_weight',
            'rebalance_state': state,
            'signal_drift': signal_drift,
            'eligible_count': int((target > 0).sum()),
            'portfolio_volatility_20d': float(self.data.features.loc[date].get('portfolio_volatility_20d', 0.0)),
            'portfolio_drawdown_60d': float(self.data.features.loc[date].get('portfolio_drawdown_60d', 0.0)),
        }
        return StrategyDecision(raw_weights=target, exposure=1.0, metrics=metrics, spec=self.spec)


def _load_bundle(config):
    results_dir = config.get('results_dir', 'results_integrated')
    data_dir = config.get('data_dir', 'data_integrated')
    factors_dir = config.get('factors_dir', os.path.join(data_dir, 'factors'))
    futures_dir = config.get('futures_dir', os.path.join(data_dir, 'futures'))
    _ensure_directory_exists(results_dir)
    return UnifiedStrategyData(config.get('symbols', SYMBOLS)).load(factors_dir, futures_dir)


def _slice_backtest_window(frame: pd.DataFrame, start_date: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame.loc[frame.index >= pd.Timestamp(start_date)].copy()


def run_all_strategies(config):
    data_bundle = _load_bundle(config)
    engine = SharedBacktestEngine(data_bundle, config, config.get('symbols', SYMBOLS))
    start_date = config.get('backtest_start', '2022-04-01')
    full_start = str(data_bundle.features.index.min().date())

    results = {}
    telemetry = {}
    positions = {}

    print('  Building baseline training history...')
    baseline_training = EnhancedBaselineStrategy(data_bundle, config, config.get('symbols', SYMBOLS))
    baseline_res_full, baseline_telem_full, baseline_pos_full = engine.run(baseline_training, start_date=full_start)

    print('  Running baseline strategy...')
    baseline_live = EnhancedBaselineStrategy(data_bundle, config, config.get('symbols', SYMBOLS))
    baseline_res, baseline_telem, baseline_pos = engine.run(baseline_live, start_date=start_date)
    results['baseline'] = baseline_res
    telemetry['baseline'] = baseline_telem
    positions['baseline'] = baseline_pos

    print('  Running mamba strategy...')
    mamba_base = EnhancedBaselineStrategy(data_bundle, config, config.get('symbols', SYMBOLS))
    mamba_strategy = OverlayMambaStrategy(
        base_strategy=mamba_base,
        data_bundle=data_bundle,
        config=config,
        strategy_name='mamba',
        training_returns=baseline_res_full['ret'] if not baseline_res_full.empty else None,
        strategy_positions=baseline_pos_full,
        strategy_telemetry=baseline_telem_full,
    )
    mamba_res, mamba_telem, mamba_pos = engine.run(mamba_strategy, start_date=start_date)
    results['mamba'] = mamba_res
    telemetry['mamba'] = mamba_telem
    positions['mamba'] = mamba_pos

    print('  Building equal_weight training history...')
    equal_training = EqualWeightStrategy(data_bundle, config, config.get('symbols', SYMBOLS))
    equal_res_full, equal_telem_full, equal_pos_full = engine.run(equal_training, start_date=full_start)

    print('  Running equal_weight strategy...')
    equal_live = EqualWeightStrategy(data_bundle, config, config.get('symbols', SYMBOLS))
    equal_res, equal_telem, equal_pos = engine.run(equal_live, start_date=start_date)
    results['equal_weight'] = equal_res
    telemetry['equal_weight'] = equal_telem
    positions['equal_weight'] = equal_pos

    print('  Running mamba_equal_weight strategy...')
    mamba_equal_base = EqualWeightStrategy(data_bundle, config, config.get('symbols', SYMBOLS))
    mamba_equal_base.spec.name = 'mamba_equal_weight'
    mamba_equal_strategy = OverlayMambaStrategy(
        base_strategy=mamba_equal_base,
        data_bundle=data_bundle,
        config=config,
        strategy_name='mamba_equal_weight',
        training_returns=equal_res_full['ret'] if not equal_res_full.empty else None,
        strategy_positions=equal_pos_full,
        strategy_telemetry=equal_telem_full,
    )
    mamba_equal_res, mamba_equal_telem, mamba_equal_pos = engine.run(mamba_equal_strategy, start_date=start_date)
    results['mamba_equal_weight'] = mamba_equal_res
    telemetry['mamba_equal_weight'] = mamba_equal_telem
    positions['mamba_equal_weight'] = mamba_equal_pos

    return results, telemetry, positions, data_bundle.prices_close
