from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict


class BaseStrategy(ABC):

    def __init__(self, config: Dict):
        self.config = config
        self.symbols = config['symbols']
        self.realistic_fee_rate = config['realistic_fee_rate']
        self.slippage_rate = config['slippage_rate']
        self.cooling_period = config['cooling_period']
        self.hysteresis_threshold = config['hysteresis_threshold']
        self.ar_stop_threshold = config['ar_stop_threshold']
        self.constant_funding_rate = config.get('constant_funding_rate', 0.0001)
        self._last_trade_time: Dict[str, pd.Timestamp] = {}

    @abstractmethod
    def get_position_weights(self, date, telemetry, **kwargs):
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        pass

    def apply_risk_management(self, weights, telemetry, prev_weights=None, date=None):
        ar = telemetry.get('absorption_ratio', telemetry.get('ar', 0.5))
        if ar > self.ar_stop_threshold:
            return {s: 0.0 for s in self.symbols}

        if prev_weights is None:
            prev_weights = {s: 0.0 for s in self.symbols}

        filtered = {}
        for sym in self.symbols:
            new_w = weights.get(sym, 0.0)
            old_w = prev_weights.get(sym, 0.0)

            if abs(new_w - old_w) < self.hysteresis_threshold:
                filtered[sym] = old_w
            else:
                if date is not None and old_w == 0.0 and new_w != 0.0:
                    last = self._last_trade_time.get(sym)
                    if last is not None:
                        days_since = (date - last).days
                        if days_since < self.cooling_period:
                            filtered[sym] = 0.0
                            continue

                filtered[sym] = new_w
                if date is not None and new_w != old_w:
                    self._last_trade_time[sym] = date

        return filtered

    def enforce_market_neutrality(self, weights, target_gross_exposure=1.0):
        active = {s: w for s, w in weights.items() if w != 0}
        if not active:
            return weights

        # Mean centering active weights to achieve exact 0 net
        mean_w = sum(active.values()) / len(active)
        centered = {s: w - mean_w for s, w in active.items()}
        
        # Scale to max gross exposure
        total_abs_centered = sum(abs(w) for w in centered.values())
        if total_abs_centered < 1e-10:
            return {s: 0.0 for s in self.symbols}
            
        result = {s: 0.0 for s in self.symbols}
        for s, w in centered.items():
            result[s] = (w / total_abs_centered) * target_gross_exposure

        return result
