import pandas as pd
from strategies.base_strategy import BaseStrategy
from backtesting.engine import BacktestingEngine


class EntropyOnlyStrategy(BaseStrategy):

    def get_position_weights(self, date, telemetry, telemetry_history=None, prev_weights=None):
        weights = {sym: telemetry[f'weight_{sym}'] for sym in self.symbols if f'weight_{sym}' in telemetry and not pd.isna(telemetry[f'weight_{sym}'])}
        normalized_weights = self.enforce_market_neutrality(weights, target_gross_exposure=1.0)
        risk_adjusted = self.apply_risk_management(normalized_weights, telemetry, prev_weights=prev_weights, date=date)
        return self.enforce_market_neutrality(risk_adjusted, target_gross_exposure=1.0)

    def get_strategy_name(self):
        return "Entropy-Only Baseline"


def run_entropy_only_strategy(config):
    engine = BacktestingEngine(config)
    strategy = EntropyOnlyStrategy(config)
    return engine.run_backtest(strategy)
