import os
import pandas as pd
import numpy as np
import duckdb
from typing import Dict, List
from strategies.base_strategy import BaseStrategy


class BacktestingEngine:
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = config['data_dir']
        self.results_dir = config['results_dir']
        self.symbols = config['symbols']
        self.constant_funding_rate = config.get('constant_funding_rate', 0.0001)
        
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        self.duckdb_conn = duckdb.connect(':memory:')
            
    def _ensure_parquet(self):
        factors_dir = os.path.join(self.data_dir, 'factors')
        futures_dir = os.path.join(self.data_dir, 'futures')
        
        weights_csv = os.path.join(factors_dir, 'cross_sectional_weights_and_features.csv')
        weights_parquet = os.path.join(factors_dir, 'cross_sectional_weights_and_features.parquet')
        
        if not os.path.exists(weights_parquet) and os.path.exists(weights_csv):
            df = pd.read_csv(weights_csv)
            # Standardize date column just in case
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.normalize()
            df.to_parquet(weights_parquet)
            
        prices_parquet = os.path.join(futures_dir, 'prices_compiled.parquet')
        if not os.path.exists(prices_parquet):
            prices_list = []
            for symbol in self.symbols:
                price_path = os.path.join(futures_dir, f"{symbol}.csv")
                if os.path.exists(price_path):
                    df = pd.read_csv(price_path)
                    if 'timestamp' in df.columns:
                        df = df.rename(columns={'timestamp': 'date'})
                    df['date'] = pd.to_datetime(df['date']).dt.normalize()
                    
                    df_resampled = df.groupby('date').last()[['close']]
                    df_resampled = df_resampled.rename(columns={'close': symbol})
                    prices_list.append(df_resampled)
                    
            if prices_list:
                prices_df = pd.concat(prices_list, axis=1)
                prices_df.sort_index(inplace=True)
                prices_df = prices_df.ffill()
                prices_df.reset_index().to_parquet(prices_parquet)
                
    def load_data(self) -> pd.DataFrame:
        self._ensure_parquet()
        factors_dir = os.path.join(self.data_dir, 'factors')
        futures_dir = os.path.join(self.data_dir, 'futures')
        
        weights_parquet = os.path.join(factors_dir, 'cross_sectional_weights_and_features.parquet')
        prices_parquet = os.path.join(futures_dir, 'prices_compiled.parquet')
        
        if not os.path.exists(weights_parquet) or not os.path.exists(prices_parquet):
            raise FileNotFoundError("Parquet files could not be generated. Check if CSV data exists.")
        
        # SQL-based fast join via DuckDB
        query = f"""
        SELECT f.*, p.* EXCLUDE (date)
        FROM '{weights_parquet}' f
        LEFT JOIN '{prices_parquet}' p ON f.date = p.date
        ORDER BY f.date
        """
        return self.duckdb_conn.execute(query).df()
        
    def run_backtest(self, strategy: BaseStrategy) -> Dict:
        df = self.load_data()
        
        # Determine available price columns based on config symbols present in joined Dataframe
        price_cols = [s for s in self.symbols if s in df.columns]
        prices_df = df[price_cols]
        
        # Vectorized T+1 returns computation
        fwd_returns_df = prices_df.shift(-1) / prices_df - 1
        
        n_days = len(df)
        nav = np.ones(n_days)
        
        telemetry_history = []
        positions = []
        dates = pd.to_datetime(df['date']).tolist()
        
        prev_weights = {symbol: 0.0 for symbol in self.symbols}
        
        # Convert entire DataFrame to C-optimized list of dicts for extremely fast sequential iteration
        # This massively outperforms Pandas `.iloc[t].to_dict()`
        records = df.to_dict('records')
        fwd_returns_records = fwd_returns_df.to_dict('records')
        
        fee_rate = strategy.realistic_fee_rate
        slip_rate = strategy.slippage_rate
        fund_rate = self.constant_funding_rate
        
        for t, row in enumerate(records):
            telemetry = row
            # Mask out weights for assets where price is NaN today (can't trade it)
            for sym in self.symbols:
                if pd.isna(row.get(sym, np.nan)):
                    telemetry[f'weight_{sym}'] = 0.0
            
            telemetry['date'] = dates[t]
            telemetry_history.append(telemetry)
            
            if hasattr(strategy, 'update_model'):
                strategy.update_model(dates[t], telemetry_history, nav[:t+1])
            
            weights = strategy.get_position_weights(
                dates[t], telemetry,
                telemetry_history=telemetry_history,
                prev_weights=prev_weights,
            )
            
            positions.append({**weights, 'date': dates[t]})
            
            if t < n_days - 1:
                step_ret = 0.0
                step_cost = 0.0
                step_funding = 0.0
                next_prev_weights = {}
                
                t_plus_1_rets = fwd_returns_records[t]
                
                for sym in self.symbols:
                    w_target = weights.get(sym, 0.0)
                    w_prev = prev_weights.get(sym, 0.0)
                    
                    # Transaction Cost
                    dw = abs(w_target - w_prev)
                    if dw > 1e-10:
                        step_cost += dw * (fee_rate + slip_rate)
                        
                    # Crypto Funding Rates: Longs pay funding (w * rate), Shorts receive funding (-w * rate)
                    # Simplified as position * funding_rate
                    step_funding += w_target * fund_rate
                    
                    # Return contribution
                    asset_ret = t_plus_1_rets.get(sym, np.nan)
                    if pd.notna(asset_ret) and w_target != 0:
                        ret_contrib = w_target * asset_ret
                        step_ret += ret_contrib
                        next_prev_weights[sym] = w_target * (1 + asset_ret)
                    else:
                        next_prev_weights[sym] = w_target
                
                gross_port_ret = 1 + step_ret
                if gross_port_ret > 0:
                    for sym in self.symbols:
                        next_prev_weights[sym] /= gross_port_ret
                        
                nav[t+1] = nav[t] * (1 + step_ret - step_cost - step_funding)
                prev_weights = next_prev_weights
                
        self._save_results(positions, list(nav), telemetry_history, strategy)
        
        return {
            'nav': list(nav),
            'positions': positions,
            'telemetry': telemetry_history,
            'dates': dates,
            'strategy_name': strategy.get_strategy_name()
        }
        
    def _save_results(
        self, 
        position_history: List[Dict], 
        nav_history: List[float], 
        telemetry_history: List[Dict],
        strategy: BaseStrategy
    ):
        strategy_name = strategy.get_strategy_name().lower().replace(' ', '_')
        
        nav_df = pd.DataFrame({
            'date': [p['date'] for p in position_history],
            'nav': nav_history
        })
        nav_df.to_csv(os.path.join(self.results_dir, f'{strategy_name}_nav.csv'), index=False)
        
        positions_df = pd.DataFrame(position_history)
        positions_df.to_csv(os.path.join(self.results_dir, f'{strategy_name}_positions.csv'), index=False)
        
        telemetry_df = pd.DataFrame(telemetry_history)
        telemetry_df.to_csv(os.path.join(self.results_dir, f'{strategy_name}_telemetry.csv'), index=False)