"""Performance Metrics Calculation for Cryptocurrency Trading Strategies.

This module provides comprehensive performance metrics calculation including
Sharpe ratio, maximum drawdown, regime-specific performance, and other
quantitative finance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def calculate_performance_metrics(nav_history: List[float], dates: List[pd.Timestamp]) -> Dict:
    """Calculate comprehensive performance metrics from NAV history."""
    if len(nav_history) < 2:
        return {}
        
    # Convert to pandas Series for easier manipulation
    nav_series = pd.Series(nav_history, index=dates)
    
    # Calculate daily returns
    returns = nav_series.pct_change().dropna()
    
    # Basic metrics
    total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    rolling_max = nav_series.expanding().max()
    drawdowns = (nav_series - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Sortino ratio (using downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sortino_ratio': sortino_ratio,
        'num_trading_days': len(returns),
        'avg_daily_return': returns.mean(),
        'daily_return_std': returns.std()
    }


def calculate_regime_specific_metrics(
    nav_history: List[float], 
    dates: List[pd.Timestamp], 
    telemetry_history: List[Dict]
) -> Dict:
    """Calculate performance metrics broken down by market regime."""
    if len(nav_history) < 2 or len(telemetry_history) == 0:
        return {}
        
    nav_series = pd.Series(nav_history, index=dates)
    returns = nav_series.pct_change().dropna()
    
    # Extract regime information from telemetry
    regimes = []
    for i, telemetry in enumerate(telemetry_history):
        if i == 0:
            continue  # Skip first entry (no return calculated)
        ar = telemetry.get('absorption_ratio', telemetry.get('ar', 0.5))
        # Simple regime classification based on AR and volatility
        if ar > 0.7:
            regimes.append('extreme_risk')
        elif abs(returns.iloc[i-1]) > 0.02:  # High absolute return
            regimes.append('trend')
        else:
            regimes.append('mean_reversion')
            
    if len(regimes) != len(returns):
        return {}
        
    regime_returns = pd.Series(returns.values, index=regimes)
    
    regime_metrics = {}
    for regime in ['trend', 'mean_reversion', 'extreme_risk']:
        if regime in regime_returns.index:
            regime_ret = regime_returns[regime]
            if len(regime_ret) > 0:
                regime_metrics[f'{regime}_return'] = regime_ret.sum()
                regime_metrics[f'{regime}_sharpe'] = (regime_ret.mean() / regime_ret.std()) * np.sqrt(252) if regime_ret.std() > 0 else 0
                regime_metrics[f'{regime}_count'] = len(regime_ret)
                
    return regime_metrics


def compare_strategies(results_list: List[Dict]) -> pd.DataFrame:
    """Compare multiple strategy results and return comparison DataFrame."""
    comparison_data = []
    
    for result in results_list:
        metrics = calculate_performance_metrics(result['nav'], result['dates'])
        regime_metrics = calculate_regime_specific_metrics(
            result['nav'], result['dates'], result['telemetry']
        )
        
        combined_metrics = {**metrics, **regime_metrics}
        combined_metrics['strategy_name'] = result['strategy_name']
        comparison_data.append(combined_metrics)
        
    return pd.DataFrame(comparison_data)


def generate_performance_report(results: Dict) -> str:
    """Generate a formatted performance report string."""
    metrics = calculate_performance_metrics(results['nav'], results['dates'])
    regime_metrics = calculate_regime_specific_metrics(
        results['nav'], results['dates'], results['telemetry']
    )
    
    report = f"""
=== {results['strategy_name']} Performance Report ===

Overall Performance:
  Total Return: {metrics.get('total_return', 0):.2%}
  Annualized Return: {metrics.get('annualized_return', 0):.2%}
  Volatility: {metrics.get('volatility', 0):.2%}
  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}
  Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}
  Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}
  Win Rate: {metrics.get('win_rate', 0):.2%}
  Profit Factor: {metrics.get('profit_factor', 0):.3f}
  Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}

Regime-Specific Performance:
"""
    
    for regime in ['trend', 'mean_reversion', 'extreme_risk']:
        if f'{regime}_return' in regime_metrics:
            report += f"  {regime.replace('_', ' ').title()}: "
            report += f"{regime_metrics[f'{regime}_return']:.2%} return, "
            report += f"{regime_metrics[f'{regime}_sharpe']:.3f} Sharpe\n"
            
    return report