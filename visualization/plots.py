"""Enhanced Plotting Functions for Strategy Visualization.

This module provides advanced plotting functions for visualizing strategy performance,
position exposures, and regime transitions with professional styling.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List


def plot_enhanced_nav_curve(results_list: List[Dict], save_path: str = None):
    """Plot enhanced NAV curves comparing multiple strategies."""
    plt.figure(figsize=(12, 8))
    
    # Set professional styling
    sns.set_style("whitegrid")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, results in enumerate(results_list):
        color = colors[i % len(colors)]
        nav_series = pd.Series(results['nav'], index=results['dates'])
        plt.plot(nav_series.index, nav_series.values, 
                linewidth=2.5, label=results['strategy_name'], color=color)
    
    plt.title('Strategy Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Net Asset Value (NAV)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved enhanced NAV curve to {save_path}")
    else:
        plt.show()


def plot_position_exposures(results: Dict, save_path: str = None):
    """Plot position exposures over time for a single strategy."""
    positions_df = pd.DataFrame(results['positions'])
    dates = positions_df['date']
    
    # Remove date column and get symbol columns
    symbol_cols = [col for col in positions_df.columns if col != 'date']
    position_data = positions_df[symbol_cols].set_index(dates)
    
    # Create subplot for each symbol or group them
    n_symbols = len(symbol_cols)
    if n_symbols <= 6:
        fig, axes = plt.subplots(n_symbols, 1, figsize=(12, 3*n_symbols), sharex=True)
        if n_symbols == 1:
            axes = [axes]
    else:
        # Group symbols into fewer plots
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Split symbols into groups
        groups = np.array_split(symbol_cols, 3)
        position_groups = []
        for group in groups:
            group_data = position_data[group].sum(axis=1)
            position_groups.append(group_data)
            
        for i, (ax, group_data) in enumerate(zip(axes, position_groups)):
            ax.fill_between(group_data.index, 0, group_data.values, alpha=0.7)
            ax.set_ylabel(f'Group {i+1} Exposure', fontsize=10)
            ax.grid(True, alpha=0.3)
            
    if n_symbols <= 6:
        for i, (ax, symbol) in enumerate(zip(axes, symbol_cols)):
            ax.fill_between(position_data.index, 0, position_data[symbol].values, alpha=0.7)
            ax.set_ylabel(symbol, fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == len(axes) - 1:
                ax.set_xlabel('Date', fontsize=12)
    
    plt.suptitle(f'{results["strategy_name"]} - Position Exposures Over Time', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved position exposure plot to {save_path}")
    else:
        plt.show()


def plot_regime_transitions(telemetry_history: List[Dict], dates: List[pd.Timestamp], save_path: str = None):
    """Plot market regime transitions over time."""
    regimes = []
    ar_values = []
    
    for telemetry in telemetry_history:
        ar = telemetry.get('absorption_ratio', telemetry.get('ar', 0.5))
        ar_values.append(ar)
        if ar > 0.7:
            regimes.append(2)  # Extreme Risk
        elif ar > 0.5:
            regimes.append(1)  # Mean-Reversion  
        else:
            regimes.append(0)  # Trend
            
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot Absorption Ratio
    ax1.plot(dates, ar_values, linewidth=2, color='#1f77b4')
    ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Extreme Risk Threshold')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Regime Boundary')
    ax1.set_ylabel('Absorption Ratio', fontsize=12)
    ax1.set_title('Market Regime Analysis', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Regime Transitions
    colors = ['green', 'orange', 'red']
    regime_names = ['Trend', 'Mean-Reversion', 'Extreme Risk']
    for i, (color, name) in enumerate(zip(colors, regime_names)):
        mask = np.array(regimes) == i
        if np.any(mask):
            ax2.scatter(np.array(dates)[mask], np.array(regimes)[mask], 
                       color=color, s=20, alpha=0.7, label=name)
    
    ax2.set_ylabel('Market Regime', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(regime_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved regime transition plot to {save_path}")
    else:
        plt.show()


def plot_performance_comparison(results_list: List[Dict], save_path: str = None):
    """Create comprehensive performance comparison dashboard."""
    from .metrics import calculate_performance_metrics
    
    # Calculate metrics for all strategies
    metrics_list = []
    strategy_names = []
    
    for results in results_list:
        metrics = calculate_performance_metrics(results['nav'], results['dates'])
        metrics_list.append(metrics)
        strategy_names.append(results['strategy_name'])
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Strategy Performance Comparison Dashboard', fontsize=16, fontweight='bold')
    
    metrics_to_plot = [
        ('annualized_return', 'Annualized Return (%)', lambda x: x * 100),
        ('sharpe_ratio', 'Sharpe Ratio', lambda x: x),
        ('max_drawdown', 'Maximum Drawdown (%)', lambda x: x * 100),
        ('volatility', 'Volatility (%)', lambda x: x * 100),
        ('win_rate', 'Win Rate (%)', lambda x: x * 100),
        ('profit_factor', 'Profit Factor', lambda x: x)
    ]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_names)))
    
    for idx, (metric_key, title, transform) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        values = [transform(metrics.get(metric_key, 0)) for metrics in metrics_list]
        
        bars = ax.bar(strategy_names, values, color=colors)
        ax.set_title(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison dashboard to {save_path}")
    else:
        plt.show()