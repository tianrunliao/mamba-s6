"""Automated Report Generation for Strategy Backtesting Results.

This module provides functions to generate comprehensive reports combining
performance metrics, visualizations, and analysis summaries.
"""

import os
import pandas as pd
from typing import Dict, List
from visualization.plots import (
    plot_enhanced_nav_curve, 
    plot_position_exposures, 
    plot_regime_transitions,
    plot_performance_comparison
)
from backtesting.metrics import (
    calculate_performance_metrics,
    calculate_regime_specific_metrics,
    generate_performance_report
)


def generate_comprehensive_report(
    results_list: List[Dict], 
    config: Dict,
    output_dir: str = None
) -> None:
    """Generate comprehensive report with all visualizations and metrics."""
    if output_dir is None:
        output_dir = config.get('results_dir', 'results')
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Generating comprehensive strategy report...")
    
    # 1. Generate enhanced NAV curve comparison
    nav_plot_path = os.path.join(output_dir, 'enhanced_nav_curve.png')
    plot_enhanced_nav_curve(results_list, save_path=nav_plot_path)
    
    # 2. Generate position exposure plots for each strategy
    for i, results in enumerate(results_list):
        pos_plot_path = os.path.join(output_dir, f'enhanced_position_plots_{i}.png')
        plot_position_exposures(results, save_path=pos_plot_path)
    
    # 3. Generate regime transition plot (using first strategy's telemetry)
    if results_list:
        regime_plot_path = os.path.join(output_dir, 'regime_analysis_new.png')
        plot_regime_transitions(
            results_list[0]['telemetry'], 
            results_list[0]['dates'],
            save_path=regime_plot_path
        )
    
    # 4. Generate performance comparison dashboard
    perf_plot_path = os.path.join(output_dir, 'performance_comparison_dashboard.png')
    plot_performance_comparison(results_list, save_path=perf_plot_path)
    
    # 5. Generate text-based performance reports
    report_path = os.path.join(output_dir, 'strategy_performance_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("MAMBA CRYPTOCURRENCY STRATEGY - COMPREHENSIVE PERFORMANCE REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        for results in results_list:
            report_text = generate_performance_report(results)
            f.write(report_text)
            f.write("\n" + "=" * 70 + "\n\n")
    
    print(f"Comprehensive report generated in {output_dir}")
    print(f"- Enhanced NAV curve: {nav_plot_path}")
    print(f"- Position exposure plots: {output_dir}/enhanced_position_plots_*.png")
    print(f"- Regime analysis: {regime_plot_path}")
    print(f"- Performance dashboard: {perf_plot_path}")
    print(f"- Text report: {report_path}")


def generate_quick_summary(results_list: List[Dict]) -> str:
    """Generate a quick summary of strategy performance."""
    summary = "STRATEGY PERFORMANCE SUMMARY\n"
    summary += "=" * 30 + "\n\n"
    
    for results in results_list:
        metrics = calculate_performance_metrics(results['nav'], results['dates'])
        summary += f"{results['strategy_name']}:\n"
        summary += f"  Total Return: {metrics.get('total_return', 0):.2%}\n"
        summary += f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}\n"
        summary += f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
        summary += f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}\n\n"
        
    return summary


def save_results_to_csv(results_list: List[Dict], output_dir: str):
    """Save all results to CSV files for further analysis."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for results in results_list:
        strategy_name = results['strategy_name'].lower().replace(' ', '_')
        
        # Save NAV history
        nav_df = pd.DataFrame({
            'date': results['dates'],
            'nav': results['nav'][1:]  # Skip initial 1.0
        })
        nav_df.to_csv(os.path.join(output_dir, f'{strategy_name}_nav.csv'), index=False)
        
        # Save positions
        positions_df = pd.DataFrame(results['positions'])
        positions_df.to_csv(os.path.join(output_dir, f'{strategy_name}_positions.csv'), index=False)
        
        # Save telemetry
        telemetry_df = pd.DataFrame(results['telemetry'])
        telemetry_df.to_csv(os.path.join(output_dir, f'{strategy_name}_telemetry.csv'), index=False)
        
    print(f"Results saved to CSV files in {output_dir}")


def generate_report(config: Dict, mamba_results: Dict, entropy_results: Dict, prices: Dict = None):
    """Generate final report comparing Mamba and Entropy strategies."""
    results_list = [mamba_results, entropy_results]
    output_dir = config.get('results_dir', 'results')
    
    # Generate comprehensive report
    generate_comprehensive_report(results_list, config, output_dir)
    
    # Print quick summary
    summary = generate_quick_summary(results_list)
    print("\n" + summary)
    
    # Save results to CSV (already done by individual strategy functions, but ensure consistency)
    save_results_to_csv(results_list, output_dir)
    
    return summary