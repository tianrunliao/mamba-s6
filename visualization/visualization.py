"""Research-style reporting for the enhanced strategy suite."""

import os
import tempfile
import warnings

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'mamba_mpl_cache'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText


warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = ['Times New Roman', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 150


STRATEGY_STYLES = {
    'mamba': {'label': 'Mamba LS', 'color': '#1f77b4', 'ls': '-', 'lw': 2.0},
    'baseline': {'label': 'Enhanced Baseline LS', 'color': '#ff7f0e', 'ls': '--', 'lw': 1.6},
    'mamba_equal_weight': {'label': 'Mamba Equal Weight', 'color': '#2ca02c', 'ls': '-.', 'lw': 1.4},
    'equal_weight': {'label': 'Cost-aware Equal Weight', 'color': '#9467bd', 'ls': ':', 'lw': 1.5},
}


def calculate_performance_metrics(returns):
    if returns is None or len(returns) == 0:
        return {}
    returns = pd.Series(returns).fillna(0.0)
    nav = (1.0 + returns).cumprod()
    ann_return = nav.iloc[-1] ** (365 / len(nav)) - 1 if len(nav) > 0 else 0.0
    ann_vol = returns.std() * np.sqrt(365)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    drawdown = nav / np.maximum(nav.cummax(), 1e-8) - 1.0
    return {
        'cumret': float(nav.iloc[-1] - 1.0),
        'ann_return': float(ann_return),
        'ann_vol': float(ann_vol),
        'sharpe': float(sharpe),
        'mdd': float(drawdown.min()),
    }


def _style_axis(axis):
    axis.set_facecolor('white')
    axis.tick_params(colors='black')
    for spine in axis.spines.values():
        spine.set_color('#333333')


def _common_index(results):
    indexes = [df.index for df in results.values() if df is not None and not df.empty]
    if not indexes:
        return pd.Index([])
    common = indexes[0]
    for idx in indexes[1:]:
        common = common.intersection(idx)
    return common


def _load_telem(config, strategy_name, index):
    path = os.path.join(config['results_dir'], f'{strategy_name}_telemetry.csv')
    if not os.path.exists(path):
        return pd.DataFrame(index=index)
    return pd.read_csv(path, parse_dates=['date']).set_index('date').reindex(index)


def _btc_nav(prices, dates):
    if 'BTCUSDT' not in prices:
        return pd.Series(1.0, index=dates)
    btc_prices = prices['BTCUSDT'].reindex(dates, method='ffill')
    nav = (1.0 + btc_prices.pct_change().fillna(0.0)).cumprod()
    return nav / max(nav.iloc[0], 1e-8)


def _save_nav_and_drawdown(results, prices, config):
    dates = _common_index(results)
    if len(dates) == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.patch.set_facecolor('white')
    for axis in axes:
        _style_axis(axis)

    stats_lines = []
    for name, style in STRATEGY_STYLES.items():
        res_df = results.get(name, pd.DataFrame()).reindex(dates)
        if res_df.empty:
            continue
        nav = (1.0 + res_df['ret'].fillna(0.0)).cumprod()
        nav = nav / max(nav.iloc[0], 1e-8)
        dd = nav / nav.cummax() - 1.0
        axes[0].plot(nav.index, nav, color=style['color'], lw=style['lw'], ls=style['ls'], label=style['label'])
        axes[1].fill_between(dd.index, dd * 100.0, 0.0, color=style['color'], alpha=0.22, label=style['label'])
        metrics = calculate_performance_metrics(res_df['ret'])
        stats_lines.append(
            f"{style['label']:<20} {metrics['cumret']:>7.1%}  {metrics['sharpe']:>5.2f}  {metrics['mdd']:>7.1%}"
        )

    btc_nav = _btc_nav(prices, dates)
    axes[0].plot(btc_nav.index, btc_nav, color='#d62728', lw=1.0, ls='--', label='BTC')
    axes[0].set_ylabel('NAV', color='black')
    axes[0].set_title('Unified NAV Comparison', color='black', fontsize=12, pad=10)
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(facecolor='white', edgecolor='#cccccc', labelcolor='black', loc='lower right')

    if stats_lines:
        axes[0].add_artist(AnchoredText(
            "Strategy              CumRet  Sharpe      MDD\n" + "\n".join(stats_lines),
            loc='upper left',
            prop={'size': 8.5, 'family': 'monospace', 'color': 'black'},
            frameon=True,
            borderpad=0.6,
        ))

    axes[1].axhline(0.0, color='#555', lw=0.8)
    axes[1].set_ylabel('Drawdown %', color='black')
    axes[1].legend(facecolor='white', edgecolor='#cccccc', labelcolor='black', ncol=2, loc='lower right')
    axes[1].grid(True, alpha=0.15)
    plt.tight_layout()
    out_path = os.path.join(config['results_dir'], 'enhanced_nav_curve.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Enhanced NAV curve saved to {out_path}')


def _save_regime_report(results, config):
    mamba_res = results.get('mamba', pd.DataFrame())
    if mamba_res.empty:
        return
    telem = _load_telem(config, 'mamba', mamba_res.index).fillna(method='ffill').fillna(0.0)

    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor('white')
    grid = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_exp = fig.add_subplot(grid[0, :])
    ax_ar = fig.add_subplot(grid[1, :])
    ax_pie = fig.add_subplot(grid[2, 0])
    ax_hist = fig.add_subplot(grid[2, 1])
    for axis in [ax_exp, ax_ar, ax_pie, ax_hist]:
        _style_axis(axis)

    ax_exp.plot(telem.index, telem['exp'], color='#2ca02c', lw=0.9, label='Exposure')
    ax_exp.axhline(1.0, color='#888888', lw=0.8, ls='--', alpha=0.6)
    ax_exp.set_ylim(-0.05, max(1.1, float(telem['exp'].max()) + 0.05))
    ax_exp.set_ylabel('Exposure', color='black')
    ax_exp.set_title('Mamba Long-Short Regime Monitoring', color='black', fontsize=11)
    ax_exp.legend(facecolor='white', edgecolor='#cccccc', labelcolor='black')

    if 'absorption_ratio' in telem.columns:
        ax_ar.plot(telem.index, telem['absorption_ratio'], color='#9467bd', lw=1.0, label='Absorption Ratio')
        ax_ar.axhline(0.72, color='#d62728', lw=1.0, ls='--', label='Risk Trigger')
    if 'maker_fill_rate' in telem.columns:
        ax_ar.plot(telem.index, telem['maker_fill_rate'], color='#1f77b4', lw=0.9, alpha=0.75, label='Maker Fill Rate')
    ax_ar.legend(facecolor='white', edgecolor='#cccccc', labelcolor='black')
    ax_ar.set_ylabel('Signal', color='black')

    mode_counts = telem['mode'].fillna('unknown').value_counts()
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    wedges, texts = ax_pie.pie(
        mode_counts.values,
        labels=mode_counts.index,
        colors=colors[:len(mode_counts)],
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor='black', linewidth=2),
    )
    for text in texts:
        text.set_color('black')
        text.set_fontsize(9)
    ax_pie.set_title('Mode Distribution', color='black', fontsize=10)

    modes = telem['mode'].dropna().unique()
    for mode_idx, mode in enumerate(modes):
        mode_data = telem.loc[(telem['exp'] > 0.01) & (telem['mode'] == mode), 'exp']
        if len(mode_data) > 0:
            c = colors[mode_idx % len(colors)]
            ax_hist.hist(mode_data, bins=25, alpha=0.55, color=c, edgecolor='black', label=f'{mode} (avg={mode_data.mean():.2f})')
    ax_hist.legend(facecolor='white', edgecolor='#cccccc', labelcolor='black')
    ax_hist.set_xlabel('Effective Exposure', color='black')
    ax_hist.set_ylabel('Frequency', color='black')
    ax_hist.set_title('Exposure Distribution by Regime', color='black', fontsize=10)

    out_path = os.path.join(config['results_dir'], 'regime_analysis.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Regime analysis saved to {out_path}')


def _save_monthly_report(results, config):
    mamba_res = results.get('mamba', pd.DataFrame())
    if mamba_res.empty:
        return

    fig, (ax_heat, ax_rs) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor('white')
    _style_axis(ax_heat)
    _style_axis(ax_rs)

    monthly = mamba_res['ret'].resample('ME').apply(lambda x: (1.0 + x).prod() - 1.0)
    pivot_df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'ret': monthly.values,
    }).pivot(index='year', columns='month', values='ret')

    norm = TwoSlopeNorm(vmin=-0.2, vcenter=0.0, vmax=0.2)
    image = ax_heat.imshow(pivot_df.values, aspect='auto', cmap='RdYlGn', norm=norm)
    ax_heat.set_xticks(range(12))
    ax_heat.set_xticklabels([str(idx) for idx in range(1, 13)], color='black', fontsize=8)
    ax_heat.set_yticks(range(len(pivot_df.index)))
    ax_heat.set_yticklabels(pivot_df.index.astype(str), color='black', fontsize=8)
    ax_heat.set_title('Mamba Long-Short Monthly Return Heatmap', color='black', fontsize=10)
    for row_idx in range(pivot_df.shape[0]):
        for col_idx in range(pivot_df.shape[1]):
            value = pivot_df.values[row_idx, col_idx]
            if not np.isnan(value):
                ax_heat.text(col_idx, row_idx, f'{value:.1%}', ha='center', va='center', fontsize=6.5, color='black')
    colorbar = plt.colorbar(image, ax=ax_heat, fraction=0.02, pad=0.02)
    colorbar.ax.tick_params(colors='black', labelsize=7)

    def rolling_sharpe(returns, window=90):
        ann = returns.rolling(window).mean() * 365
        vol = returns.rolling(window).std() * np.sqrt(365)
        return ann / (vol + 1e-8)

    for name in ['mamba', 'baseline', 'mamba_equal_weight', 'equal_weight']:
        res_df = results.get(name, pd.DataFrame())
        if res_df.empty:
            continue
        style = STRATEGY_STYLES[name]
        rs = rolling_sharpe(res_df['ret'])
        ax_rs.plot(rs.index, rs, color=style['color'], lw=1.1, ls=style['ls'], label=style['label'])
    ax_rs.axhline(0.0, color='#555', lw=0.7)
    ax_rs.legend(facecolor='white', edgecolor='#cccccc', labelcolor='black', ncol=2)
    ax_rs.set_ylabel('Sharpe', color='black')
    ax_rs.set_title('90-Day Rolling Sharpe', color='black', fontsize=10)
    ax_rs.grid(True, alpha=0.15)

    plt.tight_layout()
    out_path = os.path.join(config['results_dir'], 'monthly_analysis.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Monthly analysis saved to {out_path}')


def _save_position_report(positions, config):
    mamba_pos = positions.get('mamba', pd.DataFrame())
    if mamba_pos.empty:
        return

    weight_df = mamba_pos.copy().reindex(columns=config['symbols']).fillna(0.0)
    fig, axes = plt.subplots(3, 4, figsize=(24, 16))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    colors = sns.color_palette('husl', len(config['symbols']))
    for idx, symbol in enumerate(config['symbols']):
        _style_axis(axes[idx])
        axes[idx].plot(weight_df.index, weight_df[symbol], linewidth=1.4, color=colors[idx])
        axes[idx].set_title(f"Target: {symbol.replace('USDT', '')}", color='black', fontsize=12, fontweight='bold')
        axes[idx].annotate(symbol.replace('USDT', ''), xy=(0.05, 0.85), xycoords='axes fraction', color=colors[idx], fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.15)
        axes[idx].set_xticks([])
    plt.tight_layout()
    out_path = os.path.join(config['results_dir'], 'enhanced_position_plots.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Enhanced position plots saved to {out_path}')


def generate_report(config, results, positions, prices):
    if not results or all(df.empty for df in results.values()):
        print('No results to plot.')
        return

    print('\n=== [4/4] Generating Enhanced Statistical Report ===')
    _save_nav_and_drawdown(results, prices, config)
    _save_regime_report(results, config)
    _save_monthly_report(results, config)
    _save_position_report(positions, config)
