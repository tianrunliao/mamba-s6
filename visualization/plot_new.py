"""
plot_new.py  – Generate paper figures from real backtest results.
All output files carry a _new suffix so nothing in the existing folder is overwritten.

Figures produced:
  nav_curve_new.png        – NAV comparison: Mamba vs Entropy Baseline
  regime_analysis_new.png  – Exposure time-series + AR signal + mode distribution
  monthly_analysis_new.png – Monthly return heatmap and rolling Sharpe

Usage:
  python plot_new.py                          # uses results_integrated/
  python plot_new.py --res-dir path/to/dir   # custom results directory
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import pandas as pd

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'mamba_mpl_cache'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 150

# --- Path resolution ---
parser = argparse.ArgumentParser(description="Generate backtest paper figures")
parser.add_argument('--res-dir', default=None,
                    help='Path to results directory (default: <repo_root>/results_integrated)')
args, _ = parser.parse_known_args()

VIZ_DIR  = os.path.dirname(os.path.abspath(__file__))         # visualization/
ROOT_DIR = os.path.dirname(VIZ_DIR)                           # mamba/
RES_DIR  = args.res_dir or os.path.join(ROOT_DIR, 'results_integrated')
OUT_DIR  = RES_DIR   # save _new files alongside results


# ── Load data ────────────────────────────────────────────────────────────────
mamba   = pd.read_csv(os.path.join(RES_DIR, 'mamba_nav.csv'),    parse_dates=['date']).set_index('date')
entropy = pd.read_csv(os.path.join(RES_DIR, 'baseline_nav.csv'), parse_dates=['date']).set_index('date')
telem   = pd.read_csv(os.path.join(RES_DIR, 'mamba_telemetry.csv'), parse_dates=['date']).set_index('date')

# Align entropy to mamba period and rebase to 1
entropy_al = entropy.loc[mamba.index[0]:].copy()
entropy_al['nav'] = (1 + entropy_al['ret']).cumprod()

# ── Helper: compute metrics ───────────────────────────────────────────────────
def metrics(rets, label):
    nav = (1 + rets).cumprod()
    n_years = len(rets) / 365
    ann_ret = nav.iloc[-1] ** (1 / n_years) - 1
    ann_vol = rets.std() * np.sqrt(365)
    sharpe  = ann_ret / ann_vol
    mdd     = ((nav - nav.cummax()) / nav.cummax()).min()
    return dict(label=label, ann_ret=ann_ret, ann_vol=ann_vol,
                sharpe=sharpe, mdd=mdd, cumret=nav.iloc[-1]-1)


m_met = metrics(mamba['ret'],      'Mamba (Neural Net)')
e_met = metrics(entropy_al['ret'], 'Entropy Baseline')

# ── Figure 1: NAV Comparison ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                         gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
fig.patch.set_facecolor('white')
for ax in axes:
    ax.set_facecolor('white')

ax1, ax2 = axes

# NAV curves
ax1.plot(mamba.index, mamba['nav'],       color='#4fc3f7', lw=1.8, label='Mamba S6 神经网络策略')
ax1.plot(entropy_al.index, entropy_al['nav'], color='#ffb74d', lw=1.4, ls='--', label='信息熵基准策略')

# Shade AR-stop periods
ar_mask = telem['mode'] == 'ar_stop'
ar_starts = ar_mask[ar_mask & ~ar_mask.shift(1).fillna(False)].index
ar_ends   = ar_mask[ar_mask & ~ar_mask.shift(-1).fillna(False)].index
for s, e in zip(ar_starts, ar_ends):
    ax1.axvspan(s, e, alpha=0.18, color='#ef5350', zorder=0)

# Annotations for key events
events = {
    'UST脱锚\n(2022-05)': pd.Timestamp('2022-05-09'),
    'FTX崩溃\n(2022-11)': pd.Timestamp('2022-11-08'),
    'BTC\n突破6万\n(2024-03)': pd.Timestamp('2024-03-20'),
}
nav_interp = mamba['nav']
for label, dt in events.items():
    closest = nav_interp.index[nav_interp.index.get_indexer([dt], method='nearest')[0]]
    y_val = nav_interp[closest]
    ax1.annotate(label, xy=(closest, y_val),
                 xytext=(0, 30), textcoords='offset points',
                 fontsize=7.5, color='black', ha='center',
                 arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8))

ar_patch = mpatches.Patch(color='#ef5350', alpha=0.3, label='AR风险停仓期')
ax1.legend(handles=[*ax1.get_legend_handles_labels()[0], ar_patch],
           fontsize=9, facecolor='white', edgecolor='#cccccc', labelcolor='black', loc='upper left')

ax1.set_ylabel('净值 (NAV)', color='black', fontsize=10)
ax1.tick_params(colors='black')
ax1.spines[:].set_color('#333')
ax1.yaxis.label.set_color('black')
ax1.set_title('图1  净值曲线对比 (2022-04-01 ~ 2026-02-18)', color='black', fontsize=12, pad=10)

# Add stats box
stats_text = (
    f"Mamba:  收益{m_met['cumret']:.1%}  夏普{m_met['sharpe']:.2f}  MDD{m_met['mdd']:.1%}\n"
    f"基准:   收益{e_met['cumret']:.1%}  夏普{e_met['sharpe']:.2f}  MDD{e_met['mdd']:.1%}"
)
ax1.text(0.01, 0.03, stats_text, transform=ax1.transAxes,
         fontsize=8.5, color='black',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

# Drawdown panel
mamba_dd = (mamba['nav'] - mamba['nav'].cummax()) / mamba['nav'].cummax() * 100
entr_dd  = (entropy_al['nav'] - entropy_al['nav'].cummax()) / entropy_al['nav'].cummax() * 100
ax2.fill_between(mamba_dd.index, mamba_dd, 0, alpha=0.7, color='#4fc3f7', label='Mamba')
ax2.fill_between(entr_dd.index,  entr_dd,  0, alpha=0.5, color='#ffb74d', label='基准')
ax2.axhline(0, color='#cccccc', lw=0.5)
ax2.set_ylabel('回撤 (%)', color='black', fontsize=9)
ax2.tick_params(colors='black')
ax2.spines[:].set_color('#333')
ax2.legend(fontsize=8, facecolor='white', edgecolor='#cccccc', labelcolor='black')

plt.tight_layout(rect=[0, 0, 1, 0.97])
out1 = os.path.join(OUT_DIR, 'nav_curve_new.png')
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out1}')


# ── Figure 2: Regime / Exposure Analysis ─────────────────────────────────────
fig2 = plt.figure(figsize=(13, 9))
fig2.patch.set_facecolor('white')
gs = GridSpec(3, 2, figure=fig2, hspace=0.45, wspace=0.4)

# 2a: Exposure time series
ax_exp = fig2.add_subplot(gs[0, :])
ax_exp.set_facecolor('white')
ax_exp.plot(telem.index, telem['exp'], color='#81c784', lw=0.9, label='动态暴露 $E_t$')
ax_exp.axhline(1.0, color='#aaaaaa', lw=0.7, ls='--', alpha=0.6)
ax_stop = telem[telem['mode'] == 'ar_stop']
ax_exp.scatter(ax_stop.index, [0]*len(ax_stop), color='#ef5350', s=4, zorder=5, label='AR停仓')
ax_exp.set_ylim(-0.05, 1.15)
ax_exp.set_ylabel('暴露水平', color='black', fontsize=9)
ax_exp.tick_params(colors='black'); ax_exp.spines[:].set_color('#333')
ax_exp.legend(fontsize=8, facecolor='white', edgecolor='#cccccc', labelcolor='black')
ax_exp.set_title('图2a  Mamba动态暴露与AR风险信号', color='black', fontsize=10)

# 2b: AR time series
ax_ar = fig2.add_subplot(gs[1, :])
ax_ar.set_facecolor('white')
ax_ar.plot(telem.index, telem['absorption_ratio'], color='#ce93d8', lw=0.9, label='吸收比率 AR')
ax_ar.axhline(0.8, color='#ef5350', lw=1.2, ls='--', label='风险阈值 0.8')
ax_ar.fill_between(telem.index, telem['absorption_ratio'], 0.8,
                   where=telem['absorption_ratio'] > 0.8, alpha=0.35, color='#ef5350')
ax_ar.set_ylim(0, 1.05)
ax_ar.set_ylabel('AR 值', color='black', fontsize=9)
ax_ar.tick_params(colors='black'); ax_ar.spines[:].set_color('#333')
ax_ar.legend(fontsize=8, facecolor='white', edgecolor='#cccccc', labelcolor='black')
ax_ar.set_title('图2b  吸收比率 (AR) 时间序列', color='black', fontsize=10)

# 2c: Mode distribution pie
ax_pie = fig2.add_subplot(gs[2, 0])
ax_pie.set_facecolor('white')
mode_counts = telem['mode'].value_counts()
labels_cn = {'mean_reversion': '均值回归\n(65.1%)', 'ar_stop': 'AR风险停仓\n(34.9%)', 'trend': '趋势'}
pie_colors = ['#4fc3f7', '#ef5350', '#81c784']
wedges, texts = ax_pie.pie(
    mode_counts.values,
    labels=[labels_cn.get(m, m) for m in mode_counts.index],
    colors=pie_colors[:len(mode_counts)],
    startangle=90,
    wedgeprops=dict(width=0.55, edgecolor='black', linewidth=2)
)
for t in texts:
    t.set_color('white'); t.set_fontsize(9)
ax_pie.set_title('图2c  市场状态分布', color='black', fontsize=10)

# 2d: Exposure histogram  
ax_hist = fig2.add_subplot(gs[2, 1])
ax_hist.set_facecolor('white')
non_zero_exp = telem.loc[telem['exp'] > 0.01, 'exp']
ax_hist.hist(non_zero_exp, bins=30, color='#4fc3f7', alpha=0.8, edgecolor='black')
ax_hist.axvline(non_zero_exp.mean(), color='#ffb74d', lw=1.5, ls='--',
                label=f'非零均值={non_zero_exp.mean():.3f}')
ax_hist.set_xlabel('暴露水平（非停仓日）', color='black', fontsize=9)
ax_hist.set_ylabel('频次', color='black', fontsize=9)
ax_hist.tick_params(colors='black'); ax_hist.spines[:].set_color('#333')
ax_hist.legend(fontsize=8, facecolor='white', edgecolor='#cccccc', labelcolor='black')
ax_hist.set_title('图2d  有效暴露分布', color='black', fontsize=10)

out2 = os.path.join(OUT_DIR, 'regime_analysis_new.png')
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out2}')


# ── Figure 3: Monthly Return Heatmap + Rolling Sharpe ────────────────────────
fig3, (ax_heat, ax_rs) = plt.subplots(2, 1, figsize=(13, 8),
                                       gridspec_kw={'height_ratios': [2, 1]})
fig3.patch.set_facecolor('white')
ax_heat.set_facecolor('white')
ax_rs.set_facecolor('white')

# Monthly returns
mamba_monthly = mamba['ret'].resample('ME').apply(lambda x: (1+x).prod()-1)
entr_monthly  = entropy_al['ret'].resample('ME').apply(lambda x: (1+x).prod()-1)

pivot_df = pd.DataFrame({
    'year': mamba_monthly.index.year,
    'month': mamba_monthly.index.month,
    'mamba_ret': mamba_monthly.values
}).pivot(index='year', columns='month', values='mamba_ret')

# Color map centered at 0
norm = TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)
im = ax_heat.imshow(pivot_df.values, aspect='auto', cmap='RdYlGn', norm=norm)
ax_heat.set_xticks(range(12))
ax_heat.set_xticklabels(['1月','2月','3月','4月','5月','6月',
                          '7月','8月','9月','10月','11月','12月'],
                         color='black', fontsize=8)
ax_heat.set_yticks(range(len(pivot_df.index)))
ax_heat.set_yticklabels(pivot_df.index.astype(str), color='black', fontsize=8)
ax_heat.set_title('图3a  Mamba策略月度收益热图', color='black', fontsize=10, pad=8)

for i in range(pivot_df.shape[0]):
    for j in range(pivot_df.shape[1]):
        val = pivot_df.values[i, j]
        if not np.isnan(val):
            ax_heat.text(j, i, f'{val:.1%}', ha='center', va='center',
                         fontsize=6.5, color='black' if abs(val) < 0.1 else 'white')
plt.colorbar(im, ax=ax_heat, orientation='vertical', label='月度收益', 
             fraction=0.02, pad=0.02).ax.tick_params(colors='black', labelsize=7)

# Rolling Sharpe (90-day)
def rolling_sharpe(rets, window=90):
    roll_ann = rets.rolling(window).mean() * 365
    roll_vol = rets.rolling(window).std() * np.sqrt(365)
    return roll_ann / (roll_vol + 1e-8)

rs_mamba = rolling_sharpe(mamba['ret'])
rs_entr  = rolling_sharpe(entropy_al['ret'])
ax_rs.plot(rs_mamba.index, rs_mamba, color='#4fc3f7', lw=1.2, label='Mamba 滚动夏普(90日)')
ax_rs.plot(rs_entr.index,  rs_entr,  color='#ffb74d', lw=1.0, ls='--', label='基准 滚动夏普(90日)')
ax_rs.axhline(0, color='#cccccc', lw=0.7)
ax_rs.fill_between(rs_mamba.index, rs_mamba, 0, where=rs_mamba > 0, alpha=0.2, color='#4fc3f7')
ax_rs.fill_between(rs_mamba.index, rs_mamba, 0, where=rs_mamba < 0, alpha=0.2, color='#ef5350')
ax_rs.set_ylabel('夏普比率', color='black', fontsize=9)
ax_rs.tick_params(colors='black'); ax_rs.spines[:].set_color('#333')
ax_rs.legend(fontsize=8, facecolor='white', edgecolor='#cccccc', labelcolor='black')
ax_rs.set_title('图3b  90日滚动夏普比率', color='black', fontsize=10)
fig3.patch.set_facecolor('white')

plt.tight_layout()
out3 = os.path.join(OUT_DIR, 'monthly_analysis_new.png')
plt.savefig(out3, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out3}')

print("\n=== All figures generated successfully ===")
print("Files created:")
for f in [out1, out2, out3]:
    size = os.path.getsize(f) // 1024
    print(f"  {os.path.basename(f)}: {size} KB")
