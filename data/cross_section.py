import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import logging

logger = logging.getLogger(__name__)


def winsorize(series, stds=3):
    mu, sigma = series.mean(), series.std()
    return series.clip(lower=mu - stds * sigma, upper=mu + stds * sigma)


def standardize(series):
    return (series - series.mean()) / (series.std() + 1e-8)


def calc_absorption_ratio(prices_dict, symbols, date, window=20):
    price_list = []
    for sym in symbols:
        if sym in prices_dict and date in prices_dict[sym].index:
            loc = prices_dict[sym].index.get_loc(date)
            if loc >= window:
                ps = prices_dict[sym].iloc[loc - window + 1:loc + 1]
                if len(ps) == window:
                    price_list.append(ps.rename(sym))
    if len(price_list) < 3:
        return 0.5
    price_df = pd.concat(price_list, axis=1).ffill()
    rets = price_df.pct_change().dropna()
    if len(rets) < window * 0.8:
        return 0.5
    eigenvals = np.linalg.eigvalsh(rets.cov())
    if len(eigenvals) == 0 or eigenvals.sum() == 0:
        return 0.5
    return float(eigenvals[-1] / eigenvals.sum())


def compute_weights_and_features(combined_factors, prices_dict, symbols, dir_map):
    all_dates = sorted(combined_factors['date'].unique())
    all_dates = [d for d in all_dates if d >= pd.Timestamp('2021-01-01')]

    daily_results = []
    nav_history = [1.0]
    nav_dates = []

    for date_idx, date in enumerate(all_dates):
        date_data = combined_factors[combined_factors['date'] == date]
        if len(date_data) < 3:
            continue

        symbols_today = date_data['symbol'].tolist()
        factor_cols = [c for c in date_data.columns if c.startswith('factor_')]
        xs_df = date_data.set_index('symbol')[factor_cols].copy()

        for col in factor_cols:
            xs_df[col] = winsorize(xs_df[col])
            xs_df[col] = standardize(xs_df[col])

        xs_filled = xs_df.fillna(0)
        norm_df = (xs_filled - xs_filled.min()) / (xs_filled.max() - xs_filled.min() + 1e-8) + 1e-4
        prob_df = norm_df.div(norm_df.sum(axis=0), axis=1)
        n_sym = len(xs_filled)

        if n_sym <= 1:
            factor_weights = pd.Series(1.0 / len(factor_cols), index=factor_cols)
        else:
            k = 1.0 / np.log(n_sym)
            entropy_vals = -k * (prob_df * np.log(prob_df)).sum(axis=0)
            factor_weights = (1 - entropy_vals)
            factor_weights = factor_weights / factor_weights.sum()

        z_scores = xs_df.fillna(0)
        scores = pd.Series(0.0, index=xs_df.index)
        for col in factor_cols:
            scores += z_scores[col] * dir_map.get(col, 1) * factor_weights[col]

        centered = scores - scores.mean()
        if centered.abs().sum() > 0:
            target_w = centered / centered.abs().sum()
        else:
            target_w = pd.Series(0.0, index=scores.index)

        feat_skew = float(skew(scores))
        feat_kurt = float(kurtosis(scores))
        feat_disp = float(scores.std())
        feat_ar = calc_absorption_ratio(prices_dict, symbols_today, date)

        feat_ret_5d, feat_vol_5d = 0.0, 0.0
        if date_idx >= 5 and nav_dates:
            recent_rets = []
            for i in range(max(0, date_idx - 5), date_idx):
                if i < len(all_dates):
                    pd_ = all_dates[i]
                    if pd_ in nav_dates:
                        idx = nav_dates.index(pd_)
                        if idx > 0:
                            recent_rets.append(nav_history[idx] / nav_history[idx - 1] - 1)
            if recent_rets:
                feat_ret_5d = float(np.mean(recent_rets))
                feat_vol_5d = float(np.std(recent_rets)) if len(recent_rets) > 1 else 0.0

        feat_btc_mom = 0.0
        if 'BTCUSDT' in prices_dict:
            btc = prices_dict['BTCUSDT']
            if date in btc.index:
                loc = btc.index.get_loc(date)
                if loc >= 20:
                    feat_btc_mom = float(np.clip(btc.iloc[loc] / btc.iloc[loc - 20] - 1, -1, 1))

        n_pos = int((target_w.abs() > 0.01).sum())

        row = {
            'date': date,
            'factor_skewness': feat_skew,
            'factor_kurtosis': feat_kurt,
            'dispersion': feat_disp,
            'absorption_ratio': feat_ar,
            'portfolio_return_5d': feat_ret_5d,
            'portfolio_volatility_5d': feat_vol_5d,
            'btc_momentum': feat_btc_mom,
            'n_positions_norm': n_pos / max(len(symbols), 1),
        }

        for col in factor_cols:
            fw = factor_weights[col] if col in factor_weights and not pd.isna(factor_weights[col]) else 0.0
            row[f'feat_fw_{col}'] = float(fw)

        for sym in symbols:
            row[f'weight_{sym}'] = float(target_w[sym]) if sym in target_w.index else 0.0

        daily_results.append(row)

        if not nav_dates:
            nav_dates = [date]
            nav_history = [1.0]
        else:
            port_ret = 0.0
            for sym in target_w.index:
                if sym in prices_dict and date in prices_dict[sym].index:
                    loc = prices_dict[sym].index.get_loc(date)
                    if loc > 0:
                        port_ret += target_w[sym] * (prices_dict[sym].iloc[loc] / prices_dict[sym].iloc[loc - 1] - 1)
            nav_history.append(nav_history[-1] * (1 + port_ret))
            nav_dates.append(date)

    return daily_results


def save_weight_results(daily_results, factors_dir):
    import os
    if not daily_results:
        logger.warning("No cross-sectional results calculated")
        return
    weights_df = pd.DataFrame(daily_results)
    weights_df.to_csv(os.path.join(factors_dir, 'cross_sectional_weights_and_features.csv'), index=False)

    weight_cols = [c for c in weights_df.columns if c.startswith('weight_')]
    weights_df[['date'] + weight_cols].to_csv(os.path.join(factors_dir, 'daily_weights.csv'), index=False)
