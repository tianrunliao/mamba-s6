import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def build_training_data(telemetry_history, seq_len=30):
    n = len(telemetry_history)
    if n < seq_len + 5:
        return torch.empty(0), torch.empty(0), torch.empty(0)

    X_list, y_cir_list, y_regime_list = [], [], []

    for i in range(seq_len, n - 5):
        seq = []
        for j in range(i - seq_len, i):
            t = telemetry_history[j]
            feat = [
                t.get('factor_skewness', 0.0),
                t.get('factor_kurtosis', 0.0),
                t.get('dispersion', t.get('disp', 1.0)),
                t.get('absorption_ratio', t.get('ar', 0.5)),
                t.get('portfolio_return_5d', 0.0),
                t.get('portfolio_volatility_5d', 0.0),
                t.get('btc_momentum', t.get('btc_mom', 0.0)),
                t.get('n_positions_norm', t.get('n_pos', 0.0) / max(n, 1)),
            ]
            fw_keys = sorted([k for k in t.keys() if k.startswith('feat_fw_')])
            feat.extend([t[k] for k in fw_keys])
            seq.append(feat)

        X_list.append(seq)

        future_rets = []
        for j in range(i, min(i + 5, n)):
            t_future = telemetry_history[j]
            # Use 'portfolio_return_5d' or a generic proxy if we don't have direct asset return
            # Here we just use the precomputed portfolio_return_5d delta as proxy for market movement
            future_rets.append(t_future.get('portfolio_return_5d', 0.0))

        if future_rets:
            mean_ret = np.mean(future_rets)
            var_ret = np.var(future_rets)
            cir = mean_ret / np.sqrt(var_ret + 0.01)
            cir_label = float(2.0 / (1.0 + np.exp(-cir)))
        else:
            cir_label = 1.0
        y_cir_list.append([cir_label])

        t_cur = telemetry_history[i - 1]
        ar = t_cur.get('absorption_ratio', t_cur.get('ar', 0.5))
        vol = np.sqrt(var_ret) if future_rets else 0.0
        abs_mean = abs(mean_ret) if future_rets else 0.0

        if ar > 0.7 and vol > 0.02:
            regime = 2
        elif abs_mean > 0.01 and vol < 0.03:
            regime = 0
        else:
            regime = 1
        y_regime_list.append(regime)

    if not X_list:
        return torch.empty(0), torch.empty(0), torch.empty(0)

    return (
        torch.tensor(X_list, dtype=torch.float32),
        torch.tensor(y_cir_list, dtype=torch.float32),
        torch.tensor(y_regime_list, dtype=torch.long),
    )


def train_mamba_net(net, X, y_cir, y_regime,
                    epochs=100, lr=1e-3, batch_size=64,
                    lambda_regime=0.2, patience=15, val_split=0.15, verbose=False):
    n = X.shape[0]
    if n < 4:
        return net

    n_val = max(1, int(n * val_split))
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_cir_train, y_cir_val = y_cir[:-n_val], y_cir[-n_val:]
    y_regime_train, y_regime_val = y_regime[:-n_val], y_regime[-n_val:]

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    wait = 0
    best_state = None

    net.train()
    for epoch in range(epochs):
        perm = torch.randperm(X_train.shape[0])
        for start in range(0, X_train.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            xb, yc, yr = X_train[idx], y_cir_train[idx], y_regime_train[idx]

            optimizer.zero_grad()
            pred_exp, pred_reg = net(xb)
            loss = (F.mse_loss(pred_exp, yc)
                    + lambda_regime * F.cross_entropy(pred_reg, yr))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        net.eval()
        with torch.no_grad():
            ve, vr = net(X_val)
            vl = F.mse_loss(ve, y_cir_val) + lambda_regime * F.cross_entropy(vr, y_regime_val)
            vl = vl.item()
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
        net.train()

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def train_mamba_model(model, X, y_cir, y_regime, config):
    return train_mamba_net(
        model, X, y_cir, y_regime,
        epochs=config.get('train_epochs', 100),
        lr=config.get('learning_rate', 1e-3),
        batch_size=config.get('batch_size', 64),
        lambda_regime=config.get('lambda_regime', 0.2),
        patience=config.get('patience', 15),
        val_split=config.get('val_split', 0.15),
        verbose=config.get('verbose', False),
    )
