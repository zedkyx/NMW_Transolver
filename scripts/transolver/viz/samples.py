import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Tuple, Dict, Optional

from scripts.transolver.viz.style import apply_paper_rcparams, save_paper_fig, style_colorbar

apply_paper_rcparams()


def _robust_range(a: np.ndarray, b: np.ndarray, qlo=0.0, qhi=100.0):
    """
    Get a robust range for colorbars.
    Default qlo=0.0, qhi=100.0 as requested (no trimming unless necessary).
    """
    x = np.concatenate([a, b], axis=0)
    if qlo > 0 or qhi < 100:
        vmin, vmax = np.percentile(x, [qlo, qhi])
    else:
        vmin = float(np.nanmin(x))
        vmax = float(np.nanmax(x))
        
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(x))
        vmax = float(np.nanmax(x))
        if vmin == vmax:
            vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def _scatter_ax(ax, xy, values, cbar_label, cmap, vmin=None, vmax=None, s=4, extent_xy=None):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=s, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8)
    if extent_xy is not None and len(extent_xy) > 0:
        ex = np.asarray(extent_xy[:, 0], dtype=float)
        ey = np.asarray(extent_xy[:, 1], dtype=float)
        xm, xM = float(np.nanmin(ex)), float(np.nanmax(ex))
        ym, yM = float(np.nanmin(ey)), float(np.nanmax(ey))
        rx, ry = xM - xm, yM - ym
        pad_x = max(rx * 0.02, 1e-9)
        pad_y = max(ry * 0.02, 1e-9)
        ax.set_xlim(xm - pad_x, xM + pad_x)
        ax.set_ylim(ym - pad_y, yM + pad_y)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    style_colorbar(cbar, label=cbar_label)


def plot_samples(model, loader, y_normalizer, device, save_dir, prefix="sample", num_samples=3):
    """
    Standard interface for plotting samples during training or evaluation.
    Handles different batch formats (with/without mask, kNN, SATO indices).
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Try to get channel names from the loader's dataset if possible
    channel_names = ["Mises"] # Default
    
    samples_count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # Deterministic unpacking based on batch length
            # This follows the logic in loops.py
            pos_b = batch[0].to(device)
            fx_b = batch[1].to(device)
            y_b = batch[2].to(device)
            
            mask_b = None
            knn_idx = None
            knn_valid = None
            sato_indices = None
            
            # Flexible unpacking (matching keti_trainer.py and loops.py logic)
            if len(batch) == 8: # pos, fx, y, mask, knn_idx, knn_valid, order, inverse
                mask_b = batch[3].to(device)
                knn_idx = batch[4].to(device)
                knn_valid = batch[5].to(device)
                sato_indices = {"order": batch[6].to(device), "inverse": batch[7].to(device)}
            elif len(batch) == 7: # pos, fx, y, knn_idx, knn_valid, order, inverse
                knn_idx = batch[3].to(device)
                knn_valid = batch[4].to(device)
                sato_indices = {"order": batch[5].to(device), "inverse": batch[6].to(device)}
            elif len(batch) == 6: # pos, fx, y, mask, knn_idx, knn_valid
                mask_b = batch[3].to(device)
                knn_idx = batch[4].to(device)
                knn_valid = batch[5].to(device)
            elif len(batch) == 5: # pos, fx, y, knn_idx, knn_valid
                knn_idx = batch[3].to(device)
                knn_valid = batch[4].to(device)
            elif len(batch) == 4: # pos, fx, y, mask
                mask_b = batch[3].to(device)
            
            # Forward pass with all available info
            out_n = model(pos_b, fx=fx_b, mask=mask_b, knn_idx=knn_idx, knn_valid=knn_valid, sato_indices=sato_indices)
            
            pred = y_normalizer.decode(out_n).cpu().numpy()
            true = y_normalizer.decode(y_b).cpu().numpy()
            pos = pos_b.cpu().numpy()
            
            for j in range(pred.shape[0]):
                case_id = samples_count
                plot_combined_channels_atmospheric(
                    pos[j], pred[j], true[j], 
                    save_dir, prefix, case_id, channel_names
                )
                samples_count += 1
                if samples_count >= num_samples:
                    break
            
            if samples_count >= num_samples:
                break
    
    print(f"[+] Saved {samples_count} sample plots to {save_dir}")


def plot_combined_channels_atmospheric(pos_np, pred_np, true_np, save_dir, prefix, case_id, channel_names, padding_mask=None):
    """
    Atmospheric plot: Prediction, Ground Truth, Relative Error (Case), Absolute Error.
    过滤掉坐标为 (0,0) 的点（padding 点）。
    不绘制整图大标题（fig.suptitle），子图标题保留，便于论文排版。
    """
    num_channels = pred_np.shape[1]
    eps = 1e-12
    
    if padding_mask is not None:
        valid_mask = padding_mask.astype(bool)
    else:
        # Backward-compatible fallback for old caches that used (0,0) padding.
        valid_mask = ~((pos_np[:, 0] == 0) & (pos_np[:, 1] == 0))
    pos_valid = pos_np[valid_mask]
    pred_valid = pred_np[valid_mask]
    true_valid = true_np[valid_mask]
    
    fig, axes = plt.subplots(num_channels, 4, figsize=(20, 5 * num_channels), constrained_layout=True)
    if num_channels == 1:
        axes = axes.reshape(1, 4)

    for c in range(num_channels):
        y_true = true_valid[:, c]
        y_pred = pred_valid[:, c]
        col_name = channel_names[c] if c < len(channel_names) else f"Ch{c}"
        
        err = np.abs(y_pred - y_true)
        denom_case = float(np.max(np.abs(y_true)) + eps)
        rel_case = err / denom_case
        max_rel_case = np.max(rel_case)

        # Full range as per user rule
        vmin, vmax = _robust_range(y_true, y_pred, qlo=0.0, qhi=100.0)
        rc_vmin, rc_vmax = _robust_range(rel_case, rel_case, qlo=0.0, qhi=100.0)
        e_vmin, e_vmax = _robust_range(err, err, qlo=0.0, qhi=100.0)

        _scatter_ax(axes[c, 0], pos_valid, y_pred, f'{col_name} Prediction', 'coolwarm', vmin, vmax)
        _scatter_ax(axes[c, 1], pos_valid, y_true, f'{col_name} Ground Truth', 'coolwarm', vmin, vmax)
        _scatter_ax(axes[c, 2], pos_valid, rel_case, f'{col_name} RelErr (Case) Max={max_rel_case:.4f}', 'coolwarm', rc_vmin, rc_vmax)
        _scatter_ax(axes[c, 3], pos_valid, err, f'{col_name} AbsErr', 'coolwarm', e_vmin, e_vmax)

    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f'{prefix}_case{case_id}_combined.png')
    save_paper_fig(fig, fig_path, caption=f'{prefix} case{case_id} combined channels')
    plt.close(fig)


def plot_violin_distribution(metrics_df, save_dir, filename="error_distribution_violin.png"):
    """
    Draw violin plots for error distributions (L2 and Linf).
    Expects metrics_df to have columns: ['Case', 'Channel', 'Metric', 'Value']
    """
    sns.set_theme(style="whitegrid")
    
    # Create two subplots: one for Relative L2, one for Relative Linf
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    l2_df = metrics_df[metrics_df['Metric'] == 'RelL2']
    linf_df = metrics_df[metrics_df['Metric'] == 'RelLinf']
    
    if not l2_df.empty:
        sns.violinplot(data=l2_df, x='Channel', y='Value', hue='Channel', ax=ax1, inner="box", palette="muted", legend=False)
        ax1.set_ylabel('Relative L2 Error')
        ax1.set_xlabel('Channel')
        
    if not linf_df.empty:
        sns.violinplot(data=linf_df, x='Channel', y='Value', hue='Channel', ax=ax2, inner="box", palette="muted", legend=False)
        ax2.set_ylabel('Relative L_inf Error')
        ax2.set_xlabel('Channel')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_paper_fig(fig, os.path.join(save_dir, filename), caption="Relative L2 / L_inf error distribution (violin)")
    plt.close()


def plot_box_distribution(metrics_df, save_dir, filename="error_distribution_boxplot.png"):
    """
    Draw box plots for error distributions (RelL2 and RelLinf).
    Expects metrics_df columns: ['Case', 'Channel', 'Metric', 'Value']
    """
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    l2_df = metrics_df[metrics_df["Metric"] == "RelL2"]
    linf_df = metrics_df[metrics_df["Metric"] == "RelLinf"]

    if not l2_df.empty:
        sns.boxplot(data=l2_df, x="Channel", y="Value", hue="Channel", ax=ax1, palette="muted", legend=False)
        ax1.set_ylabel("Relative L2 Error")
        ax1.set_xlabel("Channel")

    if not linf_df.empty:
        sns.boxplot(data=linf_df, x="Channel", y="Value", hue="Channel", ax=ax2, palette="muted", legend=False)
        ax2.set_ylabel("Relative L_inf Error")
        ax2.set_xlabel("Channel")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_paper_fig(fig, os.path.join(save_dir, filename), caption="Relative L2 / L_inf error distribution (boxplot)")
    plt.close()


def plot_case_grid(
    pos_list,
    pred_list,
    true_list,
    case_ids,
    metrics_list,
    channel_names,
    save_dir,
    filename="extreme_cases_grid.png",
    title="Extreme Cases Analysis",
    padding_mask_list=None,
):
    """
    Plot 6 cases in a 2x3 grid, showing the first channel's prediction or a combined error.
    过滤掉坐标为 (0,0) 的点（padding 点）。
    title 参数保留兼容调用，不再绘制整图大标题（论文出图约定）。
    """
    num_cases = len(pos_list)
    rows = (num_cases + 2) // 3
    cols = min(num_cases, 3)
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), constrained_layout=True)
    if num_cases == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.atleast_1d(axes).ravel()
    
    for i in range(num_cases):
        pos = pos_list[i]
        pred = pred_list[i][:, 0] # Default to first channel
        true = true_list[i][:, 0]
        case_id = case_ids[i]
        metric = metrics_list[i]
        
        if padding_mask_list is not None:
            valid_mask = padding_mask_list[i].astype(bool)
        else:
            # Backward-compatible fallback for old caches that used (0,0) padding.
            valid_mask = ~((pos[:, 0] == 0) & (pos[:, 1] == 0))
        pos_valid = pos[valid_mask]
        pred_valid = pred[valid_mask]
        true_valid = true[valid_mask]
        
        err = np.abs(pred_valid - true_valid)
        denom = np.max(np.abs(true_valid)) + 1e-12
        rel_err = err / denom
        
        # Display Relative Error map for each case in the grid
        vmin, vmax = _robust_range(rel_err, rel_err, qlo=0.0, qhi=100.0)
        sc = axes_flat[i].scatter(pos_valid[:, 0], pos_valid[:, 1], c=rel_err, s=4, cmap='coolwarm', vmin=vmin, vmax=vmax, alpha=0.8)
        axes_flat[i].set_aspect('equal', 'box')
        axes_flat[i].set_xlabel('X (m)')
        axes_flat[i].set_ylabel('Y (m)')
        cbar = plt.colorbar(sc, ax=axes_flat[i], fraction=0.046, pad=0.04)
        style_colorbar(cbar, label=f"Case {case_id} RelErr RelL2={metric:.4f}")
        
    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
        
    os.makedirs(save_dir, exist_ok=True)
    save_paper_fig(fig, os.path.join(save_dir, filename), caption=str(title))
    plt.close()


def plot_peak_scatter(true_peaks, pred_peaks, save_dir, prefix, channel_name):
    """
    True Peaks vs Predicted Peaks scatter plot.
    """
    true_peaks = np.array(true_peaks)
    pred_peaks = np.array(pred_peaks)
    
    val_min = min(np.min(true_peaks), np.min(pred_peaks))
    val_max = max(np.max(true_peaks), np.max(pred_peaks))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([val_min, val_max], [val_min, val_max], 'k--', alpha=0.5, label='Ideal (y=x)', zorder=1)
    
    x_range = np.linspace(val_min, val_max, 100)
    ax.plot(x_range, x_range * 1.05, 'r:', alpha=0.4, label='±5% Error', zorder=1)
    ax.plot(x_range, x_range * 0.95, 'r:', alpha=0.4, zorder=1)
    
    ax.scatter(true_peaks, pred_peaks, color='royalblue', alpha=0.6, edgecolors='w', s=40, zorder=2)
    
    ax.set_xlabel(f'Ground Truth Peak ({channel_name})')
    ax.set_ylabel(f'Predicted Peak ({channel_name})')
    
    ax.set_xlim(val_min, val_max)
    ax.set_ylim(val_min, val_max)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{prefix}_peak_scatter_{channel_name}.png")
    save_paper_fig(fig, save_path, caption=f'Peak Comparison: {channel_name}')
    plt.close()


def compute_sample_rel_l2(
    all_preds: np.ndarray,
    all_trues: np.ndarray,
    padding_mask: Optional[np.ndarray] = None,
    node_weight: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    逐样本、逐通道相对 L2，与 eval_keti / 训练 loss 定义一致。

    all_preds/all_trues: [N, P, C]
    Returns:
        rel_ch: [N, C]
        rel_mean: [N]  通道平均
    """
    num_samples, _, num_channels = all_preds.shape
    rel_ch = np.zeros((num_samples, num_channels), dtype=np.float64)
    for i in range(num_samples):
        if padding_mask is not None:
            valid = padding_mask[i].astype(bool)
            y_p = all_preds[i, valid]
            y_t = all_trues[i, valid]
            q_i = None if node_weight is None else np.asarray(node_weight[i, valid], dtype=np.float64)
        else:
            y_p = all_preds[i]
            y_t = all_trues[i]
            q_i = None if node_weight is None else np.asarray(node_weight[i], dtype=np.float64)
        for c in range(num_channels):
            diff = y_p[:, c] - y_t[:, c]
            yt = y_t[:, c]
            if q_i is None:
                rel_ch[i, c] = np.linalg.norm(diff) / (np.linalg.norm(yt) + eps)
            else:
                num = np.sum(q_i * diff**2)
                den = np.sum(q_i * yt**2)
                rel_ch[i, c] = np.sqrt(num + eps) / (np.sqrt(den + eps))
    rel_mean = rel_ch.mean(axis=1)
    return rel_ch, rel_mean


def plot_rel_vs_time_scatter(
    t: np.ndarray,
    rel_mean: np.ndarray,
    save_dir: str,
    filename: str,
    title: str,
    split: Optional[np.ndarray] = None,
) -> str:
    """
    RelL2 随物理时间 t 的散点图。
    split 为 None 时单色；否则按 train/test 分色。
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    t = np.asarray(t, dtype=np.float64)
    rel_mean = np.asarray(rel_mean, dtype=np.float64)

    if split is None:
        ax.scatter(t, rel_mean, s=10, alpha=0.55, c="steelblue", edgecolors="none", label=f"n={len(t)}")
        ax.legend(loc="upper right", fontsize=10)
    else:
        split = np.asarray(split)
        style = [("train", "steelblue"), ("test", "crimson")]
        for label, color in style:
            mask = split == label
            if not np.any(mask):
                continue
            ax.scatter(
                t[mask],
                rel_mean[mask],
                s=10,
                alpha=0.45,
                c=color,
                edgecolors="none",
                label=f"{label} (n={mask.sum()})",
            )
        ax.legend(loc="upper right", fontsize=10)

    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("RelL2 (channel mean)")
    ax.grid(True, linestyle=":", alpha=0.5)

    save_path = os.path.join(save_dir, filename)
    save_paper_fig(fig, save_path, caption=title)
    plt.close(fig)
    return save_path


# Kept for backward compatibility if needed, but not actively used in refactored eval_keti.py
def plot_combined_channels_pointwise_rel(pos_np, pred_np, true_np, save_dir, prefix, case_id, channel_names):
    """
    过滤掉坐标为 (0,0) 的点（padding 点）。
    """
    num_channels = pred_np.shape[1]
    eps = 1e-12
    
    # 过滤掉 (0,0) 的点（padding 点）
    valid_mask = ~((pos_np[:, 0] == 0) & (pos_np[:, 1] == 0))
    pos_valid = pos_np[valid_mask]
    pred_valid = pred_np[valid_mask]
    true_valid = true_np[valid_mask]
    
    fig, axes = plt.subplots(num_channels, 4, figsize=(22, 6 * num_channels), constrained_layout=True)
    if num_channels == 1: axes = axes.reshape(1, 4)
    for c in range(num_channels):
        y_true, y_pred = true_valid[:, c], pred_valid[:, c]
        col_name = channel_names[c] if c < len(channel_names) else f"Ch{c}"
        err = np.abs(y_pred - y_true)
        rel_point = err / (np.abs(y_true) + eps)
        max_rel_point = np.max(rel_point)
        vmin, vmax = _robust_range(y_true, y_pred, qlo=1.0, qhi=99.0)
        rp_vmin, rp_vmax = _robust_range(rel_point, rel_point, qlo=1.0, qhi=99.0)
        e_vmin, e_vmax = _robust_range(err, err, qlo=1.0, qhi=99.0)
        _scatter_ax(axes[c, 0], pos_valid, y_pred, f'{col_name} Prediction', 'coolwarm', vmin, vmax)
        _scatter_ax(axes[c, 1], pos_valid, y_true, f'{col_name} Ground Truth', 'coolwarm', vmin, vmax)
        _scatter_ax(axes[c, 2], pos_valid, rel_point, f'{col_name} RelErr (Point)\nMax={max_rel_point:.4f}', 'coolwarm', rp_vmin, rp_vmax)
        _scatter_ax(axes[c, 3], pos_valid, err, f'{col_name} AbsErr', 'coolwarm', e_vmin, e_vmax)
    os.makedirs(save_dir, exist_ok=True)
    save_paper_fig(plt.gcf(), os.path.join(save_dir, f'{prefix}_case{case_id}_pointwise_rel.png'))
    plt.close(fig)

def plot_combined_channels_pointwise_rel_raw(pos_np, pred_np, true_np, save_dir, prefix, case_id, channel_names):
    """
    过滤掉坐标为 (0,0) 的点（padding 点）。
    """
    num_channels = pred_np.shape[1]
    eps = 1e-12
    
    # 过滤掉 (0,0) 的点（padding 点）
    valid_mask = ~((pos_np[:, 0] == 0) & (pos_np[:, 1] == 0))
    pos_valid = pos_np[valid_mask]
    pred_valid = pred_np[valid_mask]
    true_valid = true_np[valid_mask]
    
    fig, axes = plt.subplots(num_channels, 4, figsize=(22, 6 * num_channels), constrained_layout=True)
    if num_channels == 1: axes = axes.reshape(1, 4)
    for c in range(num_channels):
        y_true, y_pred = true_valid[:, c], pred_valid[:, c]
        col_name = channel_names[c] if c < len(channel_names) else f"Ch{c}"
        err = np.abs(y_pred - y_true)
        rel_point = err / (np.abs(y_true) + eps)
        max_rel_point = np.max(rel_point)
        vmin, vmax = _robust_range(y_true, y_pred, qlo=0.0, qhi=100.0)
        rp_vmin, rp_vmax = _robust_range(rel_point, rel_point, qlo=0.0, qhi=100.0)
        e_vmin, e_vmax = _robust_range(err, err, qlo=0.0, qhi=100.0)
        _scatter_ax(axes[c, 0], pos_valid, y_pred, f'{col_name} Prediction', 'coolwarm', vmin, vmax)
        _scatter_ax(axes[c, 1], pos_valid, y_true, f'{col_name} Ground Truth', 'coolwarm', vmin, vmax)
        _scatter_ax(axes[c, 2], pos_valid, rel_point, f'{col_name} RelErr (Point, Raw)\nMax={max_rel_point:.4f}', 'coolwarm', rp_vmin, rp_vmax)
        _scatter_ax(axes[c, 3], pos_valid, err, f'{col_name} AbsErr', 'coolwarm', e_vmin, e_vmax)
    os.makedirs(save_dir, exist_ok=True)
    save_paper_fig(plt.gcf(), os.path.join(save_dir, f'{prefix}_case{case_id}_pointwise_rel_raw.png'))
    plt.close(fig)

def plot_combined_channels_pointwise_rel_trimmed(pos_np, pred_np, true_np, save_dir, prefix, case_id, channel_names):
    """
    过滤掉坐标为 (0,0) 的点（padding 点），然后再进行百分位裁剪。
    """
    num_channels = pred_np.shape[1]
    eps = 1e-12
    
    # 先过滤掉 (0,0) 的点（padding 点）
    valid_mask_pad = ~((pos_np[:, 0] == 0) & (pos_np[:, 1] == 0))
    pos_pad_f = pos_np[valid_mask_pad]
    pred_pad_f = pred_np[valid_mask_pad]
    true_pad_f = true_np[valid_mask_pad]
    
    # 再进行百分位裁剪
    mises_idx = 0
    for i, name in enumerate(channel_names):
        if name.lower() == "mises": mises_idx = i; break
    m_true = true_pad_f[:, mises_idx]
    p1, p99 = np.percentile(m_true, 1.0), np.percentile(m_true, 99.0)
    mask = (m_true >= p1) & (m_true <= p99)
    p_f, pred_f, t_f = pos_pad_f[mask], pred_pad_f[mask], true_pad_f[mask]
    
    fig, axes = plt.subplots(num_channels, 4, figsize=(22, 6 * num_channels), constrained_layout=True)
    if num_channels == 1: axes = axes.reshape(1, 4)
    for c in range(num_channels):
        y_true, y_pred = t_f[:, c], pred_f[:, c]
        col_name = channel_names[c] if c < len(channel_names) else f"Ch{c}"
        err = np.abs(y_pred - y_true)
        rel_point = err / (np.abs(y_true) + eps)
        max_rel_point = np.max(rel_point)
        vmin, vmax = _robust_range(y_true, y_pred, qlo=0.0, qhi=100.0)
        rp_vmin, rp_vmax = _robust_range(rel_point, rel_point, qlo=0.0, qhi=100.0)
        e_vmin, e_vmax = _robust_range(err, err, qlo=0.0, qhi=100.0)
        _scatter_ax(axes[c, 0], p_f, y_pred, f'{col_name} Prediction', 'coolwarm', vmin, vmax)
        _scatter_ax(axes[c, 1], p_f, y_true, f'{col_name} Ground Truth', 'coolwarm', vmin, vmax)
        _scatter_ax(axes[c, 2], p_f, rel_point, f'{col_name} RelErr (Point)\nMax={max_rel_point:.4f}', 'coolwarm', rp_vmin, rp_vmax)
        _scatter_ax(axes[c, 3], p_f, err, f'{col_name} AbsErr', 'coolwarm', e_vmin, e_vmax)
    os.makedirs(save_dir, exist_ok=True)
    save_paper_fig(plt.gcf(), os.path.join(save_dir, f'{prefix}_case{case_id}_pointwise_rel_trimmed.png'))
    plt.close(fig)


def resolve_rel_err_denoms(
    channel_names: List[str],
    override: Optional[str] = None,
) -> Tuple[str, ...]:
    """
    相对误差分母模式：
      - local: |e| / (|true| + eps)
      - max:   |e| / (case内该通道 max(|true|) + eps)
    默认：位移通道只统计 max；应力通道同时统计 local + max。
    """
    if override is not None and str(override).strip():
        dens = tuple(x.strip().lower() for x in str(override).split(",") if x.strip())
        for d in dens:
            if d not in ("local", "max"):
                raise ValueError(f"unsupported rel_err_denom={d!r}, expect local|max")
        if not dens:
            raise ValueError("rel_err_denoms is empty")
        return dens

    names = [str(n).strip().lower() for n in channel_names]
    is_disp = any(
        n in ("u1", "u2", "u3", "ux", "uy", "uz")
        or (len(n) >= 2 and n[0] == "u" and n[1:].isdigit())
        for n in names
    )
    if is_disp:
        return ("max",)
    return ("local", "max")


def _finalize_error_stats(
    error_values: List[float],
    threshold_counts: Dict[float, int],
    total_high_stress: int,
    error_thresholds: List[float],
) -> Dict:
    if total_high_stress <= 0 or len(error_values) == 0:
        out = {
            "mean_error": 0.0,
            "max_error": 0.0,
            "median_error": 0.0,
            "p95_error": 0.0,
            "p99_error": 0.0,
            "threshold_stats": {th: {"count": 0, "percentage": 0.0} for th in error_thresholds},
        }
        return out
    error_array = np.asarray(error_values, dtype=np.float64)
    out = {
        "mean_error": float(np.mean(error_array)),
        "max_error": float(np.max(error_array)),
        "median_error": float(np.median(error_array)),
        "p95_error": float(np.percentile(error_array, 95)),
        "p99_error": float(np.percentile(error_array, 99)),
        "threshold_stats": {},
    }
    for th in error_thresholds:
        count = int(threshold_counts.get(th, 0))
        out["threshold_stats"][th] = {
            "count": count,
            "percentage": 100.0 * count / total_high_stress,
        }
    return out


def analyze_high_stress_region_error(
    all_pos_list: List[np.ndarray],
    all_pred_list: List[np.ndarray],
    all_true_list: List[np.ndarray],
    channel_names: List[str],
    stress_channel_idx: int = 0,
    lower_ratio: float = 2 / 3,
    upper_ratio: float = 1.0,
    error_thresholds: List[float] = None,
    padding_mask_list: List[np.ndarray] = None,
    per_channel: bool = False,
    relative_denoms: Tuple[str, ...] = None,
    node_weight_list: List[np.ndarray] = None,
) -> Dict:
    """
    分析所有样本的高应力/高响应区域误差统计。

    relative_denoms:
      - "local": 除当地 |true|
      - "max":   除该 case 筛选通道的 max(|true|)（与划高值区同一 max）

    node_weight_list:
      可选求积权重 q（与 pos 同长）。区域仍按应力阈值选点；
      另算 RelL2_hs_q = sqrt(sum q e^2)/sqrt(sum q y^2)，
      lt*_pct_q = sum_{rel<th} q / sum_{HS} q。
    """
    if error_thresholds is None:
        error_thresholds = [0.01, 0.05, 0.10, 0.20]
    if relative_denoms is None:
        relative_denoms = resolve_rel_err_denoms(channel_names)
    relative_denoms = tuple(relative_denoms)
    if not relative_denoms:
        raise ValueError("relative_denoms must be non-empty")

    num_cases = len(all_pos_list)
    num_channels = all_pred_list[0].shape[1]
    eps = 1e-12
    primary_denom = relative_denoms[0]
    use_q = node_weight_list is not None

    stats = {}
    for c in range(num_channels):
        denoms = {}
        for d in relative_denoms:
            denoms[d] = {
                "error_values": [],
                "threshold_stats": {th: 0 for th in error_thresholds},
                "threshold_mass": {th: 0.0 for th in error_thresholds},
                "total_mass": 0.0,
            }
        stats[c] = {
            "channel_name": channel_names[c] if c < len(channel_names) else f"Ch{c}",
            "total_points": 0,
            "high_stress_points": 0,
            "relative_denoms": list(relative_denoms),
            "primary_denom": primary_denom,
            "denoms": denoms,
            # 兼容旧字段：指向 primary_denom
            "error_values": denoms[primary_denom]["error_values"],
            "threshold_stats": denoms[primary_denom]["threshold_stats"],
            # 高应力区 RelL2（无求积）：逐 case 的 ||e||2/||y||2，再 case-mean；另给 pooled
            "rel_l2_pointwise_case": [],
            "rel_l2_sse": 0.0,
            "rel_l2_ssy": 0.0,
            # 高应力区 RelL2（求积 q）
            "rel_l2_quad_case": [],
            "rel_l2_quad_sse": 0.0,
            "rel_l2_quad_ssy": 0.0,
            "has_quadrature": use_q,
        }

    for case_idx in range(num_cases):
        pos = all_pos_list[case_idx]
        pred = all_pred_list[case_idx]
        true = all_true_list[case_idx]

        if padding_mask_list is not None:
            valid_mask = padding_mask_list[case_idx].astype(bool)
        else:
            valid_mask = ~((pos[:, 0] == 0) & (pos[:, 1] == 0))

        pos_valid = pos[valid_mask]
        pred_valid = pred[valid_mask]
        true_valid = true[valid_mask]
        q_valid = None
        if use_q:
            q_case = np.asarray(node_weight_list[case_idx], dtype=np.float64).reshape(-1)
            if q_case.shape[0] != pos.shape[0]:
                raise ValueError(
                    f"node_weight length {q_case.shape[0]} != pos N={pos.shape[0]} "
                    f"(case {case_idx})"
                )
            q_valid = q_case[valid_mask]

        for c in range(num_channels):
            if per_channel:
                stress_values = np.abs(true_valid[:, c])
            else:
                stress_values = np.abs(true_valid[:, stress_channel_idx])

            max_stress = float(np.max(stress_values)) if len(stress_values) else 0.0
            lower_bound = lower_ratio * max_stress
            upper_bound = upper_ratio * max_stress

            high_stress_mask = (stress_values >= lower_bound) & (stress_values <= upper_bound)
            high_stress_pred = pred_valid[high_stress_mask, c]
            high_stress_true = true_valid[high_stress_mask, c]
            q_hs = q_valid[high_stress_mask] if q_valid is not None else None

            err = np.abs(high_stress_pred - high_stress_true)
            stats[c]["total_points"] += len(pos_valid)
            n_hs = int(high_stress_mask.sum())
            stats[c]["high_stress_points"] += n_hs

            if n_hs > 0:
                e = high_stress_pred.astype(np.float64) - high_stress_true.astype(np.float64)
                y = high_stress_true.astype(np.float64)
                sse = float(np.sum(e * e))
                ssy = float(np.sum(y * y))
                if ssy > eps:
                    stats[c]["rel_l2_pointwise_case"].append(float(np.sqrt(sse / ssy)))
                    stats[c]["rel_l2_sse"] += sse
                    stats[c]["rel_l2_ssy"] += ssy
                if q_hs is not None:
                    qe = q_hs * (e * e)
                    qy = q_hs * (y * y)
                    sse_q = float(np.sum(qe))
                    ssy_q = float(np.sum(qy))
                    if ssy_q > eps:
                        stats[c]["rel_l2_quad_case"].append(float(np.sqrt(sse_q / ssy_q)))
                        stats[c]["rel_l2_quad_sse"] += sse_q
                        stats[c]["rel_l2_quad_ssy"] += ssy_q

            for denom_name in relative_denoms:
                if denom_name == "local":
                    rel_err = err / (np.abs(high_stress_true) + eps)
                elif denom_name == "max":
                    rel_err = err / (max_stress + eps)
                else:
                    raise ValueError(f"unsupported relative denom: {denom_name}")

                dstat = stats[c]["denoms"][denom_name]
                dstat["error_values"].extend(rel_err.tolist())
                for th in error_thresholds:
                    dstat["threshold_stats"][th] += int(np.sum(rel_err < th))
                if q_hs is not None and n_hs > 0:
                    mass = float(np.sum(q_hs))
                    dstat["total_mass"] += mass
                    for th in error_thresholds:
                        dstat["threshold_mass"][th] += float(np.sum(q_hs[rel_err < th]))

    for c in range(num_channels):
        total_high_stress = stats[c]["high_stress_points"]
        case_vals = stats[c]["rel_l2_pointwise_case"]
        if case_vals:
            stats[c]["rel_l2_pointwise_mean"] = float(np.mean(case_vals))
            stats[c]["rel_l2_pointwise_median"] = float(np.median(case_vals))
        else:
            stats[c]["rel_l2_pointwise_mean"] = float("nan")
            stats[c]["rel_l2_pointwise_median"] = float("nan")
        if stats[c]["rel_l2_ssy"] > eps:
            stats[c]["rel_l2_pointwise_pooled"] = float(
                np.sqrt(stats[c]["rel_l2_sse"] / stats[c]["rel_l2_ssy"])
            )
        else:
            stats[c]["rel_l2_pointwise_pooled"] = float("nan")

        case_q = stats[c]["rel_l2_quad_case"]
        if case_q:
            stats[c]["rel_l2_quad_mean"] = float(np.mean(case_q))
            stats[c]["rel_l2_quad_median"] = float(np.median(case_q))
        else:
            stats[c]["rel_l2_quad_mean"] = float("nan")
            stats[c]["rel_l2_quad_median"] = float("nan")
        if stats[c]["rel_l2_quad_ssy"] > eps:
            stats[c]["rel_l2_quad_pooled"] = float(
                np.sqrt(stats[c]["rel_l2_quad_sse"] / stats[c]["rel_l2_quad_ssy"])
            )
        else:
            stats[c]["rel_l2_quad_pooled"] = float("nan")

        for denom_name in relative_denoms:
            dstat = stats[c]["denoms"][denom_name]
            finalized = _finalize_error_stats(
                dstat["error_values"],
                dstat["threshold_stats"],
                total_high_stress,
                error_thresholds,
            )
            dstat.update(finalized)
            # q-weighted threshold percentages (mass fraction)
            total_mass = float(dstat.get("total_mass", 0.0))
            th_q = {}
            for th in error_thresholds:
                mass_ok = float(dstat["threshold_mass"].get(th, 0.0))
                pct = 100.0 * mass_ok / total_mass if total_mass > eps else 0.0
                th_q[th] = {"mass": mass_ok, "percentage": pct}
            dstat["threshold_stats_q"] = th_q

        # 兼容旧顶层字段 = primary denom
        primary = stats[c]["denoms"][primary_denom]
        stats[c]["mean_error"] = primary["mean_error"]
        stats[c]["max_error"] = primary["max_error"]
        stats[c]["median_error"] = primary["median_error"]
        stats[c]["p95_error"] = primary["p95_error"]
        stats[c]["p99_error"] = primary["p99_error"]
        stats[c]["threshold_stats"] = primary["threshold_stats"]

    return stats


def plot_high_stress_error_analysis(
    pos_np: np.ndarray,
    pred_np: np.ndarray,
    true_np: np.ndarray,
    save_dir: str,
    prefix: str,
    case_id: int,
    channel_names: List[str],
    stress_channel_idx: int = 0,
    lower_ratio: float = 2 / 3,
    upper_ratio: float = 1.0,
    padding_mask: np.ndarray = None,
    per_channel: bool = False,
    relative_denom: str = None,
):
    """
    绘制单个case的高应力区域误差分析图（不绘制 fig.suptitle，子图标题保留）。
    relative_denom: "local" 或 "max"；默认按通道类型自动选择（位移=max，应力=local）。
    """
    num_channels = pred_np.shape[1]
    eps = 1e-12
    if relative_denom is None:
        relative_denom = resolve_rel_err_denoms(channel_names)[0]
    if relative_denom not in ("local", "max"):
        raise ValueError(f"unsupported relative_denom={relative_denom!r}")

    if padding_mask is not None:
        valid_mask = padding_mask.astype(bool)
    else:
        valid_mask = ~((pos_np[:, 0] == 0) & (pos_np[:, 1] == 0))

    pos_valid = pos_np[valid_mask]
    pred_valid = pred_np[valid_mask]
    true_valid = true_np[valid_mask]

    def _rel_err(err, y_true, max_stress):
        if relative_denom == "local":
            return err / (np.abs(y_true) + eps)
        return err / (max_stress + eps)

    denom_tag = f"/{relative_denom}"

    if per_channel:
        for c in range(num_channels):
            stress_values = np.abs(true_valid[:, c])
            max_stress = float(np.max(stress_values)) if len(stress_values) else 0.0
            lower_bound = lower_ratio * max_stress
            upper_bound = upper_ratio * max_stress

            high_stress_mask = (stress_values >= lower_bound) & (stress_values <= upper_bound)
            high_stress_pos = pos_valid[high_stress_mask]
            high_stress_pred = pred_valid[high_stress_mask, c]
            high_stress_true = true_valid[high_stress_mask, c]

            if len(high_stress_pos) == 0:
                print(f"    [!] Case {case_id}, Channel {c}: 没有找到高应力区域点（范围: [{lower_bound:.4f}, {upper_bound:.4f}]）")
                continue

            col_name = channel_names[c] if c < len(channel_names) else f"Ch{c}"
            err = np.abs(high_stress_pred - high_stress_true)
            rel_err = _rel_err(err, high_stress_true, max_stress)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
            _hs_s = 10
            vmin, vmax = _robust_range(high_stress_true, high_stress_true, qlo=0.0, qhi=100.0)
            _scatter_ax(
                axes[0], high_stress_pos, high_stress_true,
                f"{col_name} True (High Stress Region)\nRange: [{lower_bound:.2f}, {upper_bound:.2f}]",
                "coolwarm", vmin, vmax, s=_hs_s, extent_xy=pos_valid,
            )
            _scatter_ax(
                axes[1], high_stress_pos, high_stress_pred,
                f"{col_name} Prediction (High Stress Region)",
                "coolwarm", vmin, vmax, s=_hs_s, extent_xy=pos_valid,
            )
            rel_vmin, rel_vmax = _robust_range(rel_err, rel_err, qlo=0.0, qhi=100.0)
            max_rel = np.max(rel_err)
            mean_rel = np.mean(rel_err)
            _scatter_ax(
                axes[2], high_stress_pos, rel_err,
                f"{col_name} RelErr{denom_tag}\nMax={max_rel:.4f}, Mean={mean_rel:.4f}",
                "coolwarm", rel_vmin, rel_vmax, s=_hs_s, extent_xy=pos_valid,
            )

            os.makedirs(save_dir, exist_ok=True)
            save_paper_fig(
                plt.gcf(),
                os.path.join(save_dir, f"{prefix}_case{case_id}_high_stress_error_ch{c}_{col_name}.png"),
            )
            plt.close(fig)
    else:
        stress_values = np.abs(true_valid[:, stress_channel_idx])
        max_stress = float(np.max(stress_values)) if len(stress_values) else 0.0
        lower_bound = lower_ratio * max_stress
        upper_bound = upper_ratio * max_stress

        high_stress_mask = (stress_values >= lower_bound) & (stress_values <= upper_bound)
        high_stress_pos = pos_valid[high_stress_mask]
        high_stress_pred = pred_valid[high_stress_mask]
        high_stress_true = true_valid[high_stress_mask]

        if len(high_stress_pos) == 0:
            print(f"    [!] Case {case_id}: 没有找到高应力区域点（范围: [{lower_bound:.4f}, {upper_bound:.4f}]）")
            return

        fig, axes = plt.subplots(num_channels, 3, figsize=(18, 5 * num_channels), constrained_layout=True)
        if num_channels == 1:
            axes = axes.reshape(1, 3)

        _hs_s = 10
        for c in range(num_channels):
            y_pred = high_stress_pred[:, c]
            y_true = high_stress_true[:, c]
            col_name = channel_names[c] if c < len(channel_names) else f"Ch{c}"

            err = np.abs(y_pred - y_true)
            rel_err = _rel_err(err, y_true, max_stress)

            vmin, vmax = _robust_range(y_true, y_true, qlo=0.0, qhi=100.0)
            _scatter_ax(
                axes[c, 0], high_stress_pos, y_true,
                f"{col_name} True (High Stress Region)\nRange: [{lower_bound:.2f}, {upper_bound:.2f}]",
                "coolwarm", vmin, vmax, s=_hs_s, extent_xy=pos_valid,
            )
            _scatter_ax(
                axes[c, 1], high_stress_pos, y_pred,
                f"{col_name} Prediction (High Stress Region)",
                "coolwarm", vmin, vmax, s=_hs_s, extent_xy=pos_valid,
            )
            rel_vmin, rel_vmax = _robust_range(rel_err, rel_err, qlo=0.0, qhi=100.0)
            max_rel = np.max(rel_err)
            mean_rel = np.mean(rel_err)
            _scatter_ax(
                axes[c, 2], high_stress_pos, rel_err,
                f"{col_name} RelErr{denom_tag}\nMax={max_rel:.4f}, Mean={mean_rel:.4f}",
                "coolwarm", rel_vmin, rel_vmax, s=_hs_s, extent_xy=pos_valid,
            )

        os.makedirs(save_dir, exist_ok=True)
        save_paper_fig(plt.gcf(), os.path.join(save_dir, f"{prefix}_case{case_id}_high_stress_error.png"))
        plt.close(fig)
