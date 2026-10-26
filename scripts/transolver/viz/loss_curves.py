import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.transolver.viz.style import apply_paper_rcparams, save_paper_fig

apply_paper_rcparams()


def _sanitize_channel_filename(name: str) -> str:
    s = str(name).strip().replace(" ", "_")
    for c in '/\\:*?"<>|':
        s = s.replace(c, "_")
    return s or "ch"


def save_loss_curves_npy(
    train_losses,
    eval_losses,
    epochs_list,
    save_dir,
    train_losses_ch=None,
    eval_losses_ch=None,
    channel_names=None,
):
    """
    将损失曲线数值写入 save_dir 下的 .npy 文件：
    - epochs.npy：epoch 索引（int64）
    - loss_train.npy / loss_eval.npy：整体相对 L2（float64）
    - 每个输出通道：{通道名}_loss_train.npy、{通道名}_loss_eval.npy（如 U1_loss_train.npy）
    """
    if not train_losses or not eval_losses or not epochs_list:
        return
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    ep = np.asarray(epochs_list, dtype=np.int64)
    tr = np.asarray(train_losses, dtype=np.float64)
    ev = np.asarray(eval_losses, dtype=np.float64)
    if tr.shape != ev.shape or tr.shape != ep.shape:
        raise ValueError(
            f"loss curve length mismatch: epochs {ep.shape}, train {tr.shape}, eval {ev.shape}"
        )

    paths = []
    p_ep = os.path.join(save_dir, "epochs.npy")
    p_tr = os.path.join(save_dir, "loss_train.npy")
    p_ev = os.path.join(save_dir, "loss_eval.npy")
    np.save(p_ep, ep)
    np.save(p_tr, tr)
    np.save(p_ev, ev)
    paths.extend([p_ep, p_tr, p_ev])

    if train_losses_ch and eval_losses_ch:
        train_arr = np.asarray(train_losses_ch, dtype=np.float64)
        eval_arr = np.asarray(eval_losses_ch, dtype=np.float64)
        if train_arr.ndim == 1:
            train_arr = train_arr.reshape(-1, 1)
            eval_arr = eval_arr.reshape(-1, 1)
        if train_arr.shape != eval_arr.shape:
            raise ValueError(
                f"per-channel train/eval shape mismatch: {train_arr.shape} vs {eval_arr.shape}"
            )
        _, C = train_arr.shape
        if channel_names is None or len(channel_names) != C:
            channel_names = [f"Ch{i}" for i in range(C)]
        for i in range(C):
            tag = _sanitize_channel_filename(channel_names[i])
            pt = os.path.join(save_dir, f"{tag}_loss_train.npy")
            pe = os.path.join(save_dir, f"{tag}_loss_eval.npy")
            np.save(pt, train_arr[:, i])
            np.save(pe, eval_arr[:, i])
            paths.extend([pt, pe])

    print(f"Loss curve arrays saved ({len(paths)} files) under: {save_dir}")


def plot_loss_curves(train_losses, eval_losses, epochs_list, save_dir, best_eval_info=None):
    """
    Plot training and evaluation loss curves.
    """
    if not train_losses or not eval_losses:
        return

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(epochs_list, train_losses, "b-", label="Training Loss", linewidth=1.8, alpha=0.8)
    ax.plot(epochs_list, eval_losses, "r-", label="Evaluation Loss", linewidth=1.8, alpha=0.8)

    if best_eval_info:
        best_epoch, best_loss = best_eval_info
        ax.plot(best_epoch, best_loss, "ro", markersize=8, label=f"Best Eval: {best_loss:.6f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    loss_plot_path = os.path.join(save_dir, "loss_curves.png")
    save_paper_fig(fig, loss_plot_path, caption="Training and Evaluation Loss Curves")
    plt.close(fig)
    print(f"Loss curves saved to: {loss_plot_path}")


def plot_loss_curves_cylinder(
    *,
    epochs_list,
    save_dir,
    rel_train,
    rel_eval,
    mse_train,
    phys_w_train=None,
    best_eval_info=None,
    filename: str = "loss_curves.png",
) -> None:
    """
    Cylinder training: Rel L2, Train MSE, weighted phys loss on one log-scale figure.
    """
    if not epochs_list or not rel_train or not rel_eval or not mse_train:
        return
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.plot(epochs_list, rel_train, "b-", label="Train Rel L2", linewidth=1.8, alpha=0.85)
    ax.plot(epochs_list, rel_eval, "r-", label="Eval Rel L2", linewidth=1.8, alpha=0.85)
    ax.plot(epochs_list, mse_train, "g-", label="Train MSE", linewidth=1.8, alpha=0.85)
    if phys_w_train:
        ax.plot(epochs_list, phys_w_train, "m-", label="Train phys_loss_w", linewidth=1.8, alpha=0.85)
    if best_eval_info:
        best_epoch, best_loss = best_eval_info
        ax.plot(best_epoch, best_loss, "ro", markersize=8, label=f"Best Eval Rel L2: {best_loss:.6f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    out_path = os.path.join(save_dir, filename)
    save_paper_fig(fig, out_path, caption="Training and Evaluation Loss Curves")
    plt.close(fig)
    print(f"Loss curves saved to: {out_path}")


def plot_loss_curves_channels(train_losses_ch, eval_losses_ch, epochs_list, save_dir, channel_names=None):
    """
    Plot per-channel training/evaluation loss curves.

    train_losses_ch / eval_losses_ch: list of length E, each element is list/tuple/np.ndarray of length C
    """
    if not train_losses_ch or not eval_losses_ch:
        return
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    train_arr = np.asarray(train_losses_ch, dtype=np.float64)
    eval_arr = np.asarray(eval_losses_ch, dtype=np.float64)
    if train_arr.ndim != 2 or eval_arr.ndim != 2:
        raise ValueError(f"train/eval per-channel losses must be 2D, got {train_arr.shape} and {eval_arr.shape}")
    if train_arr.shape != eval_arr.shape:
        raise ValueError(f"train/eval per-channel losses shape mismatch: {train_arr.shape} vs {eval_arr.shape}")

    E, C = train_arr.shape
    if channel_names is None or len(channel_names) != C:
        channel_names = [f"Ch{i}" for i in range(C)]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i in range(C):
        ax.plot(epochs_list, train_arr[:, i], linewidth=1.8, alpha=0.85, label=f"{channel_names[i]} Train")
        ax.plot(
            epochs_list,
            eval_arr[:, i],
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            label=f"{channel_names[i]} Eval",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 Loss (per-channel)")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    out_path = os.path.join(save_dir, "loss_curves_channels.png")
    save_paper_fig(fig, out_path, caption="Per-Channel Training/Evaluation Loss Curves")
    plt.close(fig)
    print(f"Per-channel loss curves saved to: {out_path}")
