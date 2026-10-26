from __future__ import annotations

import torch

from utils import TestLoss


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute relative L2 error for a batch.

    pred/target: [B, N, C]
    return: scalar tensor (mean over batch)

    Notes:
    - This reuses `utils.TestLoss.rel` to avoid duplicated implementations.
    - We keep a small epsilon in the denominator for numerical stability.
    """
    loss = TestLoss(d=2, p=2, size_average=True, reduction=True, eps=1e-12)
    return loss.rel(pred, target)


def weighted_relative_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_weight: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Function-space relative L2 with optional quadrature weights.

    pred/target: [B, N, C]
    node_weight: [B, N], [B, N, 1], or [N]
    mask: [B, N] or [B, N, 1], True=valid

    Returns mean over batch. For relative L2, q does not need unit mass because
    the same quadrature weights appear in numerator and denominator.
    """
    if node_weight is None:
        if mask is None:
            return relative_l2(pred, target)
        mask_f = _prepare_mask(mask, pred).unsqueeze(-1)
        diff = (pred - target) * mask_f
        denom = target * mask_f
        num = torch.sum(diff.square(), dim=(1, 2))
        den = torch.sum(denom.square(), dim=(1, 2))
        return (torch.sqrt(num + eps) / torch.sqrt(den + eps)).mean()

    q = _prepare_node_weight(node_weight, pred)
    if mask is not None:
        q = q * _prepare_mask(mask, pred)
    w = q.unsqueeze(-1)
    num = torch.sum(w * (pred - target).square(), dim=(1, 2))
    den = torch.sum(w * target.square(), dim=(1, 2))
    return (torch.sqrt(num + eps) / torch.sqrt(den + eps)).mean()


def mse_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    逐点 MSE / MAE（在反归一化后的物理量上计算，与 eval_keti 一致）。

    pred/target: [B, N, C]
    mask: [B, N] 或 [B, N, 1]，True=有效点
    """
    if mask is not None:
        mask_f = _prepare_mask(mask, pred).unsqueeze(-1)
        diff = (pred - target) * mask_f
        n_valid = mask_f.sum().clamp(min=1.0)
        mse = (diff ** 2).sum() / n_valid
        mae = diff.abs().sum() / n_valid
    else:
        diff = pred - target
        mse = (diff ** 2).mean()
        mae = diff.abs().mean()
    return mse, mae


def weighted_mse_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_weight: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Function-space MSE/MAE with optional per-node quadrature weights.

    For absolute metrics, q is normalized to unit mass per sample before
    averaging samples equally.
    """
    if node_weight is None:
        return mse_mae(pred, target, mask=mask)

    q = _prepare_node_weight(node_weight, pred)
    if mask is not None:
        q = q * _prepare_mask(mask, pred)
    mass = q.sum(dim=1, keepdim=True).clamp(min=eps)
    w = (q / mass).unsqueeze(-1)
    diff = pred - target
    mse_per_sample = (w * diff.square()).sum(dim=1).mean(dim=-1)
    mae_per_sample = (w * diff.abs()).sum(dim=1).mean(dim=-1)
    return mse_per_sample.mean(), mae_per_sample.mean()


def _prepare_mask(mask: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.squeeze(-1)
    return mask.to(dtype=pos.dtype, device=pos.device)


def _prepare_node_weight(node_weight: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    b, n = ref.shape[:2]
    q = node_weight.to(dtype=ref.dtype, device=ref.device)
    if q.dim() == 1:
        if q.shape[0] != n:
            raise ValueError(f"node_weight length {q.shape[0]} != N={n}")
        q = q.unsqueeze(0).expand(b, -1)
    elif q.dim() == 2:
        if q.shape == (1, n) and b != 1:
            q = q.expand(b, -1)
        elif q.shape != (b, n):
            raise ValueError(f"node_weight shape {tuple(q.shape)} != {(b, n)}")
    elif q.dim() == 3 and q.shape[-1] == 1:
        q = q.squeeze(-1)
        if q.shape == (1, n) and b != 1:
            q = q.expand(b, -1)
        elif q.shape != (b, n):
            raise ValueError(f"node_weight shape {tuple(q.shape)} != {(b, n, 1)}")
    else:
        raise ValueError(
            f"node_weight must have shape [N], [B,N], or [B,N,1], got {tuple(q.shape)}"
        )
    if not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError("node_weight must be finite and non-negative")
    return q


def pointwise_gradients(
    pos: torch.Tensor,
    y: torch.Tensor,
    k: int = 8,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
    knn_idx: torch.Tensor | None = None,
    knn_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Estimate pointwise gradients on irregular point clouds using local least squares.

    pos: [B, N, D]
    y:   [B, N, C]
    return: gradients [B, N, D, C]
    """
    B, N, D = pos.shape
    if N <= 1:
        return torch.zeros(B, N, D, y.shape[-1], device=pos.device, dtype=pos.dtype)

    k_eff = min(k, N - 1)
    if k_eff <= 0:
        return torch.zeros(B, N, D, y.shape[-1], device=pos.device, dtype=pos.dtype)

    if knn_idx is None:
        dist = torch.cdist(pos, pos)  # [B, N, N]
        eye = torch.eye(N, device=pos.device, dtype=torch.bool)
        dist.masked_fill_(eye[None, ...], float("inf"))

        if mask is not None:
            mask_f = _prepare_mask(mask, pos)
            dist = dist.masked_fill(mask_f[:, None, :] < 0.5, float("inf"))
        else:
            mask_f = None

        knn_dist, knn_idx = torch.topk(dist, k_eff, dim=-1, largest=False)
        neighbor_valid = torch.isfinite(knn_dist)
    else:
        neighbor_valid = knn_valid
        mask_f = _prepare_mask(mask, pos) if mask is not None else None

    idx_pos = knn_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    idx_y = knn_idx.unsqueeze(-1).expand(-1, -1, -1, y.shape[-1])
    pos_exp = pos.unsqueeze(2).expand(-1, -1, k_eff, -1)
    y_exp = y.unsqueeze(2).expand(-1, -1, k_eff, -1)
    pos_neighbors = torch.take_along_dim(pos_exp, idx_pos, dim=1)
    y_neighbors = torch.take_along_dim(y_exp, idx_y, dim=1)

    pos_center = pos.unsqueeze(2)
    y_center = y.unsqueeze(2)
    delta_pos = pos_neighbors - pos_center
    delta_y = y_neighbors - y_center

    if neighbor_valid is not None:
        nv = neighbor_valid.unsqueeze(-1).to(dtype=pos.dtype)
        delta_pos = delta_pos * nv
        delta_y = delta_y * nv

    # Least squares: g = (A^T A + eps I)^-1 A^T b
    At = delta_pos.transpose(-1, -2)  # [B, N, D, k]
    ATA = torch.matmul(At, delta_pos)  # [B, N, D, D]
    eye_d = torch.eye(D, device=pos.device, dtype=pos.dtype).view(1, 1, D, D)
    ATA = ATA + eps * eye_d
    ATb = torch.matmul(At, delta_y)  # [B, N, D, C]

    grads = torch.linalg.solve(ATA, ATb)  # [B, N, D, C]
    return grads


def gradient_loss(
    pos: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    k: int = 8,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
    knn_idx: torch.Tensor | None = None,
    knn_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient loss on irregular point clouds.

    Returns:
        mean_loss (scalar), per_channel_loss [C]
    """
    grad_pred = pointwise_gradients(pos, pred, k=k, mask=mask, eps=eps, knn_idx=knn_idx, knn_valid=knn_valid)
    grad_true = pointwise_gradients(pos, target, k=k, mask=mask, eps=eps, knn_idx=knn_idx, knn_valid=knn_valid)
    diff = grad_pred - grad_true  # [B, N, D, C]
    diff_sq = (diff ** 2).sum(dim=-2)  # [B, N, C]
    true_sq = (grad_true ** 2).sum(dim=-2)  # [B, N, C]

    if mask is not None:
        mask_f = _prepare_mask(mask, pos).unsqueeze(-1)
        diff_sq = diff_sq * mask_f
        true_sq = true_sq * mask_f
        num = diff_sq.sum(dim=(0, 1))
        den = true_sq.sum(dim=(0, 1))
        per_ch = torch.sqrt(num + eps) / (torch.sqrt(den + eps))
    else:
        num = diff_sq.sum(dim=(0, 1))
        den = true_sq.sum(dim=(0, 1))
        per_ch = torch.sqrt(num + eps) / (torch.sqrt(den + eps))

    mean = per_ch.mean()
    return mean, per_ch


def build_knn_cache(
    pos: torch.Tensor,
    k: int = 8,
    mask: torch.Tensor | None = None,
    block_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build kNN indices for point clouds.

    pos: [B, N, D]
    mask: [B, N] or [B, N, 1] (True=valid)
    Returns:
        knn_idx: [B, N, k]
        knn_valid: [B, N, k]
    """
    B, N, _ = pos.shape
    if N <= 1:
        knn_idx = torch.zeros(B, N, 1, dtype=torch.long, device=pos.device)
        knn_valid = torch.zeros(B, N, 1, dtype=torch.bool, device=pos.device)
        return knn_idx, knn_valid

    k_eff = min(k, N - 1)
    if block_size <= 0:
        block_size = N

    knn_idx = torch.empty((B, N, k_eff), dtype=torch.long, device=pos.device)
    knn_valid = torch.empty((B, N, k_eff), dtype=torch.bool, device=pos.device)
    eye = torch.eye(N, device=pos.device, dtype=torch.bool)
    mask_f = _prepare_mask(mask, pos) if mask is not None else None

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        pos_block = pos[:, start:end, :]  # [B, bs, D]
        dist = torch.cdist(pos_block, pos)  # [B, bs, N]
        dist.masked_fill_(eye[None, start:end, :], float("inf"))
        if mask_f is not None:
            dist = dist.masked_fill(mask_f[:, None, :] < 0.5, float("inf"))
            dist = dist.masked_fill(mask_f[:, start:end].unsqueeze(-1) < 0.5, float("inf"))

        knn_dist, knn_idx_block = torch.topk(dist, k_eff, dim=-1, largest=False)
        knn_idx[:, start:end, :] = knn_idx_block
        knn_valid[:, start:end, :] = torch.isfinite(knn_dist)

    return knn_idx, knn_valid


