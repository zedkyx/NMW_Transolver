from __future__ import annotations

import torch
from tqdm import *


def weighted_channel_stats(
    x: torch.Tensor,
    node_weight: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function-space channel mean/std for fields shaped ``[B, N, C]``.

    Quadrature weights are normalized independently for every sample before
    samples are averaged.  Consequently every physical case has equal weight,
    regardless of its node count or total mesh area.  Padding is removed by
    multiplying the quadrature weights by ``mask``.
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"x must have shape [B,N,C] or [N,C], got {tuple(x.shape)}")
    b, n, _ = x.shape

    q = torch.as_tensor(node_weight, device=x.device)
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

    calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    x_calc = x.to(dtype=calc_dtype)
    q = q.to(dtype=calc_dtype)
    if not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError("node_weight must be finite and non-negative")
    if mask is not None:
        m = torch.as_tensor(mask, device=x.device)
        if m.dim() == 3 and m.shape[-1] == 1:
            m = m.squeeze(-1)
        if m.shape != (b, n):
            raise ValueError(f"mask shape {tuple(m.shape)} != {(b, n)}")
        q = q * m.to(dtype=calc_dtype)

    mass = q.sum(dim=1, keepdim=True)
    if (mass <= eps).any():
        bad = torch.nonzero(mass.squeeze(-1) <= eps, as_tuple=False).flatten().tolist()
        raise ValueError(f"node_weight has zero total mass for samples {bad[:10]}")
    q = q / mass
    w = q.unsqueeze(-1)
    mean = (w * x_calc).sum(dim=1).mean(dim=0, keepdim=True).unsqueeze(0)
    variance = (w * (x_calc - mean).square()).sum(dim=1).mean(dim=0, keepdim=True).unsqueeze(0)
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    return mean.to(dtype=x.dtype), std.to(dtype=x.dtype)


class IdentityTransformer():
    def __init__(self, X=None, min_val=None, max_val=None):
        # Min-Max normalization stats (per-channel over batch and node dims)
        # Preferred: provide precomputed min/max; otherwise compute from X over dims (0,1)
        self.eps = 1e-8
        if (min_val is not None) and (max_val is not None):
            self.min = min_val.view(1, 1, -1)
            self.max = max_val.view(1, 1, -1)
        else:
            assert X is not None, "IdentityTransformer requires X or precomputed min/max"
            # X is typically [B, N, C], so reduce over (0, 1)
            self.min = X.amin(dim=(0, 1), keepdim=True)
            self.max = X.amax(dim=(0, 1), keepdim=True)

    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)
        return self

    def cuda(self):
        self.min = self.min.cuda()
        self.max = self.max.cuda()

    def cpu(self):
        self.min = self.min.cpu()
        self.max = self.max.cpu()

    def encode(self, x):
        return (x - self.min) / (self.max - self.min + self.eps)

    def decode(self, x):
        return x * (self.max - self.min + self.eps) + self.min


class UnitTransformer():
    def __init__(self, X, node_weight=None, mask=None, eps: float = 1e-8):
        self.eps = float(eps)
        self.node_weighted = node_weight is not None
        if node_weight is None:
            # Preserve the legacy result exactly when quadrature is disabled.
            self.mean = X.mean(dim=(0, 1), keepdim=True)
            self.std = X.std(dim=(0, 1), keepdim=True) + self.eps
        else:
            self.mean, weighted_std = weighted_channel_stats(
                X, node_weight, mask=mask
            )
            self.std = weighted_std + self.eps

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def encode(self, x):
        x = (x - self.mean) / (self.std)
        return x

    def decode(self, x):
        return x * self.std + self.mean

    def transform(self, X, inverse=True, component='all'):
        if component == 'all' or 'all-reduce':
            if inverse:
                orig_shape = X.shape
                return (X * (self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X - self.mean) / self.std
        else:
            if inverse:
                orig_shape = X.shape
                return (X * (self.std[:, component] - 1e-8) + self.mean[:, component]).view(orig_shape)
            else:
                return (X - self.mean[:, component]) / self.std[:, component]


class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        self.time_last = time_last  # if the time dimension is the last dim

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling mask
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                std = self.std[..., sample_idx] + self.eps  # T*batch*n
                mean = self.mean[..., sample_idx]
        # x is in shape of batch*(spatial discretization size) or T*batch*(spatial discretization size)
        x = (x * std) + mean
        return x

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

