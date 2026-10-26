"""Node quadrature weights for 1D/2D irregular point clouds.

Default: Delaunay triangulation + lumped nodal area
  q_i = sum_{T ni i} area(T) / 3

For intrinsically 1D point clouds, use sorted interval lumping:
  q_i = half length of adjacent intervals.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def triangle_areas(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Area of each triangle. points [N,2], simplices [T,3] -> [T]."""
    tri = points[simplices]  # [T,3,2]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def lumped_interval_length(
    pos: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    max_edge: float | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Lumped nodal length for intrinsically 1D point clouds."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim != 2:
        raise ValueError(f"pos must be [N,D], got {pos.shape}")
    n = pos.shape[0]
    q = np.zeros(n, dtype=np.float64)

    if mask is None:
        keep = np.ones(n, dtype=bool)
    else:
        keep = np.asarray(mask, dtype=bool)
        if keep.shape != (n,):
            raise ValueError(f"mask must be [{n}], got {keep.shape}")

    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return q
    if idx.size == 1:
        q[idx] = eps
        return q

    pts = pos[idx]
    span = pts.max(axis=0) - pts.min(axis=0)
    axis = int(np.argmax(span))
    x = pts[:, axis]
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    gaps = np.diff(x_sorted)
    gaps = np.maximum(gaps, 0.0)
    if max_edge is not None:
        gaps = np.where(gaps <= float(max_edge), gaps, 0.0)

    q_sorted = np.zeros_like(x_sorted)
    q_sorted[0] = gaps[0] / 2.0
    q_sorted[-1] = gaps[-1] / 2.0
    if idx.size > 2:
        q_sorted[1:-1] = (gaps[:-1] + gaps[1:]) / 2.0

    q[idx[order]] = q_sorted
    positive = keep & (q > 0)
    q[positive] = np.maximum(q[positive], eps)
    orphan = keep & (q <= 0)
    if np.any(orphan):
        q[orphan] = eps
    return q


def _intrinsic_dim(pos: np.ndarray, tol: float = 1e-12) -> int:
    """Return a small numerical-rank estimate for [N,D] point coordinates."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape[0] <= 1:
        return 0
    centered = pos - pos.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(centered, ord=2)
    if scale <= tol:
        return 0
    return int(np.linalg.matrix_rank(centered, tol=tol * max(1.0, scale)))


def lumped_area_delaunay(
    pos: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    max_edge: float | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Lumped nodal area from Delaunay triangulation.

    Parameters
    ----------
    pos : [N, 2]
    mask : [N] bool, optional. False nodes get q=0 and are excluded from triangulation.
    max_edge : if set, drop triangles whose longest edge exceeds this (hole / artifact filter).
    eps : floor for positive weights on kept nodes.

    Returns
    -------
    q : [N] float64, sum(q[mask]) equals sum of kept triangle areas.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"pos must be [N,2], got {pos.shape}")
    n = pos.shape[0]
    q = np.zeros(n, dtype=np.float64)

    if mask is None:
        keep = np.ones(n, dtype=bool)
    else:
        keep = np.asarray(mask, dtype=bool)
        if keep.shape != (n,):
            raise ValueError(f"mask must be [{n}], got {keep.shape}")

    idx = np.flatnonzero(keep)
    if idx.size < 3:
        q[idx] = eps
        return q

    pts = pos[idx]
    if _intrinsic_dim(pts) < 2:
        return lumped_interval_length(pos, mask=mask, max_edge=max_edge, eps=eps)

    delaunay = Delaunay(pts)
    simplices = delaunay.simplices
    areas = triangle_areas(pts, simplices)

    if max_edge is not None:
        tri = pts[simplices]
        e01 = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
        e12 = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
        e20 = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
        longest = np.maximum(np.maximum(e01, e12), e20)
        ok = longest <= float(max_edge)
        simplices = simplices[ok]
        areas = areas[ok]

    for t, (i, j, k) in enumerate(simplices):
        share = areas[t] / 3.0
        q[idx[i]] += share
        q[idx[j]] += share
        q[idx[k]] += share

    positive = keep & (q > 0)
    q[positive] = np.maximum(q[positive], eps)
    # nodes kept but with zero area (rare) get eps so they stay usable
    orphan = keep & (q <= 0)
    if np.any(orphan):
        q[orphan] = eps
    return q


def convex_hull_area(pos: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape[0] < 3:
        return 0.0
    return float(ConvexHull(pos).volume)  # 2D: volume == area


def lumped_area_delaunay_batch(
    pos,
    mask=None,
    *,
    max_edge: float | None = None,
    eps: float = 1e-12,
    show_progress: bool = True,
):
    """Batch Delaunay lumped area.

    Parameters
    ----------
    pos : array-like [B, N, D], D=1 or 2. Intrinsically 1D data uses interval weights.
    mask : array-like [B, N] bool, optional (True=valid)

    Returns
    -------
    q : np.ndarray [B, N] float32
    """
    pos_np = np.asarray(pos, dtype=np.float64)
    if pos_np.ndim != 3 or pos_np.shape[-1] not in (1, 2):
        raise ValueError(f"pos must be [B,N,D] with D=1 or 2, got {pos_np.shape}")
    b, n, d = pos_np.shape
    mask_np = None
    if mask is not None:
        mask_np = np.asarray(mask, dtype=bool)
        if mask_np.shape != (b, n):
            raise ValueError(f"mask must be [{b},{n}], got {mask_np.shape}")

    q = np.zeros((b, n), dtype=np.float64)
    indices = range(b)
    if show_progress:
        try:
            from tqdm import tqdm

            indices = tqdm(indices, desc="node_weight (Delaunay)", total=b)
        except ImportError:
            pass

    for i in indices:
        m_i = None if mask_np is None else mask_np[i]
        if d == 1:
            q[i] = lumped_interval_length(pos_np[i], mask=m_i, max_edge=max_edge, eps=eps)
        else:
            q[i] = lumped_area_delaunay(pos_np[i], mask=m_i, max_edge=max_edge, eps=eps)
    return q.astype(np.float32)
