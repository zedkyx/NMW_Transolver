from __future__ import annotations

import os
import json
import math
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import numpy as np


def _ensure_plot_dir(base_dir: str):
    plot_dir = os.path.join(base_dir, 'plot')
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def _setup_paper_font():
    """论文风格：英文字体 Times New Roman；中文回退 SimHei / Arial Unicode MS"""
    import logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def _to_nhg(t: torch.Tensor, batch_index: int = 0) -> torch.Tensor:
    """Normalize attention factor to [H, N, G] float CPU/GPU tensor."""
    if t is None:
        return None
    x = t.detach()
    if x.dim() == 4:
        x = x[batch_index]
    if x.dim() != 3:
        raise ValueError(f"expected [B,H,N,G] or [H,N,G], got {tuple(t.shape)}")
    return x


def attn_matrix_singular_values(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Singular values of A = φ ψ^T without forming N×N.

    φ, ψ: [N, G]  →  A: [N, N], rank(A) ≤ G.
    - Shared W (Transolver): A=WW^T ⇒ σ_i(A)=σ_i(W)^2，只需一次 svdvals(W).
    - General (LinearNO): economy SVD of φ then svdvals of the thin factor (same nonzero σ as svd(A)).
    """
    if phi.dim() != 2 or psi.dim() != 2:
        raise ValueError(f"phi/psi must be [N,G], got {tuple(phi.shape)}, {tuple(psi.shape)}")
    if phi.shape != psi.shape:
        raise ValueError(f"phi/psi shape mismatch: {tuple(phi.shape)} vs {tuple(psi.shape)}")

    # Prefer GPU float32; avoid host double SVD on large N.
    phi_f = phi.float()
    if phi is psi:
        s = torch.linalg.svdvals(phi_f)
        return s * s

    psi_f = psi.float()
    _, s, vh = torch.linalg.svd(phi_f, full_matrices=False)
    thin = (s.unsqueeze(1) * vh) @ psi_f.transpose(0, 1)  # [k, N], k≤G
    return torch.linalg.svdvals(thin)


def numerical_rank_from_sv(sv: torch.Tensor, rel_tol: float = 1e-3) -> int:
    """Count σ_i / σ_max > rel_tol (σ descending)."""
    if sv is None or sv.numel() == 0:
        return 0
    s = sv.detach().float().flatten()
    smax = s[0] if float(s[0].item()) > 0 else s.max()
    if float(smax.item()) <= 0:
        return 0
    return int((s / smax > rel_tol).sum().item())


def batched_numerical_ranks(
    phi: torch.Tensor,
    psi: torch.Tensor,
    mask: torch.Tensor | None = None,
    rel_tol: float = 1e-3,
    shared: bool | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Vectorized numerical rank of A=φψ^T over a batch.

    Args:
        phi, psi: [B, H, N, G]
        mask: optional [B, N] (True/1 = valid). Pad rows are zeroed (equiv. to dropping them for colspace).
        shared: if True, treat as Transolver A=WW^T (φ=ψ). Default: phi is psi.
        chunk_size: optional max samples per SVD call (OOM fallback). None = all at once.
    Returns:
        mean rank over heads, shape [B] (float tensor on same device).
    """
    if phi.dim() != 4 or psi.dim() != 4:
        raise ValueError(f"phi/psi must be [B,H,N,G], got {tuple(phi.shape)}, {tuple(psi.shape)}")
    if phi.shape != psi.shape:
        raise ValueError(f"phi/psi shape mismatch: {tuple(phi.shape)} vs {tuple(psi.shape)}")
    B, H, N, G = phi.shape
    if shared is None:
        shared = phi is psi

    if chunk_size is not None and chunk_size > 0 and B > chunk_size:
        outs = []
        for i0 in range(0, B, chunk_size):
            i1 = min(i0 + chunk_size, B)
            m = None if mask is None else mask[i0:i1]
            outs.append(
                batched_numerical_ranks(
                    phi[i0:i1],
                    psi[i0:i1],
                    mask=m,
                    rel_tol=rel_tol,
                    shared=shared,
                    chunk_size=None,
                )
            )
        return torch.cat(outs, dim=0)

    phi_f = phi.float()
    psi_f = psi.float() if not shared else phi_f
    if mask is not None:
        m = mask
        if not torch.is_tensor(m):
            m = torch.as_tensor(m, device=phi_f.device)
        m = m.to(device=phi_f.device)
        if m.dim() == 3:
            m = m.squeeze(-1)
        if m.shape != (B, N):
            raise ValueError(f"mask shape {tuple(m.shape)} != {(B, N)}")
        m4 = m.reshape(B, 1, N, 1).to(dtype=phi_f.dtype)
        phi_f = phi_f * m4
        if not shared:
            psi_f = psi_f * m4

    bh = B * H
    phi2 = phi_f.reshape(bh, N, G)
    if shared:
        sv = torch.linalg.svdvals(phi2) ** 2  # [BH, G], σ(A)=σ(W)^2
    else:
        psi2 = psi_f.reshape(bh, N, G)
        _, s, vh = torch.linalg.svd(phi2, full_matrices=False)
        thin = torch.bmm(s.unsqueeze(-1) * vh, psi2.transpose(1, 2))
        sv = torch.linalg.svdvals(thin)

    smax = sv[:, :1].clamp_min(1e-30)
    ranks = (sv / smax > rel_tol).sum(dim=-1).to(dtype=torch.float32).view(B, H)
    return ranks.mean(dim=-1)


def _factors_from_attn_item_full_batch(item: dict):
    """Return (φ, ψ, shared) with shapes [B,H,N,G] or (None, None, False)."""
    phi = item.get("phi", None)
    psi = item.get("psi", None)
    sw = item.get("slice_weights", None)
    if isinstance(phi, torch.Tensor) and isinstance(psi, torch.Tensor):
        if phi.dim() != 4 or psi.dim() != 4:
            raise ValueError(f"expected [B,H,N,G], got phi={tuple(phi.shape)} psi={tuple(psi.shape)}")
        return phi.detach(), psi.detach(), False
    if isinstance(sw, torch.Tensor):
        if sw.dim() != 4:
            raise ValueError(f"expected slice_weights [B,H,N,G], got {tuple(sw.shape)}")
        w = sw.detach()
        return w, w, True
    return None, None, False


def _phi_psi_from_attn_item(item: dict, batch_index: int = 0):
    """Resolve (φ, ψ) as [H,N,G].

    LinearNO: last_phi / last_psi
    Transolver Physics-Attn: shared W → φ=ψ=slice_weights  ⇒  A = W W^T
    """
    phi = item.get("phi", None)
    psi = item.get("psi", None)
    sw = item.get("slice_weights", None)
    if isinstance(phi, torch.Tensor) and isinstance(psi, torch.Tensor):
        return _to_nhg(phi, batch_index), _to_nhg(psi, batch_index)
    if isinstance(sw, torch.Tensor):
        w = _to_nhg(sw, batch_index)
        return w, w
    return None, None


def ranks_for_attn_item(
    item: dict,
    mask=None,
    rel_tol: float = 1e-3,
    batch_index: int = 0,
) -> dict:
    """Per-head numerical rank of A=φψ^T for one recorded layer (single sample)."""
    layer = int(item.get("layer", -1))
    phi, psi, shared = _factors_from_attn_item_full_batch(item)
    if phi is None:
        # fall back to single-index extraction
        phi1, psi1 = _phi_psi_from_attn_item(item, batch_index=batch_index)
        if phi1 is None:
            return {"layer": layer, "ranks_per_head": [], "mean_rank": float("nan"), "slice_num": 0}
        phi = phi1.unsqueeze(0)
        psi = psi1.unsqueeze(0)
        shared = phi1 is psi1
        B = 1
        bi = 0
    else:
        B = phi.shape[0]
        bi = batch_index
        phi = phi[bi : bi + 1]
        psi = psi[bi : bi + 1]

    H, N, G = phi.shape[1], phi.shape[2], phi.shape[3]
    mask_b = None
    if mask is not None:
        m = mask.detach() if torch.is_tensor(mask) else torch.as_tensor(mask)
        if m.dim() == 2:
            m = m[bi]
        mask_b = m.reshape(1, -1)
    mean_r = float(batched_numerical_ranks(phi, psi, mask=mask_b, rel_tol=rel_tol, shared=shared)[0].item())
    return {"layer": layer, "ranks_per_head": [mean_r] * H, "mean_rank": mean_r, "slice_num": G}


def accumulate_sample_ranks(running: dict, sample_layer_results: list) -> dict:
    """Accumulate per-layer mean_rank over samples. running keys: sum_rank, count, slice_num."""
    if running is None:
        running = {}
    for res in sample_layer_results:
        layer = int(res["layer"])
        if layer not in running:
            running[layer] = {
                "sum_rank": 0.0,
                "count": 0,
                "slice_num": int(res.get("slice_num", 0)),
            }
        if res["mean_rank"] == res["mean_rank"]:  # not NaN
            running[layer]["sum_rank"] += float(res["mean_rank"])
            running[layer]["count"] += 1
            running[layer]["slice_num"] = int(res.get("slice_num", running[layer]["slice_num"]))
    return running


def finalize_rank_series(running: dict, *, label: str, rel_tol: float, n_samples: int) -> dict:
    layers = sorted(running.keys())
    mean_ranks = []
    for L in layers:
        c = max(int(running[L]["count"]), 1)
        mean_ranks.append(running[L]["sum_rank"] / c)
    slice_num = int(running[layers[0]]["slice_num"]) if layers else 0
    return {
        "label": label,
        "layers": layers,
        "mean_ranks": mean_ranks,
        "slice_num": slice_num,
        "n_samples": int(n_samples),
        "rel_tol": float(rel_tol),
    }


def save_attn_rank_json(series: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(series, f, indent=2)


def load_attn_rank_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_attn_rank_curves(
    series_list,
    base_dir: str,
    tag: str = "attn_matrix_avg_rank",
    rel_tol: float = 1e-3,
):
    """Paper-style Fig.4(d): layer vs average rank. series_list: list of finalize_rank_series dicts."""
    if isinstance(series_list, dict) and "layers" in series_list:
        series_list = [series_list]
    plot_dir = _ensure_plot_dir(base_dir)
    _setup_paper_font()

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    fig.patch.set_facecolor("white")
    markers = ["o", "s", "^", "D", "v"]
    ymax = 1
    for i, ser in enumerate(series_list):
        layers = ser["layers"]
        ranks = ser["mean_ranks"]
        label = ser.get("label", f"run{i}")
        M = int(ser.get("slice_num", 0) or 0)
        ymax = max(ymax, M, max(ranks) if ranks else 0)
        ax.plot(
            layers,
            ranks,
            marker=markers[i % len(markers)],
            linewidth=1.5,
            markersize=5,
            label=label,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Avg rank")
    ax.set_xticks(sorted({L for ser in series_list for L in ser["layers"]}))
    ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1)
    if len(series_list) > 1:
        ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_png = os.path.join(plot_dir, f"{tag}.png")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # also dump combined json next to plot
    out_json = os.path.join(plot_dir, f"{tag}.json")
    payload = {
        "rel_tol": rel_tol,
        "series": series_list,
    }
    save_attn_rank_json(payload, out_json)
    return out_png, out_json


def _valid_pos_weights(pos_np: np.ndarray, slice_weights: np.ndarray, mask=None):
    """Apply optional [N] mask; return (pos_v [Nv,2], w_v [Nv,G])."""
    w = np.asarray(slice_weights, dtype=np.float64)
    p = np.asarray(pos_np, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask).astype(bool).reshape(-1)
        if m.shape[0] != p.shape[0]:
            raise ValueError(f"mask length {m.shape[0]} != N={p.shape[0]}")
        p = p[m]
        w = w[m]
    return p, w


def _plot_slice_weight_grid(
    pos_np: np.ndarray,
    slice_weights: np.ndarray,
    layer: int,
    head: int,
    tag: str,
    plot_dir: str,
    mask=None,
):
    """A: soft W[:, g] on geometry, one panel per slice. No title."""
    _setup_paper_font()
    pos_v, w_v = _valid_pos_weights(pos_np, slice_weights, mask=mask)
    num_nodes, num_slices = w_v.shape
    print(f"      [A] soft grid layer={layer} head={head}: Nv={num_nodes}, G={num_slices}", flush=True)
    ncols = min(8, num_slices)
    nrows = int(math.ceil(num_slices / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.0 * nrows), squeeze=False)
    fig.patch.set_facecolor('white')

    for s in range(num_slices):
        r, c = divmod(s, ncols)
        ax = axes[r, c]
        w = w_v[:, s]
        # 每 slice 独立量程（percentile 1,99），凸显空间分布模式
        vmin = float(np.percentile(w, 1))
        vmax = float(np.percentile(w, 99))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.min(w)), float(np.max(w))
            if vmax <= vmin:
                vmax = vmin + 1e-12
        ax.scatter(
            pos_v[:, 0], pos_v[:, 1],
            c=w,
            cmap='viridis',
            s=1.5,
            alpha=0.9,
            edgecolors='none',
            vmin=vmin,
            vmax=vmax
        )
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    total_axes = nrows * ncols
    for s in range(num_slices, total_axes):
        r, c = divmod(s, ncols)
        axes[r, c].axis('off')

    fig.tight_layout(rect=[0, 0, 0.88, 1.0])
    cb_w, cb_h = 0.018, 0.65
    cb_left = 0.91
    cb_bottom = (1.0 - cb_h) / 2.0
    cax = fig.add_axes([cb_left, cb_bottom, cb_w, cb_h])
    # colorbar: 相对权重示意（各子图独立量程）；标签简短
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r'$W$', fontsize=9)
    out_path = os.path.join(plot_dir, f'{tag}_layer{layer}_head{head}_slice_weight_grid.png')
    fig.savefig(out_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"      ✓ {out_path}", flush=True)
    return out_path


def _plot_slice_hard_assignment(
    pos_np: np.ndarray,
    slice_weights: np.ndarray,
    layer: int,
    head: int,
    tag: str,
    plot_dir: str,
    mask=None,
):
    """B: one map, color = argmax_g W[i,g]. Discrete colormap, no title."""
    _setup_paper_font()
    pos_v, w_v = _valid_pos_weights(pos_np, slice_weights, mask=mask)
    num_nodes, num_slices = w_v.shape
    hard = np.argmax(w_v, axis=1).astype(np.int32)
    counts = np.bincount(hard, minlength=num_slices)
    n_used = int((counts > 0).sum())
    print(
        f"      [B] hard assign layer={layer} head={head}: "
        f"Nv={num_nodes}, used_slices={n_used}/{num_slices}, "
        f"mass_top5={np.sort(counts)[::-1][:5].tolist()}",
        flush=True,
    )

    # tab20 wraps every 20; for G=32 use concatenated discrete colors
    base = plt.get_cmap("tab20")
    colors = [base(i % 20) for i in range(num_slices)]
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    fig.patch.set_facecolor("white")
    sc = ax.scatter(
        pos_v[:, 0],
        pos_v[:, 1],
        c=hard,
        cmap=cmap,
        vmin=-0.5,
        vmax=num_slices - 0.5,
        s=2.0,
        alpha=0.95,
        edgecolors="none",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, ticks=[0, num_slices - 1])
    cb.set_label("slice", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(plot_dir, f"{tag}_layer{layer}_head{head}_slice_hard.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"      ✓ {out_path}", flush=True)
    return out_path, {"used_slices": n_used, "counts": counts.tolist()}


def save_attention_heatmaps(
    attn_list,
    tag: str = "attn",
    base_dir: str = ".",
    pos: np.ndarray = None,
    plot_slice_grid: bool = True,
    plot_hard_assign: bool = False,
    last_layer_only: bool = True,
    mask=None,
    batch_index: int = 0,
):
    """
    attn_list: list of dicts from model.get_last_attention()
        each item: { 'layer': int, 'attn': Tensor[B,H,G,G], 'slice_weights': Tensor[B,H,N,G] }
    pos: spatial coordinates [N,2] or [B,N,2]
    plot_slice_grid (A): soft W per slice
    plot_hard_assign (B): argmax slice id map
    last_layer_only: True → only max layer; False → every layer
    mask: optional [N] or [B,N] valid-node mask
    """
    plot_dir = _ensure_plot_dir(base_dir)
    max_layer = max((item.get("layer", -1) for item in attn_list), default=-1)
    saved = []
    for item in attn_list:
        layer = item.get("layer", -1)
        slice_weights = item.get("slice_weights", None)

        if not isinstance(slice_weights, torch.Tensor):
            continue
        if last_layer_only and layer != max_layer:
            continue

        sw = slice_weights[batch_index].detach().float().cpu().numpy()  # [H, N, G]
        num_heads = sw.shape[0]

        if pos is None:
            print(f"      [!] layer={layer}: pos is None, skip spatial slice plots", flush=True)
            continue

        if isinstance(pos, torch.Tensor):
            pos_t = pos.detach().cpu().numpy()
        else:
            pos_t = np.asarray(pos)
        if pos_t.ndim == 3:
            pos_np = pos_t[batch_index]
        else:
            pos_np = pos_t

        mask_np = None
        if mask is not None:
            m = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
            if m.ndim == 2:
                m = m[batch_index]
            mask_np = m.reshape(-1).astype(bool)

        for h in range(num_heads):
            if plot_slice_grid:
                saved.append(
                    _plot_slice_weight_grid(
                        pos_np=pos_np,
                        slice_weights=sw[h],
                        layer=layer,
                        head=h,
                        tag=tag,
                        plot_dir=plot_dir,
                        mask=mask_np,
                    )
                )
            if plot_hard_assign:
                path_b, _stats = _plot_slice_hard_assignment(
                    pos_np=pos_np,
                    slice_weights=sw[h],
                    layer=layer,
                    head=h,
                    tag=tag,
                    plot_dir=plot_dir,
                    mask=mask_np,
                )
                saved.append(path_b)

    return plot_dir


