"""
KETI data loader (cache-only).

The project has been refactored to decouple "data processing / cache building"
from "training". In the container, training should ONLY rely on a pre-built
PyTorch cache at `args.cache_path`.

Expected cache format (torch.save dict):
  - pos: (B, N, 2) float32
  - fx : (B, N, F) float32
  - y  : (B, N, C) float32
Optional:
  - case_params: (B, 13) float32
  - case_names: list[str] length B
  - geom_id: (B,) int64 — merged multi-geometry cache
  - mask: (B, N) bool
  - node_weight: (B, N) float32 — Delaunay lumped nodal area (or FEM mass)
  - metadata.train_cases / metadata.test_cases for case-level split
  - metadata.train_geom_ids / metadata.test_geom_ids for geometry-level split
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import torch

from utils.quadrature import lumped_area_delaunay_batch

NODE_WEIGHT_METHOD_DELAUNAY = "delaunay_lumped_area_on_normalized_pos"
# mesh meta string lives in utils.abaqus_mesh_weight.NODE_WEIGHT_METHOD_MESH


def resolve_node_weight_method(args) -> str | None:
    """Return 'delaunay' | 'mesh' | None (weights off).

    Config:
      use_node_weight: 0/1
      node_weight_method: delaunay | mesh
    Legacy alias: node_weight_mesh=1 → method=mesh.
    """
    if not bool(int(getattr(args, "use_node_weight", 0))):
        return None
    raw = str(getattr(args, "node_weight_method", "") or "").strip().lower()
    if not raw:
        raw = "mesh" if bool(int(getattr(args, "node_weight_mesh", 0))) else "delaunay"
    if raw in ("delaunay", "mesh"):
        return raw
    raise ValueError(
        f"node_weight_method must be 'delaunay' or 'mesh', got {raw!r}"
    )


def _resolve_norm_pathway_flag(args, specific_attr: str) -> bool:
    """Resolve node_weight_in_norm_{fx,y}; negative / unset → inherit node_weight_in_norm."""
    raw = getattr(args, specific_attr, None)
    if raw is None:
        return bool(int(getattr(args, "node_weight_in_norm", 1)))
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return bool(int(getattr(args, "node_weight_in_norm", 1)))
    if v < 0:
        return bool(int(getattr(args, "node_weight_in_norm", 1)))
    return bool(v)


def resolve_node_weight_pathways(args) -> dict[str, bool]:
    """Where to apply node_weight when use_node_weight=1.

    Keys: load, attn, norm_fx, norm_y, norm, loss.
    ``norm`` is (norm_fx or norm_y) for legacy callers.
    Defaults: attn=norm=loss=1 when use_node_weight=1; norm_fx/norm_y inherit
    ``node_weight_in_norm`` unless ``node_weight_in_norm_{fx,y}`` is set (>=0).
    use_node_weight=0 → all False.
    """
    use = bool(int(getattr(args, "use_node_weight", 0)))
    if not use:
        return {
            "load": False,
            "attn": False,
            "norm_fx": False,
            "norm_y": False,
            "norm": False,
            "loss": False,
        }
    norm_fx = _resolve_norm_pathway_flag(args, "node_weight_in_norm_fx")
    norm_y = _resolve_norm_pathway_flag(args, "node_weight_in_norm_y")
    return {
        "load": True,
        "attn": bool(int(getattr(args, "node_weight_in_attn", 1))),
        "norm_fx": norm_fx,
        "norm_y": norm_y,
        "norm": bool(norm_fx or norm_y),
        "loss": bool(int(getattr(args, "node_weight_in_loss", 1))),
    }


def cache_method_matches(cache_method: str, wanted: str) -> bool:
    """Whether cache metadata.node_weight_method matches requested scheme."""
    cache_method = str(cache_method or "")
    if wanted == "mesh":
        from utils.abaqus_mesh_weight import NODE_WEIGHT_METHOD_MESH

        # Exact string: current training default (equal_div_inc).
        # Historical equal/HRZ caches must recompute or pass explicit lumping.
        return cache_method == NODE_WEIGHT_METHOD_MESH
    if wanted == "delaunay":
        # accept legacy empty / unspecified as delaunay-era caches
        return cache_method in ("", NODE_WEIGHT_METHOD_DELAUNAY) or (
            "delaunay" in cache_method.lower()
        )
    return False


def resolve_output_cols(output_cols, num_channels: int) -> list[str]:
    """Align channel names with y.shape[-1]; pad missing names as Ch{i}."""
    if output_cols is None:
        cols: list[str] = []
    elif isinstance(output_cols, str):
        cols = output_cols.split()
    else:
        cols = list(output_cols)
    if len(cols) == 1 and cols and " " in cols[0]:
        cols = cols[0].split()
    while len(cols) < num_channels:
        cols.append(f"Ch{len(cols)}")
    return cols


def _case_y_l2_norm(y: torch.Tensor) -> torch.Tensor:
    """Per-case L2 norm of y over nodes and channels. Shape: [B]."""
    return torch.sqrt((y.float() ** 2).sum(dim=(1, 2)))


def find_degenerate_y_cases(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Return indices of cases whose ||y||_2 is below eps (all channels combined).

    These samples make relative-L2 metrics explode because the denominator ~ 0.
    """
    norms = _case_y_l2_norm(y)
    return (norms < eps).nonzero(as_tuple=True)[0]


def filter_degenerate_y_cases(
    pos: torch.Tensor,
    fx: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 1e-6,
    case_params: torch.Tensor | None = None,
    sato_indices: dict | None = None,
    case_t: torch.Tensor | None = None,
    case_names: Sequence[str] | None = None,
    geom_id: torch.Tensor | None = None,
    node_weight: torch.Tensor | None = None,
    cache_mask: torch.Tensor | None = None,
    cache_path: str = "",
    filter_enabled: bool = True,
) -> tuple:
    """
    Detect near-zero ||y|| cases and optionally drop them before train/test split.

    Returns filtered tensors + num_removed + keep_idx.
    """
    bad_idx = find_degenerate_y_cases(y, eps=eps)
    num_bad = int(bad_idx.numel())
    if num_bad == 0:
        return pos, fx, y, case_params, sato_indices, case_t, 0, None, geom_id, node_weight, cache_mask

    norms = _case_y_l2_norm(y)[bad_idx]
    label = os.path.basename(cache_path) if cache_path else "cache"
    print(f"[!] Degenerate y cases in {label}: {num_bad} sample(s) with ||y||_2 < {eps:g}")
    for j, idx in enumerate(bad_idx.tolist()):
        name_suffix = ""
        if case_names is not None and 0 <= idx < len(case_names):
            name_suffix = f", case_name={case_names[idx]!r}"
        print(f"    [{j + 1}/{num_bad}] index={idx}, ||y||_2={norms[j].item():.6e}{name_suffix}")

    if not filter_enabled:
        print("    filter_degenerate_y=0: kept all samples (TrainRel may be misleading)")
        return pos, fx, y, case_params, sato_indices, case_t, 0, None, geom_id, node_weight, cache_mask

    keep = _case_y_l2_norm(y) >= eps
    keep_idx = keep.nonzero(as_tuple=True)[0]
    pos = pos[keep_idx]
    fx = fx[keep_idx]
    y = y[keep_idx]
    if case_params is not None:
        case_params = case_params[keep_idx]
    if sato_indices is not None:
        sato_indices = {
            key: val[keep_idx] for key, val in sato_indices.items()
        }
    if case_t is not None:
        case_t = case_t[keep_idx]
    if geom_id is not None:
        geom_id = geom_id[keep_idx]
    if node_weight is not None:
        node_weight = node_weight[keep_idx]
    if cache_mask is not None:
        cache_mask = cache_mask[keep_idx]
    print(f"    filter_degenerate_y=1: removed {num_bad}, remaining {int(keep_idx.numel())} sample(s)")
    return pos, fx, y, case_params, sato_indices, case_t, num_bad, keep_idx, geom_id, node_weight, cache_mask


def _resolve_weight_mask(
    pos: torch.Tensor,
    cache_mask: torch.Tensor | None,
    *,
    padding_enabled: bool,
    padding_value: float,
) -> torch.Tensor | None:
    if cache_mask is not None:
        return cache_mask.bool()
    if padding_enabled:
        pad = torch.tensor(padding_value, dtype=pos.dtype)
        return ~((pos == pad).all(dim=-1))
    return None


def normalize_node_weight_max_edge(max_edge) -> float | None:
    """Return None when max_edge <= 0, otherwise a positive float threshold."""
    if max_edge is None:
        return None
    max_edge = float(max_edge)
    return None if max_edge <= 0 else max_edge


def compute_node_weight_from_normalized_pos(
    pos_n: torch.Tensor,
    *,
    use_node_weight: bool,
    cache_mask: torch.Tensor | None = None,
    padding_enabled: bool = False,
    padding_value: float = 0.0,
    max_edge: float | None = None,
    persist_path: str = "",
    persist_source_pos_shape: tuple | None = None,
) -> torch.Tensor | None:
    """
    Compute quadrature weights in the same coordinate system seen by the model.

    `pos_n` should already be normalized by the fitted coordinate normalizer.
    `max_edge` is therefore interpreted in normalized-coordinate units; values
    <= 0 should be converted to None by normalize_node_weight_max_edge.
    """
    if not use_node_weight:
        return None

    mask = _resolve_weight_mask(
        pos_n, cache_mask, padding_enabled=padding_enabled, padding_value=padding_value
    )
    print("[*] computing node_weight from normalized pos (Delaunay / 1D interval) ...")
    if max_edge is None:
        print("    node_weight_max_edge: disabled")
    else:
        print(f"    node_weight_max_edge: {max_edge:g} (normalized-coordinate units)")
    q_np = lumped_area_delaunay_batch(
        pos_n.detach().cpu().numpy(),
        None if mask is None else mask.detach().cpu().numpy(),
        max_edge=max_edge,
        show_progress=True,
    )
    nw = torch.from_numpy(q_np)
    print(
        f"[*] node_weight computed on normalized pos: shape={tuple(nw.shape)}, "
        f"min={nw[nw > 0].min().item() if (nw > 0).any() else 0:.3e}, "
        f"max={nw.max().item():.3e}, mean={nw.mean().item():.3e}"
    )

    if persist_path:
        persist_node_weight_to_cache(
            persist_path,
            nw,
            source_pos_shape=persist_source_pos_shape or tuple(pos_n.shape),
            method=NODE_WEIGHT_METHOD_DELAUNAY,
        )
    return nw


def persist_node_weight_to_cache(
    cache_path: str,
    node_weight: torch.Tensor,
    *,
    source_pos_shape: tuple | None = None,
    method: str = NODE_WEIGHT_METHOD_DELAUNAY,
    extra_meta: dict | None = None,
) -> None:
    """Write/overwrite node_weight in an existing cache.pt (full file rewrite)."""
    if not cache_path:
        raise ValueError("persist_node_weight requires cache_path")
    cache = _load_cache(cache_path)
    pos = cache.get("pos")
    if pos is None:
        raise KeyError(f"cache missing pos: {cache_path}")
    # Only persist when shapes match the on-disk cache (no subsample/filter applied).
    if tuple(pos.shape[:2]) != tuple(node_weight.shape):
        print(
            f"[!] skip persist node_weight: cache pos {tuple(pos.shape)} "
            f"!= weight {tuple(node_weight.shape)} "
            f"(likely frame subsample / degenerate filter). "
            f"Re-run without subsample to write into cache."
        )
        return
    if source_pos_shape is not None and tuple(source_pos_shape) != tuple(pos.shape):
        print(
            f"[!] skip persist node_weight: in-memory pos {source_pos_shape} "
            f"!= cache pos {tuple(pos.shape)}"
        )
        return
    cache["node_weight"] = node_weight.detach().cpu().float()
    meta = cache.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    meta["node_weight_method"] = method
    if extra_meta:
        meta.update(extra_meta)
    cache["metadata"] = meta
    torch.save(cache, cache_path)
    print(f"[*] persisted node_weight -> {cache_path}  method={method}")


def compute_node_weight_from_abaqus_mesh(
    pos_phys: torch.Tensor,
    *,
    cache_mask: torch.Tensor | None = None,
    padding_enabled: bool = False,
    padding_value: float = 0.0,
    abaqus_root: str,
    npz_dir: str,
    instance_name: str = "QJT-1",
    tol: float = 1e-3,
    n_workers: int = 8,
    persist_path: str = "",
    persist_source_pos_shape: tuple | None = None,
    lumping: str = "equal_div_inc",
) -> torch.Tensor:
    """Lumped nodal area from Abaqus CAX mesh (physical coordinates)."""
    from utils.abaqus_mesh_weight import (
        LUMPING_TO_METHOD,
        compute_node_weight_mesh_batch,
        normalize_mesh_lumping,
        resolve_case_ids_from_npz,
    )

    mode = normalize_mesh_lumping(lumping)
    method = LUMPING_TO_METHOD[mode]
    mask = _resolve_weight_mask(
        pos_phys, cache_mask, padding_enabled=padding_enabled, padding_value=padding_value
    )
    case_ids = resolve_case_ids_from_npz(npz_dir)
    if case_ids.shape[0] != pos_phys.shape[0]:
        raise RuntimeError(
            f"npz case count {case_ids.shape[0]} != pos B={pos_phys.shape[0]} "
            f"(npz_dir={npz_dir})"
        )
    print(
        f"[*] computing node_weight from Abaqus mesh "
        f"(instance={instance_name}, lumping={mode}, physical pos) ..."
    )
    q_np, _infos = compute_node_weight_mesh_batch(
        pos_phys.detach().cpu().numpy(),
        None if mask is None else mask.detach().cpu().numpy(),
        case_ids,
        abaqus_root=abaqus_root,
        instance_name=instance_name,
        tol=tol,
        n_workers=n_workers,
        show_progress=True,
        lumping=mode,
    )
    nw = torch.from_numpy(q_np)
    print(
        f"[*] mesh node_weight: shape={tuple(nw.shape)}, "
        f"min={nw[nw > 0].min().item() if (nw > 0).any() else 0:.3e}, "
        f"max={nw.max().item():.3e}, mean={nw.mean().item():.3e}"
    )
    if persist_path:
        persist_node_weight_to_cache(
            persist_path,
            nw,
            source_pos_shape=persist_source_pos_shape or tuple(pos_phys.shape),
            method=method,
            extra_meta={
                "node_weight_instance": instance_name,
                "node_weight_abaqus_root": os.path.abspath(abaqus_root),
                "node_weight_npz_dir": os.path.abspath(npz_dir),
                "node_weight_tol": float(tol),
                "node_weight_lumping": mode,
            },
        )
    return nw


def mesh_node_weight_sidecar_path(cache_path: str, lumping: str) -> str:
    """Sidecar next to cache.pt: ``*_nw_{equal|hrz|equal_div_inc}.pt``."""
    from utils.abaqus_mesh_weight import LUMPING_TO_METHOD, normalize_mesh_lumping

    mode = normalize_mesh_lumping(lumping)
    want = LUMPING_TO_METHOD[mode]
    tag = want.replace("abaqus_cax_", "").replace("_lumped_area", "")
    base = cache_path[:-3] if str(cache_path).endswith(".pt") else str(cache_path)
    return f"{base}_nw_{tag}.pt"


def _save_mesh_nw_sidecar_if_missing(
    nw: torch.Tensor,
    *,
    cache_path: str,
    lumping: str,
    method: str,
) -> str:
    """Write ``*_nw_{lumping}.pt`` once. Never writes the shared cache.pt."""
    side = mesh_node_weight_sidecar_path(cache_path, lumping)
    if os.path.isfile(side):
        return side
    torch.save(
        {
            "node_weight": nw.detach().cpu().float(),
            "method": method,
            "lumping": lumping,
            "cache_path": os.path.abspath(cache_path),
        },
        side,
    )
    print(f"    saved mesh q sidecar: {side}")
    return side


def load_or_compute_mesh_node_weight(
    pos: torch.Tensor,
    *,
    lumping: str,
    cache_path: str,
    cached_node_weight: torch.Tensor | None = None,
    cache_meta: dict | None = None,
    cache_mask: torch.Tensor | None = None,
    padding_enabled: bool = False,
    padding_value: float = 0.0,
    abaqus_root: str = "",
    npz_dir: str = "",
    instance_name: str = "QJT-1",
) -> torch.Tensor:
    """Load mesh q matching ``lumping``: exact cache meta, else sidecar, else compute.

    Does not write back into the shared ``cache.pt`` (avoids overwriting other formulas).
    Always keeps a per-formula sidecar so later cache overwrites cannot poison eval.
    """
    from utils.abaqus_mesh_weight import LUMPING_TO_METHOD, normalize_mesh_lumping

    mode = normalize_mesh_lumping(lumping)
    want = LUMPING_TO_METHOD[mode]
    meta = cache_meta if isinstance(cache_meta, dict) else {}
    if (
        cached_node_weight is not None
        and tuple(cached_node_weight.shape) == tuple(pos.shape[:2])
        and str(meta.get("node_weight_method", "")) == want
    ):
        print(f"    reuse cache node_weight for lumping={mode} ({want})")
        nw = cached_node_weight.float()
        _save_mesh_nw_sidecar_if_missing(
            nw, cache_path=cache_path, lumping=mode, method=want
        )
        return nw
    side = mesh_node_weight_sidecar_path(cache_path, mode)
    if os.path.isfile(side):
        print(f"    load mesh q sidecar: {side}")
        blob = torch.load(side, map_location="cpu", weights_only=False)
        nw = blob["node_weight"] if isinstance(blob, dict) else blob
        if tuple(nw.shape) != tuple(pos.shape[:2]):
            raise RuntimeError(
                f"sidecar {side} shape {tuple(nw.shape)} != pos {tuple(pos.shape[:2])}"
            )
        return nw.float()
    print(f"    computing mesh q lumping={mode} (no matching cache/sidecar) ...")
    nw = compute_node_weight_from_abaqus_mesh(
        pos,
        cache_mask=cache_mask,
        padding_enabled=padding_enabled,
        padding_value=padding_value,
        abaqus_root=abaqus_root,
        npz_dir=npz_dir,
        instance_name=instance_name,
        persist_path="",
        lumping=mode,
    )
    _save_mesh_nw_sidecar_if_missing(
        nw, cache_path=cache_path, lumping=mode, method=want
    )
    return nw.float()


def frame_subsample_active(args) -> bool:
    stride = int(getattr(args, "frame_stride", 1))
    offset = int(getattr(args, "frame_offset", 0))
    max_frames = int(getattr(args, "max_frames", 0))
    return stride > 1 or offset > 0 or max_frames > 0


def default_coord_norm_path(
    cache_path: str,
    *,
    frame_stride: int = 1,
    frame_offset: int = 0,
    max_frames: int = 0,
    user_path: str = "",
) -> str:
    """Derive legacy cache-side coord_norm path; subsampled runs use a separate file."""
    if not cache_path:
        return user_path
    auto_default = cache_path.replace(".pt", "_coord_norm.pt")
    if user_path and user_path != auto_default:
        return user_path
    if frame_stride <= 1 and frame_offset <= 0 and max_frames <= 0:
        return auto_default
    base = cache_path.replace(".pt", "")
    tag = f"_coord_norm_sub_s{frame_stride}_o{frame_offset}"
    if max_frames > 0:
        tag += f"_n{max_frames}"
    return f"{base}{tag}.pt"


def coord_norm_path_in_run_dir(
    run_dir: str,
    *,
    frame_stride: int = 1,
    frame_offset: int = 0,
    max_frames: int = 0,
) -> str:
    """Per-run coord_norm file under run_dir (avoids cross-config overwrite)."""
    run_dir = os.path.abspath(run_dir)
    if frame_stride <= 1 and frame_offset <= 0 and max_frames <= 0:
        return os.path.join(run_dir, "coord_norm.pt")
    tag = f"coord_norm_sub_s{frame_stride}_o{frame_offset}"
    if max_frames > 0:
        tag += f"_n{max_frames}"
    return os.path.join(run_dir, f"{tag}.pt")


def resolve_coord_norm_path(args) -> str:
    """Resolve coord_norm path.

    Preference:
      1) explicit --coord_norm_path (not the legacy cache auto path)
      2) {run_dir}/coord_norm.pt when run_dir is set
      3) legacy {cache}_coord_norm.pt
    """
    user = str(getattr(args, "coord_norm_path", "") or "").strip()
    cache_path = str(getattr(args, "cache_path", "") or "")
    run_dir = str(getattr(args, "run_dir", "") or "").strip()
    stride = int(getattr(args, "frame_stride", 1))
    offset = int(getattr(args, "frame_offset", 0))
    max_frames = int(getattr(args, "max_frames", 0))

    cache_auto = ""
    if cache_path:
        cache_auto = default_coord_norm_path(
            cache_path,
            frame_stride=stride,
            frame_offset=offset,
            max_frames=max_frames,
            user_path="",
        )

    if user and user != cache_auto:
        path = user
    elif run_dir:
        path = coord_norm_path_in_run_dir(
            run_dir,
            frame_stride=stride,
            frame_offset=offset,
            max_frames=max_frames,
        )
    elif user:
        path = user
    else:
        path = cache_auto

    setattr(args, "coord_norm_path", path)
    return path


def apply_frame_subsample(
    pos: torch.Tensor,
    fx: torch.Tensor,
    y: torch.Tensor,
    *,
    frame_stride: int = 1,
    frame_offset: int = 0,
    max_frames: int = 0,
    case_params: torch.Tensor | None = None,
    sato_indices: dict | None = None,
    case_t: torch.Tensor | None = None,
    case_names: Sequence[str] | None = None,
    geom_id: torch.Tensor | None = None,
    node_weight: torch.Tensor | None = None,
    cache_mask: torch.Tensor | None = None,
    cache_path: str = "",
) -> tuple:
    """Temporal subsample along batch dim B."""
    stride = int(frame_stride)
    offset = int(frame_offset)
    max_frames = int(max_frames)
    if stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {stride}")
    if offset < 0:
        raise ValueError(f"frame_offset must be >= 0, got {offset}")

    if stride == 1 and offset == 0 and max_frames <= 0:
        cn = list(case_names) if case_names is not None else None
        return pos, fx, y, case_params, sato_indices, case_t, cn, geom_id, node_weight, cache_mask, False

    b = int(pos.shape[0])
    idx = torch.arange(offset, b, stride, dtype=torch.long)
    if max_frames > 0:
        idx = idx[:max_frames]
    if idx.numel() == 0:
        raise ValueError(
            f"frame subsample empty: B={b}, stride={stride}, offset={offset}, max_frames={max_frames}"
        )

    pos = pos[idx]
    fx = fx[idx]
    y = y[idx]
    if case_params is not None:
        case_params = case_params[idx]
    if sato_indices is not None:
        sato_indices = {key: val[idx] for key, val in sato_indices.items()}
    if case_t is not None:
        case_t = case_t[idx]
    if geom_id is not None:
        geom_id = geom_id[idx]
    if node_weight is not None:
        node_weight = node_weight[idx]
    if cache_mask is not None:
        cache_mask = cache_mask[idx]
    cn = None
    if case_names is not None:
        cn = [case_names[i] for i in idx.tolist()]

    label = os.path.basename(cache_path) if cache_path else "cache"
    t_msg = ""
    if case_t is not None and case_t.numel() > 0:
        t_msg = f", t=[{case_t.min():.4f}, {case_t.max():.4f}]"
    print(
        f"[*] frame subsample ({label}): B {b} -> {int(idx.numel())} "
        f"(stride={stride}, offset={offset}, max_frames={max_frames or 'all'}){t_msg}"
    )
    return pos, fx, y, case_params, sato_indices, case_t, cn, geom_id, node_weight, cache_mask, True


def _load_cache(cache_path: str) -> dict:
    if not cache_path:
        raise ValueError("cache_path is empty. Provide --cache_path pointing to a pre-built cache.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"cache_path not found: {cache_path}")

    try:
        obj = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(cache_path, map_location="cpu")

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported cache format: expected dict, got {type(obj)} at {cache_path}")
    return obj


def load_keti_data(args) -> Tuple:
    """
    Load KETI tensors from a pre-built cache.

    Returns:
      pos, fx, y, input_dim, output_dim, case_params, sato_indices, case_t, cached_node_weight
      cached_node_weight is optional metadata from cache; training/eval recompute q
      on normalized pos when use_node_weight=1.
    """
    if getattr(args, "rebuild_cache", False):
        raise RuntimeError(
            "This environment is configured for cache-only training. "
            "Please build cache offline (outside the training container) and pass --cache_path."
        )

    cache_path = getattr(args, "cache_path", "")
    cache = _load_cache(cache_path)

    for k in ("pos", "fx", "y"):
        if k not in cache:
            raise KeyError(f"cache missing key '{k}'. Found keys={sorted(cache.keys())}")

    pos = cache["pos"]
    fx = cache["fx"]
    y = cache["y"]
    case_params = cache.get("case_params", None)
    node_weight = cache.get("node_weight", None)
    cache_mask = cache.get("mask", None)

    case_t = None
    if "t" in cache:
        case_t = cache["t"].float().view(-1)
    elif fx.shape[-1] >= 1:
        case_t = fx[:, 0, 0].float().clone()

    empty_fx = bool(int(getattr(args, "empty_fx", 0)))
    if empty_fx:
        fx = torch.zeros(pos.shape[0], pos.shape[1], 0, dtype=pos.dtype)
        if case_t is None:
            raise ValueError("empty_fx=1 需要 cache 中的 't' 字段（time_input 用）")
        print(f"[*] empty_fx=1: fx 置为空 (B,N,0)，时间来自 case_t，范围 [{case_t.min():.4f}, {case_t.max():.4f}]")

    # Move SDF (or other field channel) from fx → pos before norms / model.
    if bool(int(getattr(args, "sdf_to_pos", 0))):
        if empty_fx or fx.shape[-1] == 0:
            raise ValueError("sdf_to_pos=1 需要非空 fx（不能与 empty_fx=1 同开）")
        ch = int(getattr(args, "sdf_fx_channel", -1))
        c = fx.shape[-1]
        if ch < 0:
            ch = c + ch
        if ch < 0 or ch >= c:
            raise ValueError(f"sdf_fx_channel={getattr(args, 'sdf_fx_channel', -1)} out of range for fx C={c}")
        sdf = fx[:, :, ch : ch + 1].contiguous()
        keep = [i for i in range(c) if i != ch]
        fx = fx[:, :, keep].contiguous() if keep else fx[:, :, :0]
        pos = torch.cat([pos, sdf], dim=-1)
        print(
            f"[*] sdf_to_pos=1: moved fx[..., {ch}] → pos (now C={pos.shape[-1]}); "
            f"fx remaining C={fx.shape[-1]}"
        )

    # Load SATO indices if requested
    sato_indices = None
    if bool(int(getattr(args, "use_sato", 0))):
        sato_index_path = getattr(args, "sato_index_path", "")
        if not sato_index_path and getattr(args, "cache_path", ""):
            sato_index_path = args.cache_path.replace(".pt", "_sato_index.pt")
            setattr(args, "sato_index_path", sato_index_path)

        if sato_index_path and os.path.exists(sato_index_path):
            sato_indices = torch.load(sato_index_path, map_location="cpu")
            cfg_gs = float(getattr(args, "sato_grid_size", 0.01))
            file_gs = sato_indices.get("grid_size", None) if isinstance(sato_indices, dict) else None
            print(f"[*] Loaded SATO indices from {sato_index_path}")
            if file_gs is not None:
                print(f"    index grid_size={file_gs} (config sato_grid_size={cfg_gs})")
                if abs(float(file_gs) - cfg_gs) > 1e-12:
                    print(
                        f"[!] Warning: sato_grid_size mismatch: config={cfg_gs} vs index file={file_gs}. "
                        f"Rebuild index with --grid_size {cfg_gs} if this is unintended."
                    )
            else:
                print(f"    config sato_grid_size={cfg_gs} (index file has no grid_size metadata)")
        else:
            cfg_gs = float(getattr(args, "sato_grid_size", 0.01))
            raise FileNotFoundError(
                f"use_sato=1 but SATO indices not found: {sato_index_path or '(empty path)'}. "
                f"Precompute with:\n"
                f"  PYTHONPATH=. conda run -n upt python -u "
                f"scripts/transolver/tools/cache/precompute_serialization.py "
                f"--cache_path <cache.pt> --grid_size {cfg_gs}"
            )

    if not (isinstance(pos, torch.Tensor) and isinstance(fx, torch.Tensor) and isinstance(y, torch.Tensor)):
        raise TypeError("cache tensors must be torch.Tensor: pos/fx/y")

    if pos.ndim != 3 or fx.ndim != 3 or y.ndim != 3:
        raise ValueError(f"Expected 3D tensors. Got pos={pos.shape}, fx={fx.shape}, y={y.shape}")
    if pos.shape[:2] != fx.shape[:2] or pos.shape[:2] != y.shape[:2]:
        raise ValueError(f"Shape mismatch among pos/fx/y: pos={pos.shape}, fx={fx.shape}, y={y.shape}")

    pos = pos.float()
    fx = fx.float()
    y = y.float()
    if node_weight is not None:
        node_weight = node_weight.float()
        if node_weight.shape != pos.shape[:2]:
            raise ValueError(
                f"cache node_weight shape {tuple(node_weight.shape)} != pos {tuple(pos.shape[:2])}"
            )
    if cache_mask is not None:
        cache_mask = cache_mask.bool()
        if cache_mask.shape != pos.shape[:2]:
            raise ValueError(
                f"cache mask shape {tuple(cache_mask.shape)} != pos {tuple(pos.shape[:2])}"
            )

    case_names = cache.get("case_names", None)
    geom_id = cache.get("geom_id", None)
    if geom_id is not None:
        geom_id = geom_id.long().view(-1)

    # Persist only when full cache is used (no subsample / no degenerate drop).
    can_persist = (
        int(getattr(args, "frame_stride", 1)) <= 1
        and int(getattr(args, "frame_offset", 0)) <= 0
        and int(getattr(args, "max_frames", 0)) <= 0
    )

    pos, fx, y, case_params, sato_indices, case_t, case_names, geom_id, node_weight, cache_mask, subsampled = (
        apply_frame_subsample(
            pos,
            fx,
            y,
            frame_stride=int(getattr(args, "frame_stride", 1)),
            frame_offset=int(getattr(args, "frame_offset", 0)),
            max_frames=int(getattr(args, "max_frames", 0)),
            case_params=case_params,
            sato_indices=sato_indices,
            case_t=case_t,
            case_names=case_names,
            geom_id=geom_id,
            node_weight=node_weight,
            cache_mask=cache_mask,
            cache_path=cache_path,
        )
    )
    setattr(args, "frame_subsample_applied", subsampled)
    if subsampled:
        can_persist = False

    filter_deg = bool(int(getattr(args, "filter_degenerate_y", 1)))
    deg_eps = float(getattr(args, "y_degenerate_eps", 1e-6))
    pos, fx, y, case_params, sato_indices, case_t, num_removed, keep_idx, geom_id, node_weight, cache_mask = (
        filter_degenerate_y_cases(
            pos,
            fx,
            y,
            eps=deg_eps,
            case_params=case_params,
            sato_indices=sato_indices,
            case_t=case_t,
            case_names=case_names,
            geom_id=geom_id,
            node_weight=node_weight,
            cache_mask=cache_mask,
            cache_path=cache_path,
            filter_enabled=filter_deg,
        )
    )
    if keep_idx is not None:
        can_persist = False
    if keep_idx is not None and case_names is not None:
        setattr(args, "filtered_case_names", [case_names[i] for i in keep_idx.tolist()])
    elif case_names is not None:
        setattr(args, "filtered_case_names", list(case_names))
    if geom_id is not None:
        setattr(args, "filtered_geom_id", geom_id)

    setattr(args, "filtered_cache_mask", cache_mask)
    setattr(args, "can_persist_node_weight", bool(can_persist))
    # Model preprocess / unified_pos extra dims follow actual pos width.
    setattr(args, "space_dim", int(pos.shape[-1]))

    input_dim = int(fx.shape[-1])
    output_dim = int(y.shape[-1])
    return pos, fx, y, input_dim, output_dim, case_params, sato_indices, case_t, node_weight


def get_train_val_test_indices(
    num_samples: int,
    train_split: float = 0.8,
    val_split: float = 0.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Deterministic shuffle split. Same prefix as ``get_train_test_indices`` when seed matches.

    ``val_split<=0``: remainder after train is test, ``val_idx`` is None.
    ``val_split>0``: train / val / rest=test (e.g. 0.8 / 0.1 / 0.1).
    """
    import numpy as np

    if not 0.0 < float(train_split) < 1.0:
        raise ValueError(f"train_split must be in (0,1), got {train_split}")
    val_split = float(val_split or 0.0)
    if val_split < 0.0:
        raise ValueError(f"val_split must be >=0, got {val_split}")

    indices = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    n_train = int(num_samples * float(train_split))
    if n_train < 1 or n_train >= num_samples:
        raise ValueError(f"n_train={n_train} invalid for B={num_samples}")
    train_idx = torch.tensor(indices[:n_train].tolist(), dtype=torch.long)
    if val_split <= 0.0:
        test_idx = torch.tensor(indices[n_train:].tolist(), dtype=torch.long)
        return train_idx, None, test_idx

    n_val = int(num_samples * val_split)
    if n_val < 1 or n_train + n_val >= num_samples:
        raise ValueError(
            f"val_split={val_split} gives n_val={n_val} with n_train={n_train}, B={num_samples}"
        )
    val_idx = torch.tensor(indices[n_train : n_train + n_val].tolist(), dtype=torch.long)
    test_idx = torch.tensor(indices[n_train + n_val :].tolist(), dtype=torch.long)
    if int(test_idx.numel()) < 1:
        raise ValueError("empty test split")
    return train_idx, val_idx, test_idx


def get_train_test_indices(num_samples: int, train_split: float = 0.8, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate deterministic random indices for train/test split.
    
    Args:
        num_samples: Total number of samples (B)
        train_split: Proportion of samples for training
        seed: Random seed for reproducibility
        
    Returns:
        train_idx: torch.Tensor of training indices
        test_idx: torch.Tensor of testing indices
    """
    train_idx, _, test_idx = get_train_val_test_indices(
        num_samples, train_split=train_split, val_split=0.0, seed=seed
    )
    return train_idx, test_idx


def get_train_test_indices_by_case(
    case_names: Sequence[str],
    train_cases: Sequence[str],
    test_cases: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split sample indices by case name (all timesteps of a case stay together)."""
    train_set = set(str(c) for c in train_cases)
    test_set = set(str(c) for c in test_cases)
    overlap = train_set & test_set
    if overlap:
        raise ValueError("train_cases and test_cases overlap: {}".format(sorted(overlap)))

    train_idx, test_idx = [], []
    unknown = set()
    for i, name in enumerate(case_names):
        name = str(name)
        if name in train_set:
            train_idx.append(i)
        elif name in test_set:
            test_idx.append(i)
        else:
            unknown.add(name)
    if unknown:
        raise ValueError(
            "Samples from cases not in train/test lists: {} (train={}, test={})".format(
                sorted(unknown), sorted(train_set), sorted(test_set)
            )
        )
    if not train_idx:
        raise ValueError("No training samples after case split")
    if not test_idx:
        raise ValueError("No test samples after case split")

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )


def _normalize_geom_id_list(values) -> list[int] | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        if len(values) == 0:
            return None
        return [int(x) for x in values]
    return [int(values)]


def get_train_test_indices_by_geom_id(
    geom_id: torch.Tensor,
    *,
    train_geom_ids: Sequence[int] | None = None,
    test_geom_ids: Sequence[int] | None = None,
    train_split: float = 0.8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """
    按几何 id 划分：同一 geom 的全部时间帧只进 train 或 test，互不重叠。
    """
    import numpy as np

    gid = geom_id.long().view(-1)
    unique = sorted(int(x) for x in torch.unique(gid).tolist())
    if len(unique) < 2:
        raise ValueError("geom split 需要至少 2 个几何，当前 unique geom_id={}".format(unique))

    train_list = _normalize_geom_id_list(train_geom_ids)
    test_list = _normalize_geom_id_list(test_geom_ids)

    if train_list is not None and test_list is not None:
        train_set = set(train_list)
        test_set = set(test_list)
    else:
        ids = list(unique)
        rng = np.random.default_rng(int(seed))
        rng.shuffle(ids)
        n_train = max(1, int(len(ids) * float(train_split)))
        n_train = min(n_train, len(ids) - 1)
        train_set = set(ids[:n_train])
        test_set = set(ids[n_train:])

    overlap = train_set & test_set
    if overlap:
        raise ValueError("train_geom_ids 与 test_geom_ids 重叠: {}".format(sorted(overlap)))

    unknown = set(unique) - train_set - test_set
    if unknown:
        raise ValueError("存在未分配的几何 geom_id: {}".format(sorted(unknown)))

    train_idx, test_idx = [], []
    for i, g in enumerate(gid.tolist()):
        if g in train_set:
            train_idx.append(i)
        elif g in test_set:
            test_idx.append(i)
    if not train_idx or not test_idx:
        raise ValueError(
            "geom split 后 train/test 为空: train_geoms={}, test_geoms={}".format(
                sorted(train_set), sorted(test_set)
            )
        )

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
        sorted(train_set),
        sorted(test_set),
    )


def resolve_train_test_indices(
    cache: dict,
    train_split: float = 0.8,
    seed: int = 42,
    num_samples: int | None = None,
    case_names: Sequence[str] | None = None,
    geom_id: torch.Tensor | None = None,
    split_by: str = "random",
    train_geom_ids: Sequence[int] | None = None,
    test_geom_ids: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """
    Resolve train/test indices.

    Priority:
      1. split_by=geom + geom_id（或 cache.geom_id）
      2. metadata train_cases + test_cases with per-sample case_names
      3. random split by train_split ratio
    """
    meta = cache.get("metadata", {}) or {}
    split_by = str(split_by or "random").lower()

    if geom_id is None and "geom_id" in cache:
        geom_id = cache["geom_id"]
        if isinstance(geom_id, torch.Tensor):
            geom_id = geom_id.long().view(-1)

    train_geom_ids = _normalize_geom_id_list(train_geom_ids)
    test_geom_ids = _normalize_geom_id_list(test_geom_ids)

    if split_by == "geom":
        if train_geom_ids is None:
            train_geom_ids = _normalize_geom_id_list(meta.get("train_geom_ids"))
        if test_geom_ids is None:
            test_geom_ids = _normalize_geom_id_list(meta.get("test_geom_ids"))
        if geom_id is None:
            raise ValueError("split_by=geom 需要 cache 中的 geom_id 字段")
        train_idx, test_idx, train_geoms, test_geoms = get_train_test_indices_by_geom_id(
            geom_id,
            train_geom_ids=train_geom_ids,
            test_geom_ids=test_geom_ids,
            train_split=train_split,
            seed=seed,
        )
        mode = "geom_split({} train / {} test geoms, {} train / {} test frames, seed={})".format(
            len(train_geoms),
            len(test_geoms),
            int(train_idx.numel()),
            int(test_idx.numel()),
            seed,
        )
        mode += " train_geom_ids={} test_geom_ids={}".format(train_geoms, test_geoms)
        return train_idx, test_idx, mode

    train_cases = meta.get("train_cases", None)
    test_cases = meta.get("test_cases", None)
    cn = case_names if case_names is not None else cache.get("case_names", None)

    if split_by == "case" and train_cases is not None and test_cases is not None and cn is not None:
        train_idx, test_idx = get_train_test_indices_by_case(cn, train_cases, test_cases)
        mode = "case_split({} train / {} test cases, {} train / {} test samples)".format(
            len(train_cases),
            len(test_cases),
            int(train_idx.numel()),
            int(test_idx.numel()),
        )
        return train_idx, test_idx, mode

    if train_cases is not None and test_cases is not None and cn is not None:
        train_idx, test_idx = get_train_test_indices_by_case(cn, train_cases, test_cases)
        mode = "case_split({} train / {} test cases, {} train / {} test samples)".format(
            len(train_cases),
            len(test_cases),
            int(train_idx.numel()),
            int(test_idx.numel()),
        )
        return train_idx, test_idx, mode

    cache_b = int(cache["pos"].shape[0])
    b = int(num_samples) if num_samples is not None else cache_b
    train_idx, test_idx = get_train_test_indices(b, train_split=train_split, seed=seed)
    mode = "random_split(ratio={:.3f}, seed={})".format(train_split, seed)
    if num_samples is not None and num_samples != cache_b:
        mode += ", after_degenerate_y_filter"
    if split_by not in ("random", "case", "geom"):
        mode += " (unknown split_by={!r}, fallback random)".format(split_by)
    return train_idx, test_idx, mode


def resolve_train_val_test_indices(
    cache: dict,
    train_split: float = 0.8,
    val_split: float = 0.0,
    seed: int = 42,
    num_samples: int | None = None,
    case_names: Sequence[str] | None = None,
    geom_id: torch.Tensor | None = None,
    split_by: str = "random",
    train_geom_ids: Sequence[int] | None = None,
    test_geom_ids: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, str]:
    """Like ``resolve_train_test_indices``, plus optional val.

    ``val_split<=0``: ``val_idx`` is None (epoch eval uses test, old protocol).
    ``val_split>0``: only ``split_by=random``; train prefix matches the old 8:2 train set.
    """
    val_split = float(val_split or 0.0)
    train_idx, test_idx, mode = resolve_train_test_indices(
        cache,
        train_split=train_split,
        seed=seed,
        num_samples=num_samples,
        case_names=case_names,
        geom_id=geom_id,
        split_by=split_by,
        train_geom_ids=train_geom_ids,
        test_geom_ids=test_geom_ids,
    )
    if val_split <= 0.0:
        return train_idx, None, test_idx, mode
    if not str(mode).startswith("random_split"):
        raise ValueError(
            f"val_split={val_split} requires split_by=random, got {mode}"
        )
    cache_b = int(cache["pos"].shape[0])
    b = int(num_samples) if num_samples is not None else cache_b
    train_idx, val_idx, test_idx = get_train_val_test_indices(
        b, train_split=train_split, val_split=val_split, seed=seed
    )
    test_ratio = 1.0 - float(train_split) - val_split
    mode = "random_split(train={:.3f}, val={:.3f}, test={:.3f}, seed={})".format(
        float(train_split), val_split, test_ratio, seed
    )
    if num_samples is not None and num_samples != cache_b:
        mode += ", after_degenerate_y_filter"
    return train_idx, val_idx, test_idx, mode

