"""Abaqus CAX6/CAX8 mesh → lumped nodal area (node_weight).

Lumping modes (per element, A_e = ∫_e dA via Gauss + |J|):
  hrz            — HRZ diagonal ∫ N_i², scaled to A_e; global sum = area
  equal          — A_e/n_nodes per node; global sum = area
  equal_div_inc  — equal then /incidence; q_i = mean_{e∋i}(A_e/n_e)

Default for new training: equal_div_inc (see NODE_WEIGHT_METHOD_MESH).
Eval / paper RelL2 should prefer equal (area-conserving quadrature).

Case index mapping for geo24_len caches:
  cache[i] ↔ i-th Case_*.npz under npz_dir (sorted by case id)
           ↔ Case_{case_idx}/SHDX.inp
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Canonical metadata strings written into cache.pt
NODE_WEIGHT_METHOD_MESH_HRZ = "abaqus_cax_hrz_lumped_area"
NODE_WEIGHT_METHOD_MESH_EQUAL = "abaqus_cax_equal_lumped_area"
NODE_WEIGHT_METHOD_MESH_EQUAL_DIV_INC = "abaqus_cax_equal_div_inc_lumped_area"
# Backward-compatible alias used by trainers that only say "mesh"
NODE_WEIGHT_METHOD_MESH = NODE_WEIGHT_METHOD_MESH_EQUAL_DIV_INC

LUMPING_TO_METHOD = {
    "hrz": NODE_WEIGHT_METHOD_MESH_HRZ,
    "equal": NODE_WEIGHT_METHOD_MESH_EQUAL,
    "equal_div_inc": NODE_WEIGHT_METHOD_MESH_EQUAL_DIV_INC,
}
METHOD_TO_LUMPING = {v: k for k, v in LUMPING_TO_METHOD.items()}

# ---------------------------------------------------------------------------
# INP parse
# ---------------------------------------------------------------------------

_FLOAT_NODE_RE = re.compile(
    r"^\s*(\d+)\s*,\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)\s*,\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)"
)
_INT_RE = re.compile(r"\d+")
_KW_NAME_RE = re.compile(r"NAME\s*=\s*([^,\s]+)", re.I)
_KW_PART_RE = re.compile(r"PART\s*=\s*([^,\s]+)", re.I)


def _kw_value(line: str, pat: re.Pattern[str]) -> str | None:
    m = pat.search(line)
    return m.group(1).strip() if m else None


def _find_star_block(
    text: list[str],
    star: str,
    name: str,
    end_star: str,
) -> tuple[int, int]:
    """Return [start, end) of *STAR, name=NAME … *END STAR. Name match is exact."""
    want = name.upper()
    head = f"*{star.upper()}"
    end_head = f"*{end_star.upper()}"
    start = None
    for i, line in enumerate(text):
        u = line.lstrip().upper()
        if not u.startswith(head):
            continue
        if _kw_value(line, _KW_NAME_RE) and _kw_value(line, _KW_NAME_RE).upper() == want:
            start = i
            break
    if start is None:
        raise RuntimeError(f"{star} {name} not found")
    end = len(text)
    for j in range(start + 1, len(text)):
        u = text[j].lstrip().upper()
        if u.startswith(end_head) or u.startswith(head):
            end = j
            break
    return start, end


def _parse_nodes_elems(
    lines: list[str],
) -> tuple[dict[int, np.ndarray], list[tuple[str, int, tuple[int, ...]]]]:
    nodes: dict[int, np.ndarray] = {}
    elems: list[tuple[str, int, tuple[int, ...]]] = []
    mode = None
    cur_etype = None
    for line in lines:
        s = line.strip()
        if not s or s.startswith("**"):
            continue
        if s.startswith("*"):
            up = s.upper()
            if up.startswith("*NODE"):
                mode = "node"
                cur_etype = None
            elif up.startswith("*ELEMENT"):
                mode = "elem"
                m = re.search(r"TYPE\s*=\s*([A-Za-z0-9]+)", up)
                cur_etype = m.group(1) if m else "UNKNOWN"
            else:
                mode = None
                cur_etype = None
            continue
        if mode == "node":
            m = _FLOAT_NODE_RE.match(line)
            if not m:
                continue
            lab = int(m.group(1))
            nodes[lab] = np.array(
                [float(m.group(2)), float(m.group(3))], dtype=np.float64
            )
        elif mode == "elem" and cur_etype is not None:
            nums = [int(x) for x in _INT_RE.findall(line)]
            if len(nums) < 2:
                continue
            elems.append((cur_etype, nums[0], tuple(nums[1:])))
    return nodes, elems


def _pack_mesh(
    nodes: dict[int, np.ndarray],
    elems: list[tuple[str, int, tuple[int, ...]]],
    *,
    source: str,
) -> dict:
    labels = np.array(sorted(nodes.keys()), dtype=np.int64)
    pos = np.stack([nodes[int(l)] for l in labels], axis=0)
    lab_to_idx = {int(l): i for i, l in enumerate(labels)}
    n_cax6 = sum(1 for t, _, _ in elems if t.upper().startswith("CAX6"))
    n_cax8 = sum(1 for t, _, _ in elems if t.upper().startswith("CAX8"))
    return {
        "labels": labels,
        "pos": pos,
        "lab_to_idx": lab_to_idx,
        "elems": elems,
        "n_cax6": n_cax6,
        "n_cax8": n_cax8,
        "n_other": len(elems) - n_cax6 - n_cax8,
        "source": source,
    }


def parse_instance_inp(
    inp_path: str | Path,
    instance_name: str = "QJT-1",
) -> dict:
    """Parse mesh for an instance: orphan *Instance nodes, or referenced *Part."""
    inp_path = Path(inp_path)
    text = inp_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    try:
        start, end = _find_star_block(text, "INSTANCE", instance_name, "END INSTANCE")
    except RuntimeError as e:
        raise RuntimeError(f"instance {instance_name} not found in {inp_path}") from e

    nodes, elems = _parse_nodes_elems(text[start + 1 : end])
    if nodes:
        return _pack_mesh(nodes, elems, source=f"instance:{instance_name}")

    part_name = _kw_value(text[start], _KW_PART_RE)
    if not part_name:
        raise RuntimeError(
            f"no nodes in instance {instance_name} and no part= in {inp_path}"
        )
    try:
        p0, p1 = _find_star_block(text, "PART", part_name, "END PART")
    except RuntimeError as e:
        raise RuntimeError(
            f"part {part_name} (from instance {instance_name}) not found in {inp_path}"
        ) from e
    nodes, elems = _parse_nodes_elems(text[p0 + 1 : p1])
    if not nodes:
        raise RuntimeError(
            f"no nodes parsed in {instance_name} (part={part_name}) of {inp_path}"
        )
    return _pack_mesh(nodes, elems, source=f"part:{part_name}")


def resolve_case_ids_from_npz(npz_dir: str | Path) -> np.ndarray:
    """Sorted Case_*.npz → case_idx array aligned with cache row order."""
    npz_dir = Path(npz_dir)
    files = sorted(
        npz_dir.glob("Case_*.npz"),
        key=lambda p: int(re.search(r"Case_(\d+)", p.name).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"no Case_*.npz under {npz_dir}")
    return np.array(
        [int(re.search(r"Case_(\d+)", p.name).group(1)) for p in files],
        dtype=np.int64,
    )


# ---------------------------------------------------------------------------
# Shape functions + Gauss quadrature (planar dA)
# ---------------------------------------------------------------------------

# 3×3 Gauss–Legendre on [-1,1]
_G3 = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)], dtype=np.float64)
_W3 = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)

# 6-point triangle rule on (ξ,η) with ξ>=0, η>=0, ξ+η<=1  (degree 4)
# Dunavant / standard textbook weights (area coords mapped to ξ=L2, η=L3)
_TRI6_XI = np.array(
    [
        0.445948490915965,
        0.445948490915965,
        0.108103018168070,
        0.091576213509771,
        0.091576213509771,
        0.816847572980459,
    ],
    dtype=np.float64,
)
_TRI6_ETA = np.array(
    [
        0.445948490915965,
        0.108103018168070,
        0.445948490915965,
        0.091576213509771,
        0.816847572980459,
        0.091576213509771,
    ],
    dtype=np.float64,
)
_TRI6_W = np.array(
    [
        0.223381589678011,
        0.223381589678011,
        0.223381589678011,
        0.109951743655322,
        0.109951743655322,
        0.109951743655322,
    ],
    dtype=np.float64,
) * 0.5  # rule weights sum to 1/2 (= ref triangle area)


def _cax8_N(xi: float, eta: float) -> np.ndarray:
    """Abaqus CAX8 serendipity shape functions, node order 1..8."""
    N = np.empty(8, dtype=np.float64)
    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0)
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0)
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0)
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0)
    N[4] = 0.5 * (1.0 - xi * xi) * (1.0 - eta)
    N[5] = 0.5 * (1.0 + xi) * (1.0 - eta * eta)
    N[6] = 0.5 * (1.0 - xi * xi) * (1.0 + eta)
    N[7] = 0.5 * (1.0 - xi) * (1.0 - eta * eta)
    return N


def _cax8_dN(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    dxi = np.empty(8, dtype=np.float64)
    deta = np.empty(8, dtype=np.float64)
    dxi[0] = 0.25 * (1.0 - eta) * (2.0 * xi + eta)
    deta[0] = 0.25 * (1.0 - xi) * (xi + 2.0 * eta)
    dxi[1] = 0.25 * (1.0 - eta) * (2.0 * xi - eta)
    deta[1] = 0.25 * (1.0 + xi) * (-xi + 2.0 * eta)
    dxi[2] = 0.25 * (1.0 + eta) * (2.0 * xi + eta)
    deta[2] = 0.25 * (1.0 + xi) * (xi + 2.0 * eta)
    dxi[3] = 0.25 * (1.0 + eta) * (2.0 * xi - eta)
    deta[3] = 0.25 * (1.0 - xi) * (-xi + 2.0 * eta)
    dxi[4] = -xi * (1.0 - eta)
    deta[4] = -0.5 * (1.0 - xi * xi)
    dxi[5] = 0.5 * (1.0 - eta * eta)
    deta[5] = -(1.0 + xi) * eta
    dxi[6] = -xi * (1.0 + eta)
    deta[6] = 0.5 * (1.0 - xi * xi)
    dxi[7] = -0.5 * (1.0 - eta * eta)
    deta[7] = -(1.0 - xi) * eta
    return dxi, deta


def _cax6_N(xi: float, eta: float) -> np.ndarray:
    """Quadratic triangle, Abaqus CAX6 order: corners 1–3, mids 4–6."""
    L1 = 1.0 - xi - eta
    L2 = xi
    L3 = eta
    N = np.empty(6, dtype=np.float64)
    N[0] = L1 * (2.0 * L1 - 1.0)
    N[1] = L2 * (2.0 * L2 - 1.0)
    N[2] = L3 * (2.0 * L3 - 1.0)
    N[3] = 4.0 * L1 * L2
    N[4] = 4.0 * L2 * L3
    N[5] = 4.0 * L3 * L1
    return N


def _cax6_dN(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    # L1=1-ξ-η, L2=ξ, L3=η
    L1 = 1.0 - xi - eta
    L2 = xi
    L3 = eta
    dxi = np.empty(6, dtype=np.float64)
    deta = np.empty(6, dtype=np.float64)
    # N1 = L1(2L1-1); ∂L1/∂ξ=-1, ∂L1/∂η=-1
    dxi[0] = (-1.0) * (4.0 * L1 - 1.0)
    deta[0] = (-1.0) * (4.0 * L1 - 1.0)
    dxi[1] = 4.0 * L2 - 1.0
    deta[1] = 0.0
    dxi[2] = 0.0
    deta[2] = 4.0 * L3 - 1.0
    dxi[3] = 4.0 * (L1 - L2)
    deta[3] = 4.0 * (-L2)
    dxi[4] = 4.0 * L3
    deta[4] = 4.0 * L2
    dxi[5] = 4.0 * (-L3)
    deta[5] = 4.0 * (L1 - L3)
    return dxi, deta


def _elem_area_cax8(xy: np.ndarray) -> float:
    """Planar area of one CAX8 via 3×3 Gauss + |J|."""
    area = 0.0
    for a, wa in enumerate(_W3):
        for b, wb in enumerate(_W3):
            xi, eta = float(_G3[a]), float(_G3[b])
            dxi, deta = _cax8_dN(xi, eta)
            J = np.stack([dxi @ xy, deta @ xy], axis=0)
            det = abs(float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]))
            area += wa * wb * det
    return float(area)


def _elem_area_cax6(xy: np.ndarray) -> float:
    """Planar area of one CAX6 via 6-point triangle rule + |J|."""
    area = 0.0
    for k in range(_TRI6_W.shape[0]):
        xi, eta = float(_TRI6_XI[k]), float(_TRI6_ETA[k])
        dxi, deta = _cax6_dN(xi, eta)
        J = np.stack([dxi @ xy, deta @ xy], axis=0)
        det = abs(float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]))
        area += float(_TRI6_W[k]) * det
    return float(area)


def _equal_cax8(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Equal-share lumped area on one CAX8. xy: [8,2] → (q[8], A_e)."""
    area = _elem_area_cax8(xy)
    return np.full(8, area / 8.0, dtype=np.float64), area


def _equal_cax6(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Equal-share lumped area on one CAX6. xy: [6,2] → (q[6], A_e)."""
    area = _elem_area_cax6(xy)
    return np.full(6, area / 6.0, dtype=np.float64), area


def _hrz_cax8(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """HRZ lumped nodal area on one CAX8. xy: [8,2] → (q[8], A_e)."""
    diag = np.zeros(8, dtype=np.float64)
    area = 0.0
    for a, wa in enumerate(_W3):
        for b, wb in enumerate(_W3):
            xi, eta = float(_G3[a]), float(_G3[b])
            N = _cax8_N(xi, eta)
            dxi, deta = _cax8_dN(xi, eta)
            J = np.stack([dxi @ xy, deta @ xy], axis=0)
            det = abs(float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]))
            w = wa * wb * det
            diag += w * (N * N)
            area += w
    s = float(diag.sum())
    if s <= 0.0 or area <= 0.0:
        return np.full(8, area / 8.0, dtype=np.float64), area
    return diag * (area / s), area


def _hrz_cax6(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """HRZ lumped nodal area on one CAX6. xy: [6,2] → (q[6], A_e)."""
    diag = np.zeros(6, dtype=np.float64)
    area = 0.0
    for k in range(_TRI6_W.shape[0]):
        xi, eta = float(_TRI6_XI[k]), float(_TRI6_ETA[k])
        w0 = float(_TRI6_W[k])
        N = _cax6_N(xi, eta)
        dxi, deta = _cax6_dN(xi, eta)
        J = np.stack([dxi @ xy, deta @ xy], axis=0)
        det = abs(float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]))
        w = w0 * det
        diag += w * (N * N)
        area += w
    s = float(diag.sum())
    if s <= 0.0 or area <= 0.0:
        return np.full(6, area / 6.0, dtype=np.float64), area
    return diag * (area / s), area


def normalize_mesh_lumping(lumping: str | None) -> str:
    """Map aliases → hrz | equal | equal_div_inc."""
    s = str(lumping or "equal_div_inc").strip().lower()
    aliases = {
        "hrz": "hrz",
        "equal": "equal",
        "equal_share": "equal",
        "equal_div_inc": "equal_div_inc",
        "equal_div_incidence": "equal_div_inc",
        "divinc": "equal_div_inc",
        "div_inc": "equal_div_inc",
    }
    if s not in aliases:
        raise ValueError(
            f"unknown mesh lumping={lumping!r}; "
            f"expected one of {sorted(set(aliases))}"
        )
    return aliases[s]


def lumped_area_from_cax_mesh(
    pos: np.ndarray,
    elems: list[tuple[str, int, tuple[int, ...]]],
    lab_to_idx: dict[int, int],
    *,
    lumping: str = "equal_div_inc",
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """Assemble nodal areas from CAX elements under the chosen lumping."""
    mode = normalize_mesh_lumping(lumping)
    pos = np.asarray(pos, dtype=np.float64)
    n = pos.shape[0]
    q = np.zeros(n, dtype=np.float64)
    inc = np.zeros(n, dtype=np.int32)
    area_sum = 0.0
    n_used6 = 0
    n_used8 = 0
    n_skip = 0

    for et, _eid, conn in elems:
        up = et.upper()
        try:
            idxs = [lab_to_idx[c] for c in conn]
        except KeyError:
            n_skip += 1
            continue
        if up.startswith("CAX8"):
            if len(idxs) < 8:
                n_skip += 1
                continue
            idxs = idxs[:8]
            if mode == "hrz":
                qe, ae = _hrz_cax8(pos[idxs])
            else:
                qe, ae = _equal_cax8(pos[idxs])
            q[idxs] += qe
            for i in idxs:
                inc[i] += 1
            area_sum += float(ae)
            n_used8 += 1
        elif up.startswith("CAX6"):
            if len(idxs) < 6:
                n_skip += 1
                continue
            idxs = idxs[:6]
            if mode == "hrz":
                qe, ae = _hrz_cax6(pos[idxs])
            else:
                qe, ae = _equal_cax6(pos[idxs])
            q[idxs] += qe
            for i in idxs:
                inc[i] += 1
            area_sum += float(ae)
            n_used6 += 1
        else:
            n_skip += 1

    touched = inc > 0
    if mode == "equal_div_inc":
        q[touched] /= inc[touched].astype(np.float64)
    orphan = ~touched
    if np.any(orphan):
        q[orphan] = eps
    stats = {
        "area_sum": area_sum,
        "q_sum": float(q.sum()),
        "n_cax6_used": n_used6,
        "n_cax8_used": n_used8,
        "n_elem_skipped": n_skip,
        "n_orphan_floored": int(orphan.sum()),
        "inc_min": int(inc[touched].min()) if np.any(touched) else 0,
        "inc_max": int(inc.max()) if n else 0,
        "lumping": mode,
        "method": LUMPING_TO_METHOD[mode],
    }
    return q, stats


def map_q_to_cache(
    cache_pos: np.ndarray,
    gt_pos: np.ndarray,
    q_gt: np.ndarray,
    *,
    tol: float = 1e-3,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """Scatter INP-ordered q onto cache valid-node order (nearest / identity)."""
    cache_pos = np.asarray(cache_pos, dtype=np.float64)
    gt_pos = np.asarray(gt_pos, dtype=np.float64)
    n = cache_pos.shape[0]
    q = np.zeros(n, dtype=np.float64)

    same_n = n == gt_pos.shape[0]
    if same_n:
        d_ord = np.linalg.norm(cache_pos - gt_pos, axis=1)
        if float(d_ord.max()) <= tol:
            q[:] = q_gt
            return q, {
                "match_mode": "identity_order",
                "d_max": float(d_ord.max()),
                "n_unmatched": 0,
            }

    from scipy.spatial import cKDTree

    dist, j = cKDTree(gt_pos).query(cache_pos, k=1)
    dist = np.atleast_1d(np.asarray(dist, dtype=np.float64))
    j = np.atleast_1d(np.asarray(j, dtype=np.int64))
    ok = dist <= tol
    q[ok] = q_gt[j[ok]]
    if np.any(~ok):
        q[~ok] = eps
    return q, {
        "match_mode": "knn",
        "d_max": float(dist.max()),
        "d_max_matched": float(dist[ok].max()) if ok.any() else float("nan"),
        "n_unmatched": int((~ok).sum()),
        "match_frac": float(ok.mean()),
    }


# ---------------------------------------------------------------------------
# Per-case / batch
# ---------------------------------------------------------------------------

def resolve_shdx_inp(
    case_idx: int,
    abaqus_root: str | Path,
    inp_dir: str | Path | None = None,
) -> Path:
    """Case_*/SHDX.inp 或 cache_npz/Case_{idx}.inp（提取 ODB 后备份）。"""
    p0 = Path(abaqus_root) / f"Case_{case_idx}" / "SHDX.inp"
    if p0.is_file():
        return p0
    if inp_dir is not None:
        p1 = Path(inp_dir) / f"Case_{case_idx}.inp"
        if p1.is_file():
            return p1
    raise FileNotFoundError(
        f"no SHDX.inp for case {case_idx} under {abaqus_root}/Case_{case_idx}/ "
        f"or {inp_dir}/Case_{case_idx}.inp"
    )


def compute_q_for_case(
    cache_pos_valid: np.ndarray,
    *,
    case_idx: int,
    abaqus_root: str | Path,
    instance_name: str = "QJT-1",
    tol: float = 1e-3,
    eps: float = 1e-12,
    lumping: str = "equal_div_inc",
    inp_dir: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    inp = resolve_shdx_inp(case_idx, abaqus_root, inp_dir=inp_dir)
    gt = parse_instance_inp(inp, instance_name=instance_name)
    q_gt, st = lumped_area_from_cax_mesh(
        gt["pos"], gt["elems"], gt["lab_to_idx"], lumping=lumping, eps=eps
    )
    q, mst = map_q_to_cache(cache_pos_valid, gt["pos"], q_gt, tol=tol, eps=eps)
    info = {
        "case_idx": int(case_idx),
        "n_cache": int(cache_pos_valid.shape[0]),
        "n_inp": int(gt["pos"].shape[0]),
        "n_cax6": int(gt["n_cax6"]),
        "n_cax8": int(gt["n_cax8"]),
        **st,
        **mst,
    }
    return q, info


def _worker_one(args):
    """Picklable worker: (i, case_idx, pos_valid, abaqus_root, instance, tol, lumping, inp_dir)."""
    i, case_idx, pos_valid, abaqus_root, instance_name, tol, lumping, inp_dir = args
    q, info = compute_q_for_case(
        pos_valid,
        case_idx=case_idx,
        abaqus_root=abaqus_root,
        instance_name=instance_name,
        tol=tol,
        lumping=lumping,
        inp_dir=inp_dir,
    )
    return i, q, info


def compute_node_weight_mesh_batch(
    pos: np.ndarray,
    mask: np.ndarray | None,
    case_ids: np.ndarray,
    *,
    abaqus_root: str | Path,
    instance_name: str = "QJT-1",
    tol: float = 1e-3,
    n_workers: int = 8,
    show_progress: bool = True,
    lumping: str = "equal_div_inc",
    inp_dir: str | Path | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Compute [B,N] mesh node weights for a padded cache batch."""
    mode = normalize_mesh_lumping(lumping)
    pos = np.asarray(pos, dtype=np.float64)
    B, N, _ = pos.shape
    if mask is None:
        mask = np.ones((B, N), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    case_ids = np.asarray(case_ids, dtype=np.int64)
    if case_ids.shape[0] != B:
        raise ValueError(f"case_ids length {case_ids.shape[0]} != B={B}")

    q_all = np.zeros((B, N), dtype=np.float32)
    infos: list[dict | None] = [None] * B

    tasks = []
    for i in range(B):
        m = mask[i]
        tasks.append(
            (
                i,
                int(case_ids[i]),
                pos[i, m],
                str(abaqus_root),
                instance_name,
                float(tol),
                mode,
                str(inp_dir) if inp_dir is not None else None,
            )
        )

    if n_workers <= 1:
        iterator = (_worker_one(t) for t in tasks)
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=B, desc=f"mesh nw ({mode})")
            except ImportError:
                pass
        for i, q, info in iterator:
            m = mask[i]
            q_all[i, m] = q.astype(np.float32)
            infos[i] = info
            if show_progress and n_workers <= 1 and (i + 1) % 50 == 0:
                print(
                    f"    [{i+1}/{B}] Case_{info['case_idx']} "
                    f"area={info['area_sum']:.4g} match={info.get('match_mode')}"
                )
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_worker_one, t) for t in tasks]
            done = 0
            for fut in as_completed(futs):
                i, q, info = fut.result()
                m = mask[i]
                q_all[i, m] = q.astype(np.float32)
                infos[i] = info
                done += 1
                if show_progress and (done % 50 == 0 or done == B):
                    print(
                        f"    [{done}/{B}] Case_{info['case_idx']} "
                        f"area={info['area_sum']:.4g} "
                        f"unmatched={info.get('n_unmatched', 0)}"
                    )

    return q_all, infos  # type: ignore[return-value]
