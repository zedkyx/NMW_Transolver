#!/usr/bin/env python3
"""Fill cache.pt node_weight from Abaqus CAX6/CAX8 mesh (equal-share / incidence).

Example:
  python scripts/transolver/tools/compute_mesh_node_weight.py \
    --cache_path /home/wang/zyx/cache_geo24_len/full/u0_s2_n6420/qjt_1_full.pt \
    --abaqus_root /home/wang/keti_abaqus_geo24_len \
    --npz_dir /home/wang/keti_abaqus_geo24_len/cache_npz \
    --instance_name QJT-1 \
    --n_workers 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

# repo root on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.abaqus_mesh_weight import (
    NODE_WEIGHT_METHOD_MESH,
    compute_node_weight_mesh_batch,
    resolve_case_ids_from_npz,
)


def main():
    p = argparse.ArgumentParser(description="Compute mesh-based node_weight into cache.pt")
    p.add_argument("--cache_path", type=str, required=True)
    p.add_argument("--abaqus_root", type=str, default="/home/wang/keti_abaqus_geo24_len")
    p.add_argument(
        "--npz_dir",
        type=str,
        default="/home/wang/keti_abaqus_geo24_len/cache_npz",
        help="Case_*.npz dir; sorted case ids align with cache rows",
    )
    p.add_argument("--instance_name", type=str, default="QJT-1")
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--n_workers", type=int, default=8)
    p.add_argument(
        "--lumping",
        type=str,
        default="equal_div_inc",
        help="hrz | equal | equal_div_inc (divinc 训练默认)",
    )
    p.add_argument(
        "--inp_dir",
        type=str,
        default="",
        help="Case_*.inp 备份目录；默认与 --npz_dir 相同（geo27 提取后备份）",
    )
    p.add_argument(
        "--sidecar_only",
        type=int,
        default=0,
        help="1: 只写 *_nw_{lumping}.pt sidecar，不改写 cache.pt",
    )
    p.add_argument("--max_cases", type=int, default=0, help="0=all; >0 for debug subset")
    p.add_argument(
        "--dry_run",
        type=int,
        default=0,
        help="1: compute stats only on first case, do not write",
    )
    args = p.parse_args()
    inp_dir = args.inp_dir.strip() or args.npz_dir

    print(f"[1] load cache: {args.cache_path}")
    t0 = time.time()
    cache = torch.load(args.cache_path, map_location="cpu", weights_only=False)
    pos = cache["pos"]
    mask = cache.get("mask", None)
    B, N, _ = pos.shape
    print(f"    pos={tuple(pos.shape)}  has_mask={mask is not None}  ({time.time()-t0:.1f}s)")

    print(f"[2] resolve case ids from {args.npz_dir}")
    case_ids = resolve_case_ids_from_npz(args.npz_dir)
    if case_ids.shape[0] != B:
        raise RuntimeError(
            f"npz case count {case_ids.shape[0]} != cache B={B}. "
            "Check --npz_dir matches how this cache was built."
        )
    print(
        f"    case_ids: n={len(case_ids)}  "
        f"min={int(case_ids.min())} max={int(case_ids.max())}  "
        f"first3={case_ids[:3].tolist()}"
    )

    if args.max_cases and args.max_cases > 0:
        B_use = min(int(args.max_cases), B)
        print(f"[!] max_cases={B_use} (debug subset; will NOT write full cache)")
    else:
        B_use = B

    pos_np = pos[:B_use].numpy()
    mask_np = None if mask is None else mask[:B_use].numpy()
    ids_use = case_ids[:B_use]

    if args.dry_run:
        from utils.abaqus_mesh_weight import compute_q_for_case

        m0 = mask_np[0] if mask_np is not None else np.ones(N, dtype=bool)
        q0, info = compute_q_for_case(
            pos_np[0, m0],
            case_idx=int(ids_use[0]),
            abaqus_root=args.abaqus_root,
            instance_name=args.instance_name,
            tol=args.tol,
        )
        print("[dry_run] first case:", info)
        print(
            f"    q min/mean/max/sum = {q0.min():.4g} / {q0.mean():.4g} / "
            f"{q0.max():.4g} / {q0.sum():.4g}"
        )
        return

    print(
        f"[3] compute mesh node_weight  B={B_use}  workers={args.n_workers}  "
        f"instance={args.instance_name}"
    )
    t1 = time.time()
    q_np, infos = compute_node_weight_mesh_batch(
        pos_np,
        mask_np,
        ids_use,
        abaqus_root=args.abaqus_root,
        instance_name=args.instance_name,
        tol=args.tol,
        n_workers=args.n_workers,
        show_progress=True,
        lumping=args.lumping,
        inp_dir=inp_dir,
    )
    print(f"    done in {time.time()-t1:.1f}s")

    unmatched = sum(int(inf.get("n_unmatched", 0)) for inf in infos)
    orphans = sum(int(inf.get("n_orphan_floored", 0)) for inf in infos)
    areas = np.array([inf["area_sum"] for inf in infos], dtype=np.float64)
    print(
        f"[4] summary: total_unmatched_nodes={unmatched}  "
        f"total_orphan_floored={orphans}"
    )
    print(
        f"    area_sum per case: min={areas.min():.4g}  "
        f"mean={areas.mean():.4g}  max={areas.max():.4g}"
    )
    bad = [inf for inf in infos if inf.get("n_unmatched", 0) > 0]
    if bad:
        print(f"    [!] {len(bad)} cases with unmatched nodes, e.g. {bad[0]}")

    if B_use != B:
        print("[!] skip write: max_cases subset only")
        return

    if int(args.sidecar_only) == 1:
        from scripts.transolver.keti_data_loader import mesh_node_weight_sidecar_path
        from utils.abaqus_mesh_weight import LUMPING_TO_METHOD, normalize_mesh_lumping

        mode = normalize_mesh_lumping(args.lumping)
        method = LUMPING_TO_METHOD[mode]
        side = mesh_node_weight_sidecar_path(args.cache_path, mode)
        nw = torch.from_numpy(q_np).float()
        torch.save(
            {
                "node_weight": nw,
                "method": method,
                "lumping": mode,
                "cache_path": os.path.abspath(args.cache_path),
            },
            side,
        )
        print(f"[5] sidecar_only -> {side}  shape={tuple(nw.shape)}  lumping={mode}")
        print(f"[done] total {time.time()-t0:.1f}s")
        return

    print(f"[5] write node_weight + metadata -> {args.cache_path}")
    nw = torch.from_numpy(q_np).float()
    cache["node_weight"] = nw
    cache["case_idx"] = torch.from_numpy(case_ids.copy()).long()
    meta = cache.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    meta["node_weight_method"] = NODE_WEIGHT_METHOD_MESH
    meta["node_weight_instance"] = args.instance_name
    meta["node_weight_abaqus_root"] = os.path.abspath(args.abaqus_root)
    meta["node_weight_npz_dir"] = os.path.abspath(args.npz_dir)
    meta["node_weight_tol"] = float(args.tol)
    cache["metadata"] = meta
    torch.save(cache, args.cache_path)
    print(
        f"    saved node_weight {tuple(nw.shape)}  "
        f"method={NODE_WEIGHT_METHOD_MESH}  "
        f"q>0 min/max={float(nw[nw > 0].min()):.4g}/{float(nw.max()):.4g}"
    )
    print(f"[done] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
