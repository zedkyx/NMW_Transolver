import os
import sys
import json
import random
import subprocess
import time
import shutil
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch.utils.data.distributed import DistributedSampler
except Exception:  # pragma: no cover
    DistributedSampler = None

from torch.nn.parallel import DistributedDataParallel as DDP

from utils import UnitTransformer, IdentityTransformer
from scripts.transolver.keti_data_loader import (
    load_keti_data,
    resolve_train_val_test_indices,
    resolve_coord_norm_path,
    frame_subsample_active,
    _load_cache,
    cache_method_matches,
    compute_node_weight_from_abaqus_mesh,
    compute_node_weight_from_normalized_pos,
    load_or_compute_mesh_node_weight,
    normalize_node_weight_max_edge,
    resolve_node_weight_method,
    resolve_node_weight_pathways,
)
from scripts.transolver.keti_utils import (
    plot_loss_curves,
    plot_loss_curves_channels,
    plot_samples,
    save_loss_curves_npy,
)

from scripts.transolver.train.ddp_utils import init_ddp_if_needed, cleanup_ddp
from scripts.transolver.train.modeling import MultiNetWrapper
from scripts.transolver.train.loops import run_epoch, build_padding_mask
from scripts.transolver.train.checkpointing import save_state_dict, maybe_load_checkpoint, load_state_dict_into
from scripts.transolver.train.scheduling import (
    build_lr_scheduler,
    current_lr,
    is_meaningful_improvement,
)
from scripts.transolver.train.seed_utils import (
    make_dataloader_generator,
    make_worker_init_fn,
    set_global_seed,
)


def _patch_run_config_coord_norm(
    run_root: str,
    coord_norm_path: str,
    *,
    mesh_instance_name: str = "",
) -> None:
    """Write resolved coord_norm_path / mesh_instance_name into run_dir/config.yaml."""
    cfg_path = os.path.join(run_root, "config.yaml")
    if not os.path.isfile(cfg_path):
        return
    try:
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            return
        data = cfg.get("data")
        if not isinstance(data, dict):
            data = {}
            cfg["data"] = data
        data["coord_norm_path"] = coord_norm_path
        if mesh_instance_name:
            data["mesh_instance_name"] = str(mesh_instance_name)
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        extra = f", mesh_instance_name={mesh_instance_name}" if mesh_instance_name else ""
        print(f"[*] Wrote coord_norm_path{extra} into {cfg_path}")
    except Exception as e:
        print(f"[!] failed to patch config.yaml with coord_norm_path: {e}")


def _get_script_root() -> str:
    # scripts/transolver/train/ -> scripts/transolver/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _run_auto_eval(args, run_root: str, ckpt_dir: str) -> None:
    if not int(getattr(args, "auto_eval", 1)):
        return
    if getattr(args, "smoke_test", False) or getattr(args, "eval_only", False):
        return

    ckpt_kind = str(getattr(args, "auto_eval_ckpt", "best_eval") or "best_eval").strip().lower()
    if ckpt_kind in ("latest", "last"):
        ckpt_path = os.path.join(ckpt_dir, f"{args.save_name}_latest.pt")
    else:
        ckpt_path = os.path.join(ckpt_dir, f"{args.save_name}_best_eval.pt")
    if not os.path.isfile(ckpt_path):
        print(f"[!] auto_eval skipped: checkpoint not found: {ckpt_path}")
        return

    repo_root = _repo_root()
    eval_script = os.path.join(repo_root, "scripts", "transolver", "eval", "eval_keti.py")
    if not os.path.isfile(eval_script):
        print(f"[!] auto_eval skipped: eval script not found: {eval_script}")
        return

    config_path = os.path.join(run_root, "config.yaml")
    if not os.path.isfile(config_path):
        config_path = getattr(args, "config", "") or ""

    cmd = [
        sys.executable,
        eval_script,
        "--run_dir",
        run_root,
        "--load_ckpt",
        ckpt_path,
        "--save_name",
        str(args.save_name),
        "--num_plot",
        str(int(getattr(args, "num_plot", 3))),
    ]
    if config_path:
        cmd.extend(["--config", config_path])
    if getattr(args, "cache_path", ""):
        cmd.extend(["--cache_path", str(args.cache_path)])
    out_cols = getattr(args, "output_cols", None)
    if out_cols:
        if isinstance(out_cols, (list, tuple)):
            cmd.extend(["--output_cols", *[str(c) for c in out_cols]])
        else:
            cmd.extend(["--output_cols", str(out_cols)])
    if getattr(args, "coord_norm_path", ""):
        cmd.extend(["--coord_norm_path", str(args.coord_norm_path)])
    inst = str(getattr(args, "mesh_instance_name", "") or "")
    if inst:
        cmd.extend(["--mesh_instance_name", inst])
    if int(getattr(args, "high_stress_analysis", 1)):
        cmd.extend(["--high_stress_analysis", "1"])
    if int(getattr(args, "per_channel_stress", 1)):
        cmd.extend(["--per_channel_stress", "1"])
    cmd.extend(
        [
            "--stress_channel_idx",
            str(int(getattr(args, "stress_channel_idx", 0))),
            "--stress_lower_ratio",
            str(float(getattr(args, "stress_lower_ratio", 2 / 3))),
            "--stress_upper_ratio",
            str(float(getattr(args, "stress_upper_ratio", 1.0))),
            "--error_thresholds",
            str(getattr(args, "error_thresholds", "0.01,0.05,0.10,0.20")),
        ]
    )

    log_path = os.path.join(run_root, "logs", "eval.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(f"[*] auto_eval: starting eval_keti.py (log -> {log_path})")
    print(f"[*] auto_eval: ckpt={ckpt_path}")
    print(f"[*] auto_eval: eval output -> {os.path.join(run_root, 'eval')}/")
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    with open(log_path, "w", encoding="utf-8") as log_f:
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_f.write(line)
            log_f.flush()
    ret = proc.wait()
    if ret != 0:
        print(f"[!] auto_eval failed (exit {ret}), see {log_path}")
    else:
        print(f"[✓] auto_eval finished, log: {log_path}")


def _teardown_viz_cuda(device: torch.device) -> None:
    """Close matplotlib figures and sync CUDA to reduce rare exit-time segfaults."""
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass
    if device.type == "cuda":
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass


def run(args) -> None:
    """
    Unified KETI trainer for both DP and DDP.
    DP/DDP differences are limited to:
    - sampler (DistributedSampler)
    - model wrapping (DDP vs DataParallel/None)
    - rank0-only logging/saving/viz
    """
    # DP: CUDA_VISIBLE_DEVICES must be set BEFORE the first torch.cuda call.
    # Otherwise a shell/scheduler may have restricted visibility (e.g. only GPU 0),
    # and later assignment in this function is ignored — DataParallel then only uses cuda:0.
    if not getattr(args, "ddp", False):
        gpu_str = str(getattr(args, "gpu", "") or "").strip()
        if gpu_str:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

    # Strict CUDA determinism (slow kernels). Default off: seed lock is enough.
    want_det = bool(int(getattr(args, "deterministic", 0)))
    if want_det:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    ctx = init_ddp_if_needed(args)
    device = ctx.device
    is_main = ctx.is_main_process

    split_seed = int(getattr(args, "seed", 42))
    train_seed = int(getattr(args, "train_seed", 42))
    rank = int(getattr(ctx, "rank", 0) or 0)
    # Data-phase lock uses split seed; training RNG is re-locked to train_seed later.
    set_global_seed(
        split_seed,
        deterministic=want_det,
        rank=rank,
    )
    effective_train_seed = int(train_seed) + rank
    if is_main:
        print(
            f"[*] Split seed={split_seed} train_seed={train_seed} "
            f"effective_train={effective_train_seed} (rank={rank}) "
            f"deterministic={int(want_det)} "
            f"({'strict CUDA kernels' if want_det else 'init + shuffle'})"
        )

    # --- Output Path Alignment ---
    run_dir = getattr(args, "run_dir", "")
    if run_dir:
        # If run_dir is explicitly provided (recommended), use it directly
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        plot_dir = os.path.join(run_dir, "plot")
        run_root = run_dir
    else:
        # Legacy/Fallback: Use script-relative paths (might be symlinks)
        script_root = _get_script_root()
        ckpt_dir = os.path.join(script_root, "checkpoints")
        plot_dir = os.path.join(script_root, "plot")
        # Try to resolve run_root from symlink
        if os.path.islink(ckpt_dir):
            run_root = os.path.dirname(os.path.realpath(ckpt_dir))
        else:
            run_root = script_root

    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        
        # Backup config.yaml to the run root directory
        config_path = getattr(args, "config", "")
        if config_path and os.path.exists(config_path):
            target_config = os.path.join(run_root, "config.yaml")
            shutil.copy(config_path, target_config)
            print(f"[*] Configuration backed up to: {target_config}")
            print(f"[*] Checkpoints will be saved to: {ckpt_dir}")
            print(f"[*] Plots will be saved to: {plot_dir}")

    # DP: use logical device indices 0..N-1 (after CUDA_VISIBLE_DEVICES remapping)
    gpu_ids = []
    num_gpus = 0
    if not ctx.enabled:
        if device.type == "cuda":
            num_gpus = int(torch.cuda.device_count())
            gpu_ids = list(range(num_gpus))
            if is_main:
                vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                print(f"Using {num_gpus} visible GPU(s), DataParallel device_ids={gpu_ids}")
                if vis:
                    print(f"    CUDA_VISIBLE_DEVICES={vis}")

    # --- Data Loading ---
    pos, fx, y, input_dim, output_dim, _, sato_indices, case_t, cached_node_weight = load_keti_data(args)

    time_input = bool(int(getattr(args, "time_input", 0)))
    use_node_weight = bool(int(getattr(args, "use_node_weight", 0)))
    recompute_node_weight = bool(int(getattr(args, "recompute_node_weight", 0)))
    nw_method = resolve_node_weight_method(args)  # None | delaunay | mesh
    nw_path = resolve_node_weight_pathways(args)
    node_weight_in_attn = nw_path["attn"]
    node_weight_in_norm_fx = nw_path["norm_fx"]
    node_weight_in_norm_y = nw_path["norm_y"]
    node_weight_in_norm = nw_path["norm"]  # fx or y (legacy aggregate)
    node_weight_in_loss = nw_path["loss"]
    if time_input:
        if case_t is None:
            raise ValueError("time_input=1 需要 cache 中的 't' 字段")
        if getattr(args, "model", "Transolver_1D") == "SATO":
            raise ValueError("SATO 不支持 Time_Input，请改用 model: Transolver_1D")
        if is_main:
            print(f"[*] time_input=1: 使用 timestep_embedding(T)，T 来自 cache.t，范围 [{case_t.min():.4f}, {case_t.max():.4f}]")

    padding_enabled = bool(int(getattr(args, "padding", 0)))
    padding_value = float(getattr(args, "padding_value", 0.0))

    # --- Split ---
    cache = _load_cache(getattr(args, "cache_path", ""))
    B = pos.shape[0]
    case_names = getattr(args, "filtered_case_names", None)
    geom_id = getattr(args, "filtered_geom_id", None)
    train_idx, val_idx, test_idx, split_mode = resolve_train_val_test_indices(
        cache,
        train_split=float(getattr(args, "train_split", 0.8)),
        val_split=float(getattr(args, "val_split", 0.0)),
        seed=split_seed,
        num_samples=B,
        case_names=case_names,
        geom_id=geom_id,
        split_by=str(getattr(args, "split_by", "random")),
        train_geom_ids=getattr(args, "train_geom_ids", None),
        test_geom_ids=getattr(args, "test_geom_ids", None),
    )
    # Epoch loop eval: val when 8:1:1, else the old test split.
    loop_idx = val_idx if val_idx is not None else test_idx
    heldout_test_idx = test_idx
    if frame_subsample_active(args) and is_main:
        split_mode += ", frame_subsample(stride={}, offset={}, max_frames={})".format(
            int(getattr(args, "frame_stride", 1)),
            int(getattr(args, "frame_offset", 0)),
            int(getattr(args, "max_frames", 0)),
        )
    if is_main:
        print(f"[*] Dataset split: {split_mode}")
        print(
            f"    train={len(train_idx)} val={len(val_idx) if val_idx is not None else 0} "
            f"test={len(heldout_test_idx)}"
        )
        if val_idx is not None:
            print("    epoch EvalRel → val; auto_eval RelL2 → held-out test")
        else:
            print("    val_split=0: epoch EvalRel uses test (old protocol)")
        if str(getattr(args, "split_by", "random")).lower() == "geom":
            if meta := cache.get("metadata", {}):
                if meta.get("train_cases") is not None:
                    print(f"    train cases: {meta.get('train_cases')}")
                    print(f"    test cases : {meta.get('test_cases')}")
                if meta.get("train_geom_ids") is not None:
                    print(f"    train_geom_ids: {meta.get('train_geom_ids')}")
                    print(f"    test_geom_ids : {meta.get('test_geom_ids')}")
    pos_train, fx_train, y_train = pos[train_idx], fx[train_idx], y[train_idx]
    pos_test, fx_test, y_test = pos[loop_idx], fx[loop_idx], y[loop_idx]
    nw_train = nw_test = None
    if padding_enabled:
        mask_train = build_padding_mask(pos_train, fx_train, y_train, padding_value)
        mask_test = build_padding_mask(pos_test, fx_test, y_test, padding_value)
        if is_main:
            total_train_points = mask_train.numel()
            valid_train_points = mask_train.sum().item()
            padding_train_points = total_train_points - valid_train_points
            total_test_points = mask_test.numel()
            valid_test_points = mask_test.sum().item()
            padding_test_points = total_test_points - valid_test_points
            print(f"[*] Padding enabled (padding_value={padding_value}), will build mask for valid points")
            print(f"    Train set: {valid_train_points} valid points, {padding_train_points} padding points ({100*padding_train_points/total_train_points:.2f}%)")
            eval_name = "Val set" if val_idx is not None else "Test set"
            print(f"    {eval_name}: {valid_test_points} valid points, {padding_test_points} padding points ({100*padding_test_points/total_test_points:.2f}%)")
    else:
        mask_train = None
        mask_test = None

    # --- Normalization (coords, fx, y) ---
    def get_normalizer(data, norm_type, *, node_weight=None, mask=None):
        if norm_type == "min-max":
            if mask is not None:
                return IdentityTransformer(data[mask])
            return IdentityTransformer(data)
        return UnitTransformer(data, node_weight=node_weight, mask=mask)

    coord_stats_path = resolve_coord_norm_path(args)
    subsample_active = frame_subsample_active(args)
    if is_main:
        if subsample_active:
            print(
                f"[*] frame subsample: normalizers from subsampled train set -> {coord_stats_path}"
            )
        else:
            print(f"[*] coord_norm_path: {coord_stats_path}")
        if run_root:
            _patch_run_config_coord_norm(
                run_root,
                coord_stats_path,
                mesh_instance_name=str(getattr(args, "mesh_instance_name", "") or ""),
            )

    load_pos_from_file = (
        coord_stats_path
        and os.path.exists(coord_stats_path)
        and not subsample_active
    )
    if load_pos_from_file:
        try:
            saved = torch.load(coord_stats_path, weights_only=True)
        except TypeError:
            saved = torch.load(coord_stats_path)
        pos_min = saved["min"].view(1, 1, -1)
        pos_max = saved["max"].view(1, 1, -1)
        pos_normalizer = IdentityTransformer(
            min_val=pos_min.squeeze(0).squeeze(0),
            max_val=pos_max.squeeze(0).squeeze(0),
        )
    else:
        # 计算坐标归一化参数（如果padding_enabled，只使用有效点）
        if padding_enabled and mask_train is not None:
            # mask_train: [B, N], pos_train: [B, N, 2]
            # 使用mask过滤出所有有效点
            pos_train_valid = pos_train[mask_train]  # [N_valid, 2]
            if is_main:
                total_points = pos_train.shape[0] * pos_train.shape[1]
                valid_points = pos_train_valid.shape[0]
                padding_points = total_points - valid_points
                print(f"[*] Computing pos normalizer ({args.pos_norm_type}) from valid points only: {valid_points} valid points, {padding_points} padding points excluded ({100*padding_points/total_points:.2f}%)")
            pos_normalizer = get_normalizer(pos_train_valid, args.pos_norm_type)
        else:
            pos_normalizer = get_normalizer(pos_train, args.pos_norm_type)
        
        if coord_stats_path and is_main:
            os.makedirs(os.path.dirname(coord_stats_path), exist_ok=True)
            # 保存坐标归一化参数
            norm_stats = {
                "min": pos_normalizer.min.detach().cpu(),
                "max": pos_normalizer.max.detach().cpu()
            }
            torch.save(norm_stats, coord_stats_path)
            print(f"[*] Saved coord normalization stats to {coord_stats_path}")

    pos_all_n = pos_normalizer.encode(pos)
    node_weight = None
    if use_node_weight and nw_method is not None:
        cache_mask = getattr(args, "filtered_cache_mask", None)
        max_edge = normalize_node_weight_max_edge(getattr(args, "node_weight_max_edge", None))
        persist = bool(int(getattr(args, "persist_node_weight", 0)))
        can_persist = bool(getattr(args, "can_persist_node_weight", False))
        cache_meta = cache.get("metadata", {}) if isinstance(cache, dict) else {}
        if not isinstance(cache_meta, dict):
            cache_meta = {}
        cache_nw_method = str(cache_meta.get("node_weight_method", ""))
        shape_ok = (
            cached_node_weight is not None
            and tuple(cached_node_weight.shape) == tuple(pos.shape[:2])
        )
        fwd_lump = str(getattr(args, "node_weight_lumping", "") or "").strip().lower()
        method_ok = cache_method_matches(cache_nw_method, nw_method)
        can_reuse = shape_ok and method_ok and not recompute_node_weight and not fwd_lump
        if nw_method == "mesh" and fwd_lump:
            if is_main:
                print(
                    f"[*] forward mesh q lumping={fwd_lump} "
                    f"(exact cache meta or sidecar; persist_cache={int(persist and can_persist)})"
                )
            node_weight = load_or_compute_mesh_node_weight(
                pos,
                lumping=fwd_lump,
                cache_path=str(getattr(args, "cache_path", "")),
                cached_node_weight=cached_node_weight,
                cache_meta=cache_meta,
                cache_mask=cache_mask,
                padding_enabled=padding_enabled,
                padding_value=padding_value,
                abaqus_root=str(getattr(args, "abaqus_root", "")),
                npz_dir=str(getattr(args, "npz_dir", "")),
                instance_name=str(getattr(args, "mesh_instance_name", "QJT-1")),
            )
        elif can_reuse:
            node_weight = cached_node_weight
            if is_main:
                print(
                    f"[*] use_node_weight=1 method={nw_method}: reuse cache "
                    f"shape={tuple(node_weight.shape)} meta={cache_nw_method or 'unspecified'}"
                )
        elif nw_method == "mesh":
            if is_main:
                if shape_ok and not method_ok:
                    print(
                        f"[!] method=mesh but cache meta={cache_nw_method!r}; "
                        "recomputing from Abaqus mesh"
                    )
                elif recompute_node_weight:
                    print("[*] recompute_node_weight=1: rebuilding mesh node_weight")
                else:
                    print("[*] method=mesh: computing from Abaqus CAX mesh")
            node_weight = compute_node_weight_from_abaqus_mesh(
                pos,
                cache_mask=cache_mask,
                padding_enabled=padding_enabled,
                padding_value=padding_value,
                abaqus_root=str(getattr(args, "abaqus_root", "")),
                npz_dir=str(getattr(args, "npz_dir", "")),
                instance_name=str(getattr(args, "mesh_instance_name", "QJT-1")),
                persist_path=(
                    str(getattr(args, "cache_path", ""))
                    if (persist and can_persist)
                    else ""
                ),
                persist_source_pos_shape=tuple(pos.shape),
            )
        else:
            if is_main:
                if shape_ok and not method_ok:
                    print(
                        f"[!] method=delaunay but cache meta={cache_nw_method!r}; "
                        "recomputing Delaunay on normalized pos"
                    )
                elif recompute_node_weight:
                    print("[*] recompute_node_weight=1: rebuilding Delaunay node_weight")
                else:
                    print("[*] method=delaunay: computing on normalized pos")
            node_weight = compute_node_weight_from_normalized_pos(
                pos_all_n,
                use_node_weight=True,
                cache_mask=cache_mask,
                padding_enabled=padding_enabled,
                padding_value=padding_value,
                max_edge=max_edge,
                persist_path=(
                    str(getattr(args, "cache_path", ""))
                    if (persist and can_persist)
                    else ""
                ),
                persist_source_pos_shape=tuple(pos.shape),
            )
        nw_train = node_weight[train_idx]
        nw_test = node_weight[loop_idx]
        if is_main:
            print(
                f"[*] use_node_weight=1 method={nw_method}: train q "
                f"min={nw_train[nw_train > 0].min().item():.3e} "
                f"max={nw_train.max().item():.3e}"
            )
            print(
                f"[*] node_weight pathways: attn={int(node_weight_in_attn)} "
                f"norm_fx={int(node_weight_in_norm_fx)} norm_y={int(node_weight_in_norm_y)} "
                f"loss={int(node_weight_in_loss)}"
            )

    # 计算fx/y归一化参数（如果padding_enabled，只使用有效点；q 可只作用于 y）
    t_normalizer = None
    t_train_n = t_test_n = None

    def _norm_node_weight(flag: bool, norm_type: str):
        if not (use_node_weight and flag and nw_train is not None):
            return None
        if norm_type == "min-max":
            return None
        return nw_train

    nw_fx = _norm_node_weight(node_weight_in_norm_fx, args.fx_norm_type)
    nw_y = _norm_node_weight(node_weight_in_norm_y, args.y_norm_type)
    weighted_fx_norm = nw_fx is not None
    weighted_y_norm = nw_y is not None

    def _build_field_normalizer(data, norm_type, node_weight=None):
        if node_weight is not None:
            return get_normalizer(
                data, norm_type, node_weight=node_weight, mask=mask_train
            )
        if padding_enabled and mask_train is not None:
            return get_normalizer(data[mask_train].unsqueeze(0), norm_type)
        return get_normalizer(data, norm_type)

    if input_dim > 0:
        fx_normalizer = _build_field_normalizer(fx_train, args.fx_norm_type, nw_fx)
        y_normalizer = _build_field_normalizer(y_train, args.y_norm_type, nw_y)
        if is_main:
            if weighted_fx_norm or weighted_y_norm:
                print(
                    "[*] Computing fx/y normalization with optional quadrature: "
                    f"fx_q={int(weighted_fx_norm)} y_q={int(weighted_y_norm)} "
                    "(per-sample unit mass when q is on; cases averaged equally)"
                )
            elif padding_enabled and mask_train is not None:
                total_points = fx_train.shape[0] * fx_train.shape[1]
                valid_points = int(mask_train.sum().item())
                padding_points = total_points - valid_points
                print(
                    f"[*] Computing fx/y normalization ({args.fx_norm_type}/{args.y_norm_type}) "
                    f"from valid points only: {valid_points} valid points, "
                    f"{padding_points} padding points excluded "
                    f"({100 * padding_points / total_points:.2f}%)"
                )
            if hasattr(fx_normalizer, "mean"):
                print(
                    f"    fx_normalizer (mean-std): mean range "
                    f"[{fx_normalizer.mean.min():.6f}, {fx_normalizer.mean.max():.6f}], "
                    f"std range [{fx_normalizer.std.min():.6f}, {fx_normalizer.std.max():.6f}]"
                )
            else:
                print(
                    f"    fx_normalizer (min-max): min range "
                    f"[{fx_normalizer.min.min():.6f}, {fx_normalizer.min.max():.6f}], "
                    f"max range [{fx_normalizer.max.min():.6f}, {fx_normalizer.max.max():.6f}]"
                )
            if hasattr(y_normalizer, "mean"):
                print(
                    f"    y_normalizer (mean-std): mean range "
                    f"[{y_normalizer.mean.min():.6f}, {y_normalizer.mean.max():.6f}], "
                    f"std range [{y_normalizer.std.min():.6f}, {y_normalizer.std.max():.6f}]"
                )
            else:
                print(
                    f"    y_normalizer (min-max): min range "
                    f"[{y_normalizer.min.min():.6f}, {y_normalizer.min.max():.6f}], "
                    f"max range [{y_normalizer.max.min():.6f}, {y_normalizer.max.max():.6f}]"
                )
    else:
        fx_normalizer = None
        y_normalizer = _build_field_normalizer(y_train, args.y_norm_type, nw_y)
        if is_main:
            print(f"[*] input_dim=0 (empty_fx): 跳过 fx 归一化")
            if weighted_y_norm:
                print("[*] Computing quadrature-aware y normalization only")

    if time_input and case_t is not None:
        t_train_3d = case_t[train_idx].view(-1, 1, 1)
        t_normalizer = get_normalizer(t_train_3d, args.fx_norm_type)
        t_train_n = t_normalizer.encode(case_t[train_idx].view(-1, 1, 1)).squeeze(-1).squeeze(-1)
        t_test_n = t_normalizer.encode(case_t[loop_idx].view(-1, 1, 1)).squeeze(-1).squeeze(-1)
        if is_main:
            print(f"[*] t_normalizer ({args.fx_norm_type}): train range [{t_train_n.min():.4f}, {t_train_n.max():.4f}]")

    # 保存fx/y/t归一化参数到coord_norm文件（如果coord_stats_path存在）
    if coord_stats_path and is_main:
        try:
            # 尝试加载现有文件（如果存在）
            saved = torch.load(coord_stats_path, map_location="cpu", weights_only=True)
        except (TypeError, FileNotFoundError):
            try:
                saved = torch.load(coord_stats_path, map_location="cpu")
            except FileNotFoundError:
                saved = {}
        
        # 更新或添加fx/y/t归一化参数
        if fx_normalizer is not None:
            if hasattr(fx_normalizer, "mean"):
                saved["fx_mean"] = fx_normalizer.mean.detach().cpu()
                saved["fx_std"] = fx_normalizer.std.detach().cpu()
            else:
                saved["fx_min"] = fx_normalizer.min.detach().cpu()
                saved["fx_max"] = fx_normalizer.max.detach().cpu()
        else:
            saved.pop("fx_mean", None)
            saved.pop("fx_std", None)
            saved.pop("fx_min", None)
            saved.pop("fx_max", None)

        if t_normalizer is not None:
            if hasattr(t_normalizer, "mean"):
                saved["t_mean"] = t_normalizer.mean.detach().cpu()
                saved["t_std"] = t_normalizer.std.detach().cpu()
            else:
                saved["t_min"] = t_normalizer.min.detach().cpu()
                saved["t_max"] = t_normalizer.max.detach().cpu()

        if hasattr(y_normalizer, "mean"):
            saved["y_mean"] = y_normalizer.mean.detach().cpu()
            saved["y_std"] = y_normalizer.std.detach().cpu()
        else:
            saved["y_min"] = y_normalizer.min.detach().cpu()
            saved["y_max"] = y_normalizer.max.detach().cpu()
        saved["normalization_node_weighted"] = {
            "fx": bool(weighted_fx_norm and fx_normalizer is not None),
            "y": bool(weighted_y_norm),
            "semantics": "node_weight pathways; per-sample unit mass when norm uses q",
            "in_attn": bool(node_weight_in_attn),
            "in_norm": bool(node_weight_in_norm),
            "in_norm_fx": bool(node_weight_in_norm_fx),
            "in_norm_y": bool(node_weight_in_norm_y),
            "in_loss": bool(node_weight_in_loss),
            "lumping": str(getattr(args, "node_weight_lumping", "") or "") or None,
            "q_kind": nw_method,
        }
        saved["split"] = {
            "mode": split_mode,
            "train_idx": train_idx.detach().cpu().long(),
            "val_idx": val_idx.detach().cpu().long() if val_idx is not None else None,
            "test_idx": heldout_test_idx.detach().cpu().long(),
            "eval_during_train": "val" if val_idx is not None else "test",
        }

        torch.save(saved, coord_stats_path)
        print(f"[*] Saved fx/y normalization stats to {coord_stats_path}")
        
        # Print fx normalization stats based on actual type
        if "fx_mean" in saved:
            print(f"    fx (mean-std): mean shape={saved['fx_mean'].shape}, std shape={saved['fx_std'].shape}")
            print(f"    fx_mean range: [{saved['fx_mean'].min():.6f}, {saved['fx_mean'].max():.6f}], fx_std range: [{saved['fx_std'].min():.6f}, {saved['fx_std'].max():.6f}]")
        elif "fx_min" in saved:
            print(f"    fx (min-max): min shape={saved['fx_min'].shape}, max shape={saved['fx_max'].shape}")
            print(f"    fx_min range: [{saved['fx_min'].min():.6f}, {saved['fx_min'].max():.6f}], fx_max range: [{saved['fx_max'].min():.6f}, {saved['fx_max'].max():.6f}]")
        
        # Print y normalization stats based on actual type
        if "y_mean" in saved:
            print(f"    y (mean-std): mean shape={saved['y_mean'].shape}, std shape={saved['y_std'].shape}")
            print(f"    y_mean range: [{saved['y_mean'].min():.6f}, {saved['y_mean'].max():.6f}], y_std range: [{saved['y_std'].min():.6f}, {saved['y_std'].max():.6f}]")
        elif "y_min" in saved:
            print(f"    y (min-max): min shape={saved['y_min'].shape}, max shape={saved['y_max'].shape}")
            print(f"    y_min range: [{saved['y_min'].min():.6f}, {saved['y_min'].max():.6f}], y_max range: [{saved['y_max'].min():.6f}, {saved['y_max'].max():.6f}]")

    pos_train_n = pos_all_n[train_idx]
    fx_train_n = fx_normalizer.encode(fx_train) if fx_normalizer is not None else fx_train
    y_train_n = y_normalizer.encode(y_train)
    pos_test_n = pos_all_n[loop_idx]
    fx_test_n = fx_normalizer.encode(fx_test) if fx_normalizer is not None else fx_test
    y_test_n = y_normalizer.encode(y_test)

    if device.type == "cuda":
        pos_normalizer.cuda()
        if fx_normalizer is not None:
            fx_normalizer.cuda()
        if t_normalizer is not None:
            t_normalizer.cuda()
        y_normalizer.cuda()

    # Optional preload to GPU (keeps existing semantics)
    preload = bool(getattr(args, "preload_data_to_gpu", False))
    if preload and device.type == "cuda":
        pos_train_n = pos_train_n.to(device)
        fx_train_n = fx_train_n.to(device)
        y_train_n = y_train_n.to(device)
        pos_test_n = pos_test_n.to(device)
        fx_test_n = fx_test_n.to(device)
        y_test_n = y_test_n.to(device)
        if padding_enabled:
            mask_train = mask_train.to(device)
            mask_test = mask_test.to(device)
        if use_node_weight and nw_train is not None:
            nw_train = nw_train.to(device)
            nw_test = nw_test.to(device)
        dl_num_workers = 0
        dl_pin_memory = False
        if is_main:
            print(f"Preloaded all data tensors to GPU: {device}")
    else:
        dl_num_workers = 0
        dl_pin_memory = False

    # --- kNN cache for EdgeConv and gradient loss (separate conditions) ---
    knn_idx = None
    knn_valid = None
    knn_idx_train = None
    knn_valid_train = None
    knn_idx_test = None
    knn_valid_test = None
    
    grad_w = float(getattr(args, "grad_loss_weight", 0.0))
    use_edge_conv = bool(int(getattr(args, "use_edge_conv", 0)))
    knn_enable = bool(int(getattr(args, "knn_enable", 1)))
    
    # Check if kNN is needed for EdgeConv
    need_knn_for_edgeconv = use_edge_conv and knn_enable
    # Check if kNN is needed for gradient loss
    need_knn_for_grad = (grad_w > 0.0) and knn_enable
    # Build kNN cache if needed for either purpose
    need_knn = need_knn_for_edgeconv or need_knn_for_grad
    
    if need_knn:
        from scripts.transolver.train.metrics import build_knn_cache

        cache_path = getattr(args, "cache_path", "")
        k = int(getattr(args, "grad_k", 8))
        knn_cache_path = getattr(args, "knn_cache_path", "")
        if not knn_cache_path:
            base = cache_path.replace(".pt", "") if cache_path else "knn_cache"
            knn_cache_path = f"{base}_knn_k{k}.pt"
            args.knn_cache_path = knn_cache_path

        rebuild = bool(int(getattr(args, "knn_cache_rebuild", 0)))
        if (not rebuild) and knn_cache_path and os.path.exists(knn_cache_path):
            try:
                saved = torch.load(knn_cache_path, weights_only=True)
            except TypeError:
                saved = torch.load(knn_cache_path)
            if isinstance(saved, dict) and saved.get("k") == k:
                knn_idx = saved.get("knn_idx")
                knn_valid = saved.get("knn_valid")
        
        if (knn_idx is None) or (knn_valid is None):
            block_size = int(getattr(args, "knn_block_size", 512))
            pos_norm_cpu = IdentityTransformer(
                min_val=pos_normalizer.min.detach().cpu(),
                max_val=pos_normalizer.max.detach().cpu(),
            )
            pos_all_n = pos_norm_cpu.encode(pos.cpu())
            if padding_enabled:
                mask_all = torch.zeros(pos.shape[0], pos.shape[1], dtype=mask_train.dtype, device="cpu")
                mask_all[train_idx] = mask_train.cpu()
                mask_all[loop_idx] = mask_test.cpu()
            else:
                mask_all = None
            with torch.no_grad():
                knn_idx, knn_valid = build_knn_cache(pos_all_n, k=k, mask=mask_all, block_size=block_size)
            if knn_cache_path:
                os.makedirs(os.path.dirname(knn_cache_path) or ".", exist_ok=True)
                torch.save({"k": k, "knn_idx": knn_idx, "knn_valid": knn_valid}, knn_cache_path)

        if preload and device.type == "cuda":
            if knn_idx is not None:
                knn_idx = knn_idx.to(device)
            if knn_valid is not None:
                knn_valid = knn_valid.to(device)

        if knn_idx is not None and knn_valid is not None:
            knn_idx_train = knn_idx[train_idx]
            knn_valid_train = knn_valid[train_idx]
            knn_idx_test = knn_idx[loop_idx]
            knn_valid_test = knn_valid[loop_idx]

    # --- Build datasets with or without kNN data ---
    def _build_ds(p, f, y, m, nw, k_idx, k_val, ord, inv, t_n=None):
        tensors = [p, f, y]
        if m is not None:
            tensors.append(m)
        if nw is not None:
            tensors.append(nw)
        if k_idx is not None:
            tensors.append(k_idx)
        if k_val is not None:
            tensors.append(k_val)
        if ord is not None:
            tensors.append(ord)
        if inv is not None:
            tensors.append(inv)
        if t_n is not None:
            tensors.append(t_n)
        return TensorDataset(*tensors)

    if sato_indices is not None:
        order_train, inverse_train = sato_indices["order"][train_idx], sato_indices["inverse"][train_idx]
        order_test, inverse_test = sato_indices["order"][loop_idx], sato_indices["inverse"][loop_idx]
    else:
        order_train = inverse_train = order_test = inverse_test = None

    train_ds = _build_ds(
        pos_train_n, fx_train_n, y_train_n, mask_train, nw_train,
        knn_idx_train, knn_valid_train, order_train, inverse_train,
        t_train_n if time_input else None,
    )
    test_ds = _build_ds(
        pos_test_n, fx_test_n, y_test_n, mask_test, nw_test,
        knn_idx_test, knn_valid_test, order_test, inverse_test,
        t_test_n if time_input else None,
    )

    if ctx.enabled:
        if DistributedSampler is None:
            raise RuntimeError("DistributedSampler unavailable but args.ddp is True")
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=True,
            seed=effective_train_seed,
        )
        test_sampler = DistributedSampler(
            test_ds,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=False,
            seed=effective_train_seed,
        )
        dl_common = dict(
            num_workers=dl_num_workers,
            pin_memory=dl_pin_memory,
            worker_init_fn=make_worker_init_fn(effective_train_seed) if dl_num_workers > 0 else None,
            generator=make_dataloader_generator(effective_train_seed),
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, **dl_common)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, sampler=test_sampler, **dl_common)
    else:
        dl_common = dict(
            num_workers=dl_num_workers,
            pin_memory=dl_pin_memory,
            worker_init_fn=make_worker_init_fn(effective_train_seed) if dl_num_workers > 0 else None,
            generator=make_dataloader_generator(effective_train_seed),
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, **dl_common
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, **dl_common
        )

    # Re-lock after data prep so Linear/Conv init is not affected by earlier RNG draws.
    random.seed(effective_train_seed)
    np.random.seed(effective_train_seed)
    torch.manual_seed(effective_train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_train_seed)
    if is_main:
        print(f"[*] Re-locked train_seed={effective_train_seed} immediately before weight init")

    # --- Model ---
    model = MultiNetWrapper(args, input_dim=input_dim, output_dim=output_dim).to(device)
    if is_main:
        n_params_total = sum(p.numel() for p in model.parameters())
        n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[*] Model parameters: total={n_params_total:,}, "
            f"trainable={n_params_trainable:,} ({n_params_total / 1e6:.3f}M)"
        )

    if ctx.enabled:
        # MultiNet / SATO 等路径下，单次 forward 可能未用到全部子网参数；DDP 默认会报错，需开启 unused 检测
        model = DDP(
            model,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
            find_unused_parameters=True,
        )
        if is_main:
            print(f"Wrapped model with DDP for {ctx.world_size} GPUs (find_unused_parameters=True)")
    else:
        if device.type == "cuda" and num_gpus > 1:
            if is_main:
                print(f"Wrapping model with DataParallel for {num_gpus} GPUs")
            model = nn.DataParallel(model, device_ids=gpu_ids)
            device = torch.device(f"cuda:{gpu_ids[0]}")

    if getattr(args, "load_ckpt", ""):
        transfer = getattr(args, "transfer", None)
        transfer_json = str(getattr(args, "transfer_json", "") or "").strip()
        if transfer_json:
            if os.path.isfile(transfer_json):
                with open(transfer_json, "r", encoding="utf-8") as f:
                    transfer = json.load(f)
            else:
                transfer = json.loads(transfer_json)
            args.transfer = transfer
        maybe_load_checkpoint(
            model,
            args.load_ckpt,
            device=device,
            is_main_process=is_main,
            transfer=transfer,
            args=args,
        )

    # --- Eval-only ---
    if getattr(args, "eval_only", False):
        if is_main:
            eval_dir = os.path.join(plot_dir, args.save_name, "eval_only")
            plot_samples(model, test_loader, y_normalizer, device, eval_dir, prefix="eval")
        cleanup_ddp(ctx)
        _teardown_viz_cuda(device)
        return

    # --- Smoke test ---
    if getattr(args, "smoke_test", False):
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                idx = 0
                pos_b = batch[idx].to(device, non_blocking=True); idx += 1
                fx_b = batch[idx].to(device, non_blocking=True); idx += 1
                y_b = batch[idx].to(device, non_blocking=True); idx += 1
                if padding_enabled and mask_train is not None:
                    idx += 1
                if use_node_weight:
                    idx += 1
                if need_knn:
                    idx += 2
                if sato_indices is not None:
                    idx += 2
                t_b = batch[idx].to(device, non_blocking=True) if time_input else None
                model_fx = fx_b if fx_b.shape[-1] > 0 else None
                nw_b = None
                if use_node_weight and node_weight_in_attn:
                    # batch layout: pos,fx,y,[mask],[nw],...
                    nw_i = 3 + (1 if padding_enabled else 0)
                    nw_b = batch[nw_i].to(device, non_blocking=True)
                out = y_normalizer.decode(
                    model(pos_b, fx=model_fx, T=t_b, quadrature_weights=nw_b)
                )
                y_orig = y_normalizer.decode(y_b)
                if is_main:
                    from scripts.transolver.train.metrics import relative_l2
                    n_ch = out.shape[-1]
                    ch_rels = [relative_l2(out[..., i:i+1], y_orig[..., i:i+1]) for i in range(n_ch)]
                    avg_rel = torch.stack(ch_rels).mean()
                    print(f"Smoke Test RelL2(avg channels): {avg_rel.item():.6f}")
                break
        cleanup_ddp(ctx)
        _teardown_viz_cuda(device)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    max_train_batches = int(getattr(args, "max_train_batches", 0) or 0)
    if max_train_batches > 0:
        steps_per_epoch = min(steps_per_epoch, max_train_batches)
    lr_scheduler, lr_scheduler_mode = build_lr_scheduler(
        optimizer, args, steps_per_epoch=steps_per_epoch
    )
    early_stop_enabled = int(getattr(args, "early_stop", 1))
    early_stop_patience = int(getattr(args, "early_stop_patience", 100))
    early_stop_min_delta_rel = float(getattr(args, "early_stop_min_delta_rel", 0.001))
    epochs_without_improvement = 0
    stopped_early = False

    train_losses, eval_losses, epochs_list = [], [], []
    train_losses_ch, eval_losses_ch = [], []
    best_eval_loss = float("inf")
    best_epoch = 0

    def loss_fn(
        out_dec,
        y_dec,
        fx_b_enc,
        fx_norm,
        mask_b=None,
        pos_b=None,
        knn_idx=None,
        knn_valid=None,
        node_weight_b=None,
    ):
        """
        Relative L2 training loss, optionally quadrature-weighted.

        node_weight_b is passed only when node_weight_in_loss=1 (see run_epoch).
        """
        from scripts.transolver.train.metrics import weighted_mse_mae, weighted_relative_l2

        n_ch = out_dec.shape[-1]
        per_ch_losses = []
        mse_vals = []
        mae_vals = []
        for i in range(n_ch):
            l = weighted_relative_l2(
                out_dec[..., i : i + 1],
                y_dec[..., i : i + 1],
                node_weight=node_weight_b,
                mask=mask_b,
            )
            per_ch_losses.append(l)
            mse_i, mae_i = weighted_mse_mae(
                out_dec[..., i : i + 1],
                y_dec[..., i : i + 1],
                node_weight=node_weight_b,
                mask=mask_b,
            )
            mse_vals.append(mse_i)
            mae_vals.append(mae_i)
        
        rel_ch = torch.stack(per_ch_losses)
        rel_mean = rel_ch.mean()
        per_ch_t = rel_ch
        mean_loss = rel_mean
        grad_mean = None
        grad_ch = None

        grad_w = float(getattr(args, "grad_loss_weight", 0.0))
        if grad_w > 0.0 and pos_b is not None:
            from scripts.transolver.train.metrics import gradient_loss

            grad_k = int(getattr(args, "grad_k", 8))
            grad_eps = float(getattr(args, "grad_eps", 1e-6))
            grad_mean, grad_ch = gradient_loss(
                pos_b,
                out_dec,
                y_dec,
                k=grad_k,
                mask=mask_b,
                eps=grad_eps,
                knn_idx=knn_idx,
                knn_valid=knn_valid,
            )
            per_ch_t = rel_ch + grad_w * grad_ch
            mean_loss = per_ch_t.mean()
        
        metrics = {
            "rel": rel_mean,
            "grad": grad_mean,
            "rel_ch": rel_ch,
            "grad_ch": grad_ch,
            "mse": torch.stack(mse_vals).mean(),
            "mae": torch.stack(mae_vals).mean(),
        }
        return mean_loss, per_ch_t, metrics

    if is_main:
        print(f"Starting KETI training for {args.epochs} epochs... (ddp={ctx.enabled})")
        if lr_scheduler_mode == "batch":
            print(
                f"[*] LR scheduler: onecycle per-batch, max_lr={args.lr:.6g}, "
                f"steps_per_epoch={steps_per_epoch}, total_steps={steps_per_epoch * int(args.epochs)}, "
                f"pct_start=0.3, div_factor=25, final_div_factor=1e4, "
                f"lr_now={current_lr(optimizer):.6g}"
            )
        else:
            print(
                f"[*] LR scheduler: {lr_scheduler_mode}, init_lr={args.lr:.6g}, lr_min={getattr(args, 'lr_min', 1e-6):.6g}"
            )
        if early_stop_enabled:
            print(
                f"[*] Early stop: patience={early_stop_patience} epochs, "
                f"min_delta_rel={early_stop_min_delta_rel:.4f} ({early_stop_min_delta_rel * 100:.2f}%)"
            )
        else:
            print("[*] Early stop: disabled")

    def _save_plot_loss_curves():
        save_dir = os.path.join(plot_dir, args.save_name)
        plot_loss_curves(
            train_losses,
            eval_losses,
            epochs_list,
            save_dir,
            best_eval_info=(best_epoch, best_eval_loss),
        )
        if train_losses_ch and eval_losses_ch:
            plot_loss_curves_channels(
                train_losses_ch,
                eval_losses_ch,
                epochs_list,
                save_dir,
                channel_names=args.output_cols,
            )
        save_loss_curves_npy(
            train_losses,
            eval_losses,
            epochs_list,
            save_dir,
            train_losses_ch=train_losses_ch if train_losses_ch else None,
            eval_losses_ch=eval_losses_ch if eval_losses_ch else None,
            channel_names=getattr(args, "output_cols", None),
        )

    for epoch in range(args.epochs):
        t0 = time.time()
        if ctx.enabled:
            train_loader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]

        train_loss, train_loss_ch, train_rel, train_grad, train_rel_ch, train_grad_ch, train_mse, train_mae = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            y_normalizer=y_normalizer,
            fx_normalizer=fx_normalizer,
            loss_fn=loss_fn,
            optimizer=optimizer,
            max_grad_norm=getattr(args, "max_grad_norm", None),
            max_batches=getattr(args, "max_train_batches", 0),
            padding_enabled=padding_enabled,
            padding_value=padding_value,
            use_knn=need_knn,
            use_sato=(sato_indices is not None),
            use_time_input=time_input,
            use_node_weight=use_node_weight,
            node_weight_in_attn=node_weight_in_attn,
            node_weight_in_loss=node_weight_in_loss,
            lr_scheduler=lr_scheduler if lr_scheduler_mode == "batch" else None,
        )

        eval_interval = int(getattr(args, "eval_interval", 100))
        eval_rel = float("nan")
        eval_rel_ch = None
        eval_grad = float("nan")
        eval_grad_ch = None
        eval_mse = float("nan")
        eval_mae = float("nan")
        did_eval = (epoch + 1) % eval_interval == 0
        if did_eval:
            _, _, eval_rel, eval_grad, eval_rel_ch, eval_grad_ch, eval_mse, eval_mae = run_epoch(
                model=model,
                loader=test_loader,
                device=device,
                y_normalizer=y_normalizer,
                fx_normalizer=fx_normalizer,
                loss_fn=loss_fn,
                optimizer=None,
                max_grad_norm=None,
                max_batches=getattr(args, "max_eval_batches", 0),
                padding_enabled=padding_enabled,
                padding_value=padding_value,
                use_knn=need_knn,
                use_sato=(sato_indices is not None),
                use_time_input=time_input,
                use_node_weight=use_node_weight,
                node_weight_in_attn=node_weight_in_attn,
                node_weight_in_loss=node_weight_in_loss,
            )

            if is_meaningful_improvement(best_eval_loss, eval_rel, early_stop_min_delta_rel):
                best_eval_loss = eval_rel
                best_epoch = epoch
                epochs_without_improvement = 0
                save_state_dict(model, os.path.join(ckpt_dir, f"{args.save_name}_best_eval.pt"), is_main_process=is_main)
                if is_main:
                    print(f"  [SAVE] Best Eval: {best_eval_loss:.6f} @ epoch {epoch + 1}")
            else:
                epochs_without_improvement += eval_interval
                if is_main and best_eval_loss < float("inf"):
                    if (
                        epochs_without_improvement >= early_stop_patience
                        or epochs_without_improvement % max(1, min(early_stop_patience // 4, 50)) == 0
                    ):
                        print(
                            f"  [EARLY-STOP] No >{early_stop_min_delta_rel * 100:.2f}% eval improvement "
                            f"({epochs_without_improvement}/{early_stop_patience} epochs)"
                        )

            if lr_scheduler is not None and lr_scheduler_mode == "plateau":
                lr_scheduler.step(eval_rel)

        if lr_scheduler is not None and lr_scheduler_mode == "epoch":
            lr_scheduler.step()

        eval_tag = "ValRel" if val_idx is not None else "EvalRel"
        dt = time.time() - t0
        lr_now = current_lr(optimizer)
        if is_main:
            if did_eval:
                print(
                    f"Epoch {epoch}: "
                    f"TrainRel={train_rel:.6f}, TrainMSE={train_mse:.6e}, TrainMAE={train_mae:.6e}, "
                    f"TrainGrad={train_grad:.6f}, "
                    f"{eval_tag}={eval_rel:.6f}, EvalMSE={eval_mse:.6e}, EvalMAE={eval_mae:.6e}, "
                    f"EvalGrad={eval_grad:.6f}, "
                    f"LR={lr_now:.6g}, Time={dt:.2f}s"
                )
            else:
                print(
                    f"Epoch {epoch}: "
                    f"TrainRel={train_rel:.6f}, TrainMSE={train_mse:.6e}, TrainMAE={train_mae:.6e}, "
                    f"TrainGrad={train_grad:.6f}, "
                    f"LR={lr_now:.6g}, Time={dt:.2f}s"
                )

            train_losses.append(train_rel)
            eval_losses.append(eval_rel)
            epochs_list.append(epoch)
            if train_rel_ch is not None:
                train_losses_ch.append(train_rel_ch)
                if eval_rel_ch is None:
                    eval_losses_ch.append([float("nan")] * len(train_rel_ch))
                else:
                    eval_losses_ch.append(eval_rel_ch)

        if (epoch + 1) % int(getattr(args, "viz_interval", 100)) == 0:
            if is_main:
                _save_plot_loss_curves()
                # Note: plot_samples remains disabled to maintain training speed and avoid device issues.
                # Use eval_keti.py for full sample evaluation.
                pass

        if (
            early_stop_enabled
            and did_eval
            and epochs_without_improvement >= early_stop_patience
        ):
            stopped_early = True
            if is_main:
                print(
                    f"\n[EARLY-STOP] Triggered at epoch {epoch + 1}: "
                    f"no >{early_stop_min_delta_rel * 100:.2f}% eval improvement for "
                    f"{epochs_without_improvement} epochs. "
                    f"Best eval={best_eval_loss:.6f} @ epoch {best_epoch + 1}"
                )
            break

    save_state_dict(model, os.path.join(ckpt_dir, f"{args.save_name}_latest.pt"), is_main_process=is_main)
    if is_main:
        if stopped_early:
            print(f"Training stopped early at epoch {epoch + 1}/{args.epochs}.")
        else:
            print("Training finished.")
        print(f"Best eval checkpoint: epoch {best_epoch + 1}, EvalRel={best_eval_loss:.6f}")
        _save_plot_loss_curves()

    cleanup_ddp(ctx)
    _teardown_viz_cuda(device)

    if is_main:
        _run_auto_eval(args, run_root, ckpt_dir)

