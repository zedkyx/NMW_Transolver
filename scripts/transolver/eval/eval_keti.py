import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import time
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path before importing scripts.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# Plotting settings for professional look
from scripts.transolver.viz.style import apply_paper_rcparams

apply_paper_rcparams()

from scripts.transolver.keti_argparser import build_argparser, load_yaml_config
from scripts.transolver.keti_data_loader import (
    load_keti_data,
    resolve_train_val_test_indices,
    resolve_coord_norm_path,
    default_coord_norm_path,
    resolve_output_cols,
    _load_cache,
    cache_method_matches,
    compute_node_weight_from_abaqus_mesh,
    compute_node_weight_from_normalized_pos,
    load_or_compute_mesh_node_weight,
    normalize_node_weight_max_edge,
    resolve_node_weight_method,
    resolve_node_weight_pathways,
)
from scripts.transolver.train.loops import build_padding_mask
from scripts.transolver.viz.samples import (
    plot_combined_channels_atmospheric, 
    plot_violin_distribution, 
    plot_case_grid, 
    plot_peak_scatter,
    analyze_high_stress_region_error,
    plot_high_stress_error_analysis,
    compute_sample_rel_l2,
    plot_rel_vs_time_scatter,
    resolve_rel_err_denoms,
)
from scripts.transolver.train.modeling import MultiNetWrapper
from scripts.transolver.train.checkpointing import maybe_load_checkpoint
from utils import UnitTransformer, IdentityTransformer
from utils.attention_viz import (
    finalize_rank_series,
    save_attn_rank_curves,
    load_attn_rank_json,
    save_attn_rank_json,
    save_attention_heatmaps,
)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _resolve_eval_dir(args, timestamp: str) -> str:
    """评估结果目录：优先 {run_dir}/eval/paper_eval_{ts}，与 9096 run 结构一致。"""
    run_dir = (getattr(args, "run_dir", "") or "").strip()
    if not run_dir and getattr(args, "load_ckpt", ""):
        run_dir = os.path.dirname(os.path.dirname(args.load_ckpt))
    if run_dir:
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(_repo_root(), run_dir)
        return os.path.join(run_dir, "eval", f"paper_eval_{timestamp}")
    return os.path.join(_repo_root(), "runs", "keti", args.save_name, f"paper_eval_{timestamp}")


def _encode_tensor_batched(normalizer, x, device, batch_size: int):
    """Encode [B,...] on GPU in chunks. Full fx.to(cuda) OOMs on large-N parts (e.g. Part-1-1)."""
    if normalizer is None or x is None:
        return x
    bs = max(1, int(batch_size) or 1)
    chunks = []
    for i in range(0, int(x.shape[0]), bs):
        sl = x[i : i + bs].to(device)
        chunks.append(normalizer.encode(sl).cpu())
        del sl
    return torch.cat(chunks, dim=0) if chunks else x


def create_unit_transformer_from_stats(mean, std):
    """从保存的mean和std创建UnitTransformer"""
    normalizer = UnitTransformer.__new__(UnitTransformer)
    # 确保shape正确：[1, 1, F] 或 [1, 1, C]（与UnitTransformer的keepdim=True一致）
    # UnitTransformer的mean/std shape应该是[1, 1, F]（从[B, N, F]计算，keepdim=True）
    
    # 处理不同的shape情况
    original_mean_shape = mean.shape
    original_std_shape = std.shape
    
    if mean.ndim == 0:  # scalar -> [1, 1, 1]
        mean = mean.view(1, 1, 1)
        std = std.view(1, 1, 1)
    elif mean.ndim == 1:  # [F] -> [1, 1, F]
        mean = mean.view(1, 1, -1)
        std = std.view(1, 1, -1)
    elif mean.ndim == 2:
        if mean.shape == (1, 1):  # [1, 1] 单通道情况 -> [1, 1, 1]
            mean = mean.unsqueeze(-1)
            std = std.unsqueeze(-1)
        elif mean.shape[0] == 1:  # [1, F] -> [1, 1, F]
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        else:  # [B, F] -> [1, 1, F] (取平均)
            mean = mean.mean(dim=0, keepdim=True).unsqueeze(0)
            std = std.mean(dim=0, keepdim=True).unsqueeze(0)
    # 如果已经是[1, 1, F]或[1, 1, C]，保持不变
    
    normalizer.mean = mean
    normalizer.std = std
    return normalizer


def _fmt_tensor_range(t, empty_label="empty"):
    """Print min/max; empty tensors (empty_fx) skip min/max."""
    if t is None or (isinstance(t, torch.Tensor) and t.numel() == 0):
        return empty_label
    if isinstance(t, torch.Tensor):
        return f"[{t.min():.4f}, {t.max():.4f}]"
    t = np.asarray(t)
    if t.size == 0:
        return empty_label
    return f"[{t.min():.4f}, {t.max():.4f}]"


def _get_time_for_indices(cache, fx, indices):
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()
    else:
        indices = np.asarray(indices)
    t_src = cache.get("t")
    if t_src is not None:
        if isinstance(t_src, torch.Tensor):
            return t_src[indices].numpy().astype(np.float64)
        return np.asarray(t_src, dtype=np.float64)[indices]
    if isinstance(fx, torch.Tensor):
        return fx[indices, 0, 0].cpu().numpy().astype(np.float64)
    return fx[indices, 0, 0].astype(np.float64)


def _build_keti_tensor_dataset(
    pos_n,
    fx_n,
    y_n,
    indices=None,
    padding_enabled=False,
    mask_all=None,
    node_weight_all=None,
    knn_idx_all=None,
    knn_valid_all=None,
    sato_indices=None,
    t_n=None,
):
    if indices is not None:
        pos_n = pos_n[indices]
        fx_n = fx_n[indices]
        y_n = y_n[indices]
    tensors = [pos_n, fx_n, y_n]
    if padding_enabled and mask_all is not None:
        mask_s = mask_all if indices is None else mask_all[indices]
        tensors.append(mask_s)
    if node_weight_all is not None:
        nw_s = node_weight_all if indices is None else node_weight_all[indices]
        tensors.append(nw_s)
    if knn_idx_all is not None and knn_valid_all is not None:
        knn_idx_s = knn_idx_all if indices is None else knn_idx_all[indices]
        knn_valid_s = knn_valid_all if indices is None else knn_valid_all[indices]
        tensors.extend([knn_idx_s, knn_valid_s])
    if sato_indices is not None:
        order_s = sato_indices["order"] if indices is None else sato_indices["order"][indices]
        inverse_s = sato_indices["inverse"] if indices is None else sato_indices["inverse"][indices]
        tensors.extend([order_s, inverse_s])
    if t_n is not None:
        t_s = t_n if indices is None else t_n[indices]
        tensors.append(t_s)
    return TensorDataset(*tensors)


def _run_keti_inference(
    model,
    loader,
    y_normalizer,
    device,
    padding_enabled,
    need_knn,
    sato_in_loader,
    num_samples,
    use_time_input=False,
    use_node_weight=False,
    node_weight_in_attn=True,
    log_prefix="",
):
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            idx = 0
            b_pos = batch[idx].to(device)
            idx += 1
            b_fx = batch[idx].to(device)
            idx += 1
            b_y = batch[idx].to(device)
            idx += 1

            b_mask = None
            if padding_enabled:
                b_mask = batch[idx].to(device)
                idx += 1

            b_nw = None
            if use_node_weight:
                b_nw = batch[idx].to(device)
                idx += 1
            qw = b_nw if (b_nw is not None and node_weight_in_attn) else None

            b_knn_idx = b_knn_valid = None
            if need_knn:
                b_knn_idx = batch[idx].to(device)
                idx += 1
                b_knn_valid = batch[idx].to(device)
                idx += 1

            b_sato = None
            if sato_in_loader:
                b_order = batch[idx].to(device)
                idx += 1
                b_inverse = batch[idx].to(device)
                idx += 1
                b_sato = {"order": b_order, "inverse": b_inverse}

            b_t = None
            if use_time_input:
                b_t = batch[idx].to(device)
                idx += 1

            b_fx_model = b_fx if b_fx.shape[-1] > 0 else None
            b_out_n = model(
                b_pos,
                fx=b_fx_model,
                mask=b_mask,
                knn_idx=b_knn_idx,
                knn_valid=b_knn_valid,
                sato_indices=b_sato,
                T=b_t,
                quadrature_weights=qw,
            )
            all_preds.append(y_normalizer.decode(b_out_n).cpu())
            all_trues.append(y_normalizer.decode(b_y).cpu())

            if (batch_idx + 1) % 10 == 0:
                tag = f"[{log_prefix}] " if log_prefix else ""
                print(f"    {tag}处理了 {batch_idx + 1}/{len(loader)} batches")

    preds = torch.cat(all_preds, dim=0).numpy()
    trues = torch.cat(all_trues, dim=0).numpy()
    if len(preds) != num_samples:
        raise RuntimeError(f"推理样本数不一致: got {len(preds)}, expected {num_samples}")
    return preds, trues


def _collect_attn_rank_over_loader(
    model,
    loader,
    device,
    *,
    padding_enabled,
    need_knn,
    sato_in_loader,
    use_time_input=False,
    use_node_weight=False,
    node_weight_in_attn=True,
    rel_tol: float = 1e-3,
    label: str = "Transolver",
):
    """Full-test average rank of A=φψ^T (or WW^T) per layer.

    Forward still runs by DataLoader batches (memory). Factors are gathered on CPU,
    then each layer does **one** batched SVD over the entire test set.
    """
    from utils.attention_viz import (
        _factors_from_attn_item_full_batch,
        batched_numerical_ranks,
        finalize_rank_series,
    )

    if hasattr(model, "enable_attention_recording"):
        model.enable_attention_recording(True)
    else:
        raise RuntimeError("model does not support enable_attention_recording")

    # buffers[layer_idx][channel_idx] = {"phi": [...], "psi": [...], "shared": bool, "layer": int}
    buffers = {}
    mask_chunks = []
    n_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            idx = 0
            b_pos = batch[idx].to(device); idx += 1
            b_fx = batch[idx].to(device); idx += 1
            _b_y = batch[idx].to(device); idx += 1

            b_mask = None
            if padding_enabled:
                b_mask = batch[idx].to(device); idx += 1
            b_nw = None
            if use_node_weight:
                b_nw = batch[idx].to(device); idx += 1
            qw = b_nw if (b_nw is not None and node_weight_in_attn) else None
            b_knn_idx = b_knn_valid = None
            if need_knn:
                b_knn_idx = batch[idx].to(device); idx += 1
                b_knn_valid = batch[idx].to(device); idx += 1
            b_sato = None
            if sato_in_loader:
                b_order = batch[idx].to(device); idx += 1
                b_inverse = batch[idx].to(device); idx += 1
                b_sato = {"order": b_order, "inverse": b_inverse}
            b_t = None
            if use_time_input:
                b_t = batch[idx].to(device); idx += 1

            b_fx_model = b_fx if b_fx.shape[-1] > 0 else None
            _ = model(
                b_pos,
                fx=b_fx_model,
                mask=b_mask,
                knn_idx=b_knn_idx,
                knn_valid=b_knn_valid,
                sato_indices=b_sato,
                T=b_t,
                quadrature_weights=qw,
            )
            attn_raw = model.get_last_attention()
            if attn_raw and isinstance(attn_raw[0], list):
                channel_lists = attn_raw
            else:
                channel_lists = [attn_raw]

            B = b_pos.shape[0]
            if b_mask is None:
                mask_chunks.append(torch.ones(B, b_pos.shape[1], dtype=torch.bool))
            else:
                m = b_mask.detach().cpu()
                if m.dim() == 3:
                    m = m.squeeze(-1)
                mask_chunks.append(m.bool())

            for ch_i, ch_list in enumerate(channel_lists):
                if not ch_list:
                    continue
                for item in ch_list:
                    layer = int(item.get("layer", -1))
                    phi, psi, shared = _factors_from_attn_item_full_batch(item)
                    if phi is None:
                        continue
                    key = (layer, ch_i)
                    if key not in buffers:
                        buffers[key] = {
                            "phi": [],
                            "psi": [],
                            "shared": shared,
                            "layer": layer,
                            "slice_num": int(phi.shape[-1]),
                        }
                    # keep on CPU to free GPU during long forward
                    buffers[key]["phi"].append(phi.detach().cpu())
                    if shared:
                        buffers[key]["psi"].append(None)
                    else:
                        buffers[key]["psi"].append(psi.detach().cpu())
                    buffers[key]["shared"] = bool(buffers[key]["shared"] and shared)

            n_samples += B
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"    [attn_rank] gather {batch_idx + 1}/{len(loader)} batches, "
                    f"samples={n_samples}",
                    flush=True,
                )

    if hasattr(model, "enable_attention_recording"):
        model.enable_attention_recording(False)

    if n_samples == 0 or not buffers:
        return finalize_rank_series({}, label=label, rel_tol=rel_tol, n_samples=0)

    mask_all = torch.cat(mask_chunks, dim=0)  # [N_test, N]
    layers = sorted({k[0] for k in buffers.keys()})
    n_channels = 1 + max(k[1] for k in buffers.keys())

    print(
        f"    [attn_rank] batched SVD over full test: "
        f"N_test={n_samples}, layers={layers}, channels={n_channels}, rel_tol={rel_tol}",
        flush=True,
    )

    running = {}
    # geo24: ~1284×1×14k×32 ≈ 2GB/layer；整批 SVD，OOM 时按 256 样本分块
    svd_chunk = None
    for layer in layers:
        ch_ranks = []
        slice_num = 0
        for ch_i in range(n_channels):
            key = (layer, ch_i)
            if key not in buffers:
                continue
            buf = buffers[key]
            slice_num = int(buf["slice_num"])
            phi = torch.cat(buf["phi"], dim=0)
            shared = bool(buf["shared"])
            psi = phi if shared else torch.cat(buf["psi"], dim=0)
            print(
                f"    [attn_rank] SVD layer={layer} ch={ch_i} "
                f"shape={tuple(phi.shape)} shared={shared}",
                flush=True,
            )
            try:
                phi_g = phi.to(device=device, non_blocking=True)
                psi_g = phi_g if shared else psi.to(device=device, non_blocking=True)
                mask_g = mask_all.to(device=device, non_blocking=True)
                r = batched_numerical_ranks(
                    phi_g, psi_g, mask=mask_g, rel_tol=rel_tol, shared=shared, chunk_size=svd_chunk
                )
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                svd_chunk = 256 if svd_chunk is None else max(32, svd_chunk // 2)
                print(f"    [attn_rank] OOM → retry chunk_size={svd_chunk}", flush=True)
                phi_g = phi.to(device=device, non_blocking=True)
                psi_g = phi_g if shared else psi.to(device=device, non_blocking=True)
                mask_g = mask_all.to(device=device, non_blocking=True)
                r = batched_numerical_ranks(
                    phi_g, psi_g, mask=mask_g, rel_tol=rel_tol, shared=shared, chunk_size=svd_chunk
                )
            ch_ranks.append(r.detach().float().cpu())
            del phi_g, psi_g, mask_g, phi, psi, r
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not ch_ranks:
            continue
        mean_per_sample = torch.stack(ch_ranks, dim=0).mean(dim=0)  # [N_test]
        running[layer] = {
            "sum_rank": float(mean_per_sample.sum().item()),
            "count": int(mean_per_sample.numel()),
            "slice_num": slice_num,
        }

    return finalize_rank_series(running, label=label, rel_tol=rel_tol, n_samples=n_samples)


def _maybe_overlay_rank_series(series: dict, overlay_path: str) -> list:
    """Build list of series for one figure; overlay json may be a single series or {series:[...]}."""
    out = [series]
    path = (overlay_path or "").strip()
    if not path:
        return out
    if not os.path.isfile(path):
        print(f"    [!] attn_rank_overlay not found: {path}")
        return out
    raw = load_attn_rank_json(path)
    if isinstance(raw, dict) and "series" in raw and isinstance(raw["series"], list):
        for s in raw["series"]:
            if s.get("label") != series.get("label"):
                out.append(s)
    elif isinstance(raw, dict) and "layers" in raw:
        if raw.get("label") != series.get("label"):
            out.append(raw)
    else:
        print(f"    [!] unrecognized attn_rank_overlay format: {path}")
    return out


def run_evaluation():
    parser = build_argparser()
    parser.add_argument("--case_ids", type=str, default="", help="Comma-separated list of case IDs to plot. Can be global case IDs (e.g., '0,5,10') or test set indices prefixed with 't:' (e.g., 't:0,t:5,t:10')")
    parser.add_argument("--cylinder_physics", action="store_true", help="Ref-grid cylinder: vorticity field point-wise abs/rel errors")
    parser.add_argument("--num_vorticity_plot", type=int, default=6, help="Number of vorticity comparison plots")
    args = parser.parse_args()
    args = load_yaml_config(args, parser=parser)
    
    # Auto-detect config.yaml from checkpoint directory if cache_path is empty
    if not args.cache_path and args.load_ckpt:
        checkpoint_dir = os.path.dirname(os.path.dirname(args.load_ckpt))  # checkpoints/ -> run_dir/
        auto_config = os.path.join(checkpoint_dir, "config.yaml")
        if os.path.exists(auto_config):
            print(f"[*] Auto-loading config from checkpoint directory: {auto_config}")
            args.config = auto_config
            args = load_yaml_config(args, parser=parser)
    
    # Validate cache_path before proceeding
    if not args.cache_path:
        error_msg = "cache_path is required but not provided.\n"
        error_msg += "Please provide one of:\n"
        error_msg += "  1. --cache_path <path>\n"
        error_msg += "  2. --config <path> (with cache_path in data section)\n"
        error_msg += "  3. Ensure config.yaml exists in the checkpoint's run directory"
        raise ValueError(error_msg)
    
    args.ddp = False 
    args.eval_only = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"[步骤1] 初始化评估环境")
    print(f"{'='*80}")
    print(f"    设备: {device}")
    print(f"    Cache路径: {args.cache_path}")
    print(f"    mesh_instance_name: {getattr(args, 'mesh_instance_name', '')}")
    print(f"[✓] 初始化完成")

    # 1. Load Data
    print(f"\n{'='*80}")
    print(f"[步骤2] 加载数据")
    print(f"{'='*80}")
    pos, fx, y, input_dim, output_dim, case_params, sato_indices, case_t, cached_node_weight = load_keti_data(args)
    time_input = bool(int(getattr(args, "time_input", 0)))
    use_node_weight = bool(int(getattr(args, "use_node_weight", 0)))
    recompute_node_weight = bool(int(getattr(args, "recompute_node_weight", 0)))
    nw_method = resolve_node_weight_method(args)
    nw_path = resolve_node_weight_pathways(args)
    node_weight_in_attn = nw_path["attn"]
    node_weight_in_norm_fx = nw_path["norm_fx"]
    node_weight_in_norm_y = nw_path["norm_y"]
    node_weight_in_norm = nw_path["norm"]
    node_weight_in_loss = nw_path["loss"]
    if use_node_weight:
        print(
            f"[*] node_weight pathways: attn={int(node_weight_in_attn)} "
            f"norm_fx={int(node_weight_in_norm_fx)} norm_y={int(node_weight_in_norm_y)} "
            f"loss={int(node_weight_in_loss)} "
            f"(train pathways; eval RelL2 quadrature controlled separately)"
        )
    if use_node_weight and cached_node_weight is not None and not recompute_node_weight:
        print(
            f"[*] cache contains node_weight; reuse if method matches "
            f"(want={nw_method})"
        )
    elif use_node_weight and cached_node_weight is not None and recompute_node_weight:
        print(f"[*] recompute_node_weight=1 → rebuild method={nw_method}")
    if time_input:
        if case_t is None:
            raise ValueError("time_input=1 需要 cache 中的 't' 字段")
        if getattr(args, "model", "Transolver_1D") == "SATO":
            raise ValueError("SATO 不支持 Time_Input，请改用 model: Transolver_1D")
    B = pos.shape[0]
    
    print(f"\n[2.1] 数据基本信息")
    print(f"    总样本数: {B}")
    print(f"    pos shape: {pos.shape}")
    print(f"    fx shape: {fx.shape}, 输入维度: {input_dim}")
    print(f"    y shape: {y.shape}, 输出维度: {output_dim}")
    print(f"    pos范围: {_fmt_tensor_range(pos)}")
    print(f"    fx范围: {_fmt_tensor_range(fx)}")
    if time_input and case_t is not None:
        print(f"    t范围: {_fmt_tensor_range(case_t)}")
    print(f"    y范围: {_fmt_tensor_range(y)}")
    
    cache = _load_cache(args.cache_path)
    train_idx, val_idx, test_idx, split_mode = resolve_train_val_test_indices(
        cache,
        train_split=float(getattr(args, "train_split", 0.8)),
        val_split=float(getattr(args, "val_split", 0.0)),
        seed=int(getattr(args, "seed", 42)),
        num_samples=B,
        case_names=getattr(args, "filtered_case_names", None),
        geom_id=getattr(args, "filtered_geom_id", None),
        split_by=str(getattr(args, "split_by", "random")),
        train_geom_ids=getattr(args, "train_geom_ids", None),
        test_geom_ids=getattr(args, "test_geom_ids", None),
    )
    pos_test, fx_test, y_test = pos[test_idx], fx[test_idx], y[test_idx]
    params_test = case_params[test_idx] if case_params is not None else None
    
    print(f"\n[2.2] 数据集划分")
    print(f"    split mode: {split_mode}")
    print(f"    训练集: {len(train_idx)}个样本")
    if val_idx is not None:
        print(f"    验证集: {len(val_idx)}个样本（训练时 EvalRel，本脚本不评）")
    print(f"    测试集: {len(test_idx)}个样本（本脚本 RelL2）")
    if str(getattr(args, "split_by", "random")).lower() == "geom":
        if meta := cache.get("metadata", {}):
            if meta.get("train_cases") is not None:
                print(f"    train cases: {meta.get('train_cases')}")
                print(f"    test cases : {meta.get('test_cases')}")
            if meta.get("train_geom_ids") is not None:
                print(f"    train_geom_ids: {meta.get('train_geom_ids')}")
                print(f"    test_geom_ids : {meta.get('test_geom_ids')}")
    
    # 检查是否需要构建mask（如果数据有padding点）
    padding_enabled = bool(int(getattr(args, "padding", 0)))
    padding_value = float(getattr(args, "padding_value", 0.0))
    print(f"\n[2.3] Padding检测")
    print(f"    Padding enabled: {padding_enabled}")
    if padding_enabled:
        mask_test = build_padding_mask(pos_test, fx_test, y_test, padding_value)
        mask_all = build_padding_mask(pos, fx, y, padding_value)
        total_test_points = mask_test.numel()
        valid_test_points = mask_test.sum().item()
        padding_test_points = total_test_points - valid_test_points
        print(f"    Padding value: {padding_value}")
        print(f"    测试集总点数: {total_test_points}")
        print(f"    有效点数: {valid_test_points} ({100*valid_test_points/total_test_points:.2f}%)")
        print(f"    Padding点数: {padding_test_points} ({100*padding_test_points/total_test_points:.2f}%)")
    else:
        mask_test = None
        mask_all = None
        print(f"    Padding未启用，所有点都视为有效点")

    node_weight = None
    nw_test = None
    _cache_meta = cache.get("metadata", {}) if isinstance(cache, dict) else {}
    if not isinstance(_cache_meta, dict):
        _cache_meta = {}
    _cache_nw_method = str(_cache_meta.get("node_weight_method", ""))
    if (
        use_node_weight
        and nw_method is not None
        and cached_node_weight is not None
        and not recompute_node_weight
        and tuple(cached_node_weight.shape) == tuple(pos.shape[:2])
    ):
        if cache_method_matches(_cache_nw_method, nw_method):
            node_weight = cached_node_weight
            nw_test = node_weight[test_idx]
            print(
                f"[*] reuse cache node_weight method={nw_method} "
                f"shape={tuple(node_weight.shape)} meta={_cache_nw_method or 'unspecified'}"
            )
        else:
            print(
                f"[!] want method={nw_method} but cache meta={_cache_nw_method!r}; "
                "will recompute"
            )

    print(f"[✓] 数据加载完成")

    # 2. Normalization Alignment
    print(f"\n{'='*80}")
    print(f"[步骤3] 加载归一化参数")
    print(f"{'='*80}")
    coord_stats_path = getattr(args, "coord_norm_path", "")
    # 若提供了 load_ckpt，优先从 checkpoint 对应 run 的 config 恢复 frame_stride 等
    if args.load_ckpt:
        ckpt_run_dir = os.path.dirname(os.path.dirname(args.load_ckpt))
        if not getattr(args, "run_dir", ""):
            args.run_dir = ckpt_run_dir
        ckpt_config = os.path.join(ckpt_run_dir, "config.yaml")
        if os.path.exists(ckpt_config):
            with open(ckpt_config, "r", encoding="utf-8") as f:
                ckpt_cfg = yaml.safe_load(f)
            ckpt_data = (ckpt_cfg or {}).get("data") or {}
            for key in ("frame_stride", "frame_offset", "max_frames", "cache_path", "mesh_instance_name"):
                if key in ckpt_data and ckpt_data[key] is not None:
                    setattr(args, key, ckpt_data[key])
            ckpt_coord = ckpt_data.get("coord_norm_path", "")
            if ckpt_coord:
                args.coord_norm_path = str(ckpt_coord)
                print(f"    [*] coord_norm 从 checkpoint run config: {ckpt_coord}")
            else:
                # 新约定：优先本 run 目录下的 coord_norm.pt
                local_cn = os.path.join(ckpt_run_dir, "coord_norm.pt")
                if os.path.isfile(local_cn):
                    args.coord_norm_path = local_cn
                    print(f"    [*] coord_norm 从 run_dir: {local_cn}")
    coord_stats_path = resolve_coord_norm_path(args)
    # Prefer local run coord_norm whenever it exists (even if resolve fell back)
    if args.load_ckpt:
        _local = os.path.join(os.path.dirname(os.path.dirname(args.load_ckpt)), "coord_norm.pt")
        if os.path.isfile(_local):
            coord_stats_path = _local
            args.coord_norm_path = _local
    if not os.path.isfile(coord_stats_path):
        # 旧 run：run_dir 尚无 coord_norm 时回退到 cache 旁文件
        cache_auto = default_coord_norm_path(
            str(getattr(args, "cache_path", "") or ""),
            frame_stride=int(getattr(args, "frame_stride", 1)),
            frame_offset=int(getattr(args, "frame_offset", 0)),
            max_frames=int(getattr(args, "max_frames", 0)),
            user_path="",
        )
        if cache_auto and os.path.isfile(cache_auto):
            print(
                f"    [!] {coord_stats_path} 不存在，回退 legacy cache coord_norm: {cache_auto}"
            )
            print(
                "    [!] 警告: 共享 coord_norm 可能被后写实验覆盖，"
                "和本次训练时的统计不一定一致"
            )
            coord_stats_path = cache_auto
            args.coord_norm_path = cache_auto

    print(f"\n[3.1] 归一化参数文件路径")
    print(f"    coord_norm_path: {coord_stats_path}")

    normalization_mask = mask_all[train_idx] if padding_enabled and mask_all is not None else None

    def _ensure_normalized_node_weight(pos_normalizer_local):
        nonlocal node_weight, nw_test
        if node_weight is not None or not use_node_weight:
            return
        cache_mask = getattr(args, "filtered_cache_mask", None)
        if nw_method == "mesh":
            fwd_lump = str(getattr(args, "node_weight_lumping", "") or "").strip().lower()
            if fwd_lump:
                node_weight = load_or_compute_mesh_node_weight(
                    pos,
                    lumping=fwd_lump,
                    cache_path=str(getattr(args, "cache_path", "")),
                    cached_node_weight=cached_node_weight,
                    cache_meta=_cache_meta,
                    cache_mask=cache_mask,
                    padding_enabled=padding_enabled,
                    padding_value=padding_value,
                    abaqus_root=str(getattr(args, "abaqus_root", "")),
                    npz_dir=str(getattr(args, "npz_dir", "")),
                    instance_name=str(getattr(args, "mesh_instance_name", "QJT-1")),
                )
            else:
                node_weight = compute_node_weight_from_abaqus_mesh(
                    pos,
                    cache_mask=cache_mask,
                    padding_enabled=padding_enabled,
                    padding_value=padding_value,
                    abaqus_root=str(getattr(args, "abaqus_root", "")),
                    npz_dir=str(getattr(args, "npz_dir", "")),
                    instance_name=str(getattr(args, "mesh_instance_name", "QJT-1")),
                    persist_path="",
                    persist_source_pos_shape=tuple(pos.shape),
                )
            nw_test = node_weight[test_idx]
            print(f"[*] method=mesh: node_weight shape={tuple(node_weight.shape)}")
            return
        pos_all_n_local = pos_normalizer_local.encode(pos)
        max_edge = normalize_node_weight_max_edge(getattr(args, "node_weight_max_edge", None))
        node_weight = compute_node_weight_from_normalized_pos(
            pos_all_n_local,
            use_node_weight=True,
            cache_mask=cache_mask,
            padding_enabled=padding_enabled,
            padding_value=padding_value,
            max_edge=max_edge,
            persist_path="",
            persist_source_pos_shape=tuple(pos.shape),
        )
        nw_test = node_weight[test_idx]
        print(
            f"[*] method=delaunay: node_weight from normalized pos "
            f"shape={tuple(node_weight.shape)}"
        )

    def _train_unit_normalizer(data, *, for_fx: bool = False):
        use_q = node_weight_in_norm_fx if for_fx else node_weight_in_norm_y
        nw_train = (
            node_weight[train_idx]
            if use_node_weight and use_q and node_weight is not None
            else None
        )
        return UnitTransformer(data, node_weight=nw_train, mask=normalization_mask)
    
    # 加载归一化参数（优先从coord_norm文件加载）
    saved_coord = None
    if os.path.exists(coord_stats_path):
        print(f"    ✓ 文件存在，开始加载")
        try:
            saved = torch.load(coord_stats_path, map_location="cpu", weights_only=True)
        except TypeError:
            saved = torch.load(coord_stats_path, map_location="cpu")
        saved_coord = saved
        
        # 加载坐标归一化参数
        print(f"\n[3.2] 加载坐标归一化参数")
        if "min" in saved and "max" in saved:
            pos_normalizer = IdentityTransformer(min_val=saved["min"], max_val=saved["max"])
            print(f"    ✓ 从coord_norm文件加载pos归一化参数")
            print(f"    min shape: {saved['min'].shape}, max shape: {saved['max'].shape}")
            print(f"    min range: [{saved['min'].min():.6f}, {saved['min'].max():.6f}]")
            print(f"    max range: [{saved['max'].min():.6f}, {saved['max'].max():.6f}]")
        else:
            pos_normalizer = IdentityTransformer(pos[train_idx])
            print(f"    [!] coord_norm文件缺少pos统计信息，从训练数据计算")
            print(f"    训练集pos shape: {pos[train_idx].shape}")
        _ensure_normalized_node_weight(pos_normalizer)
        
        # 加载fx/y归一化参数（如果存在）
        print(f"\n[3.3] 加载fx/y归一化参数")
        if "y_mean" in saved and "y_std" in saved:
            y_normalizer = create_unit_transformer_from_stats(saved["y_mean"], saved["y_std"])
            if input_dim > 0 and "fx_mean" in saved and "fx_std" in saved:
                print(f"    ✓ 从coord_norm文件加载fx/y归一化参数")
                fx_normalizer = create_unit_transformer_from_stats(saved["fx_mean"], saved["fx_std"])
            else:
                fx_normalizer = None
                print(f"    ✓ 从coord_norm文件加载y归一化参数（empty_fx，跳过fx）")
            print(f"    y_mean shape: {y_normalizer.mean.shape}, y_std shape: {y_normalizer.std.shape}")
            print(f"    y_mean range: [{y_normalizer.mean.min():.6f}, {y_normalizer.mean.max():.6f}]")
            print(f"    y_std range: [{y_normalizer.std.min():.6f}, {y_normalizer.std.max():.6f}]")
            if fx_normalizer is not None:
                print(f"    fx_mean shape: {fx_normalizer.mean.shape}, fx_std shape: {fx_normalizer.std.shape}")
            print(
                f"    normalization_node_weighted: "
                f"{saved.get('normalization_node_weighted', 'legacy/unspecified')}"
            )
        else:
            print(f"    [!] coord_norm文件缺少y统计信息，从训练数据计算")
            print(f"    训练集y shape: {y[train_idx].shape}")
            fx_normalizer = (
                _train_unit_normalizer(fx[train_idx], for_fx=True) if input_dim > 0 else None
            )
            y_normalizer = _train_unit_normalizer(y[train_idx], for_fx=False)
            if fx_normalizer is not None:
                print(f"    ✓ fx归一化器: mean shape={fx_normalizer.mean.shape}, std shape={fx_normalizer.std.shape}")
            print(f"    ✓ y归一化器: mean shape={y_normalizer.mean.shape}, std shape={y_normalizer.std.shape}")
    else:
        # coord_norm文件不存在，从训练数据计算
        print(f"    [!] coord_norm文件不存在: {coord_stats_path}")
        print(f"    从训练数据计算所有归一化参数")
        print(f"    训练集pos shape: {pos[train_idx].shape}")
        print(f"    训练集fx shape: {fx[train_idx].shape}")
        print(f"    训练集y shape: {y[train_idx].shape}")
        pos_normalizer = IdentityTransformer(pos[train_idx])
        _ensure_normalized_node_weight(pos_normalizer)
        fx_normalizer = (
            _train_unit_normalizer(fx[train_idx], for_fx=True) if input_dim > 0 else None
        )
        y_normalizer = _train_unit_normalizer(y[train_idx], for_fx=False)
        print(f"    ✓ pos归一化器: min shape={pos_normalizer.min.shape}, max shape={pos_normalizer.max.shape}")
        if fx_normalizer is not None:
            print(f"    ✓ fx归一化器: mean shape={fx_normalizer.mean.shape}, std shape={fx_normalizer.std.shape}")
        print(f"    ✓ y归一化器: mean shape={y_normalizer.mean.shape}, std shape={y_normalizer.std.shape}")
    
    if input_dim == 0:
        fx_normalizer = None
        print(f"    input_dim=0 (empty_fx): 跳过 fx 归一化")
    if use_node_weight and node_weight is None:
        _ensure_normalized_node_weight(pos_normalizer)

    # Forward (attn) q must match the run's training lumping. Shared cache.pt
    # only stores one mesh formula and is often overwritten by later runs.
    fwd_lump = str(getattr(args, "node_weight_lumping", "") or "").strip().lower()
    if not fwd_lump and isinstance(saved_coord, dict):
        _nw_meta = saved_coord.get("normalization_node_weighted") or {}
        if isinstance(_nw_meta, dict) and _nw_meta.get("lumping"):
            fwd_lump = str(_nw_meta["lumping"]).strip().lower()
            print(f"[*] node_weight_lumping from coord_norm: {fwd_lump}")
    if use_node_weight and nw_method == "mesh" and fwd_lump:
        from utils.abaqus_mesh_weight import LUMPING_TO_METHOD, normalize_mesh_lumping

        lump = normalize_mesh_lumping(fwd_lump)
        want = LUMPING_TO_METHOD[lump]
        cache_method = str(_cache_meta.get("node_weight_method", ""))
        print(f"\n[3.3b] Forward (attn) mesh q: lumping={lump} ({want})")
        print(f"    cache metadata.node_weight_method={cache_method or '(empty)'}")
        if cache_method != want:
            print(
                "    [!] cache q != train lumping; load sidecar/compute "
                "(do not use overwritten cache for attn)"
            )
        node_weight = load_or_compute_mesh_node_weight(
            pos,
            lumping=lump,
            cache_path=str(getattr(args, "cache_path", "")),
            cached_node_weight=cached_node_weight,
            cache_meta=_cache_meta,
            cache_mask=getattr(args, "filtered_cache_mask", None),
            padding_enabled=padding_enabled,
            padding_value=padding_value,
            abaqus_root=str(getattr(args, "abaqus_root", "")),
            npz_dir=str(getattr(args, "npz_dir", "")),
            instance_name=str(getattr(args, "mesh_instance_name", "QJT-1")),
        )
        nw_test = node_weight[test_idx]
        q_pos = node_weight[node_weight > 0]
        print(
            f"    forward q: shape={tuple(node_weight.shape)} "
            f"min={q_pos.min().item():.3e} max={node_weight.max().item():.3e}"
        )

    # Eval RelL2 quadrature (independent of train node_weight_in_loss)
    eval_quad = bool(int(getattr(args, "eval_rel_l2_quadrature", 1)))
    eval_q_source = str(getattr(args, "eval_q_source", "mesh_equal") or "mesh_equal").strip().lower()
    eval_node_weight = None  # full [B,N]
    eval_nw_test = None
    if eval_quad and eval_q_source not in ("", "none", "0", "false"):
        print(f"\n[3.4] Eval RelL2 quadrature source: {eval_q_source}")
        if eval_q_source == "train":
            if node_weight is None:
                raise RuntimeError("eval_q_source=train but node_weight is None")
            eval_node_weight = node_weight
        elif eval_q_source == "delaunay":
            pos_all_n_tmp = pos_normalizer.encode(pos)
            eval_node_weight = compute_node_weight_from_normalized_pos(
                pos_all_n_tmp,
                use_node_weight=True,
                cache_mask=getattr(args, "filtered_cache_mask", None),
                padding_enabled=padding_enabled,
                padding_value=padding_value,
                max_edge=normalize_node_weight_max_edge(
                    getattr(args, "node_weight_max_edge", None)
                ),
                persist_path="",
                persist_source_pos_shape=tuple(pos.shape),
            )
        elif eval_q_source.startswith("mesh"):
            lump_map = {
                "mesh_equal": "equal",
                "mesh_equal_div_inc": "equal_div_inc",
                "mesh_divinc": "equal_div_inc",
                "mesh_hrz": "hrz",
                "mesh": "equal",
            }
            if eval_q_source not in lump_map:
                raise ValueError(f"unknown eval_q_source={eval_q_source}")
            lump = lump_map[eval_q_source]
            from utils.abaqus_mesh_weight import LUMPING_TO_METHOD

            want = LUMPING_TO_METHOD[lump]
            # Prefer matching cache / sidecar over full recompute
            side = args.cache_path[:-3] + f"_nw_{want.replace('abaqus_cax_', '').replace('_lumped_area', '')}.pt"
            loaded = None
            if (
                cached_node_weight is not None
                and tuple(cached_node_weight.shape) == tuple(pos.shape[:2])
                and str(_cache_meta.get("node_weight_method", "")) == want
            ):
                loaded = cached_node_weight
                print(f"    reuse cache node_weight for eval q ({want})")
            elif os.path.isfile(side):
                blob = torch.load(side, map_location="cpu", weights_only=False)
                loaded = blob["node_weight"] if isinstance(blob, dict) else blob
                print(f"    load eval q sidecar: {side}")
            if loaded is None:
                print(f"    computing eval q lumping={lump} ...")
                loaded = compute_node_weight_from_abaqus_mesh(
                    pos,
                    cache_mask=getattr(args, "filtered_cache_mask", None),
                    padding_enabled=padding_enabled,
                    padding_value=padding_value,
                    abaqus_root=str(getattr(args, "abaqus_root", "/home/wang/keti_abaqus_geo24_len")),
                    npz_dir=str(
                        getattr(args, "npz_dir", "/home/wang/keti_abaqus_geo24_len/cache_npz")
                    ),
                    instance_name=str(getattr(args, "mesh_instance_name", "QJT-1")),
                    persist_path="",
                    lumping=lump,
                )
                torch.save(
                    {
                        "node_weight": loaded.cpu().float(),
                        "method": want,
                        "lumping": lump,
                    },
                    side,
                )
                print(f"    saved eval q sidecar: {side}")
            eval_node_weight = loaded.float()
        else:
            raise ValueError(f"unknown eval_q_source={eval_q_source}")
        eval_nw_test = eval_node_weight[test_idx]
        print(
            f"    eval_q test: shape={tuple(eval_nw_test.shape)} "
            f"min={eval_nw_test[eval_nw_test > 0].min().item():.3e} "
            f"max={eval_nw_test.max().item():.3e}"
        )
    else:
        print("\n[3.4] Eval RelL2: pointwise (no quadrature)")

    t_normalizer = None
    t_test_n = None
    t_all_n = None
    if time_input and case_t is not None:
        t_normalizer = UnitTransformer(case_t[train_idx].view(-1, 1, 1))
        t_test_n = t_normalizer.encode(case_t[test_idx].view(-1, 1, 1)).squeeze(-1).squeeze(-1)
        t_all_n = t_normalizer.encode(case_t.view(-1, 1, 1)).squeeze(-1).squeeze(-1)
        print(f"    ✓ time_input: t_normalizer 已从训练集 case_t 计算")
    
    print(f"[✓] 归一化参数加载完成")
    
    # 2.5 Load kNN cache
    print(f"\n{'='*80}")
    print(f"[步骤4] 加载kNN缓存（如果需要）")
    print(f"{'='*80}")
    knn_idx_test = None
    knn_valid_test = None
    knn_idx_all = None
    knn_valid_all = None
    use_edge_conv = bool(int(getattr(args, "use_edge_conv", 0)))
    grad_w = float(getattr(args, "grad_loss_weight", 0.0))
    knn_enable = bool(int(getattr(args, "knn_enable", 1)))
    
    need_knn = (use_edge_conv or grad_w > 0.0) and knn_enable
    
    print(f"    use_edge_conv: {use_edge_conv}")
    print(f"    grad_loss_weight: {grad_w}")
    print(f"    knn_enable: {knn_enable}")
    print(f"    need_knn: {need_knn}")
    
    if need_knn:
        k = int(getattr(args, "grad_k", 8))
        knn_cache_path = getattr(args, "knn_cache_path", "")
        if not knn_cache_path:
            base = args.cache_path.replace(".pt", "") if args.cache_path else "knn_cache"
            knn_cache_path = f"{base}_knn_k{k}.pt"
        
        print(f"    kNN参数k: {k}")
        print(f"    kNN缓存路径: {knn_cache_path}")
        
        if os.path.exists(knn_cache_path):
            print(f"    ✓ kNN缓存文件存在，开始加载")
            saved = torch.load(knn_cache_path, map_location="cpu")
            if isinstance(saved, dict) and saved.get("k") == k:
                knn_idx_all = saved.get("knn_idx")
                knn_valid_all = saved.get("knn_valid")
                if knn_idx_all is not None and knn_valid_all is not None:
                    knn_idx_test = knn_idx_all[test_idx]
                    knn_valid_test = knn_valid_all[test_idx]
                    print(f"    ✓ kNN缓存加载成功")
                    print(f"    knn_idx_test shape: {knn_idx_test.shape}")
                    print(f"    knn_valid_test shape: {knn_valid_test.shape}")
                else:
                    print(f"    [!] kNN缓存文件中缺少knn_idx或knn_valid")
            else:
                print(f"    [!] kNN缓存文件的k参数不匹配（期望{k}，实际{saved.get('k')}）")
        else:
            print(f"    [!] kNN缓存文件不存在: {knn_cache_path}")
    else:
        print(f"    kNN未启用，跳过")
    
    print(f"[✓] kNN缓存处理完成")

    # 3. Prepare Model
    print(f"\n{'='*80}")
    print(f"[步骤5] 加载模型")
    print(f"{'='*80}")
    print(f"\n[5.1] 创建模型")
    print(f"    输入维度: {input_dim}")
    print(f"    输出维度: {output_dim}")
    model = MultiNetWrapper(args, input_dim=input_dim, output_dim=output_dim).to(device)
    print(f"    ✓ 模型创建完成，已移动到设备: {device}")
    
    print(f"\n[5.2] 查找checkpoint")
    load_path = args.load_ckpt
    if not load_path:
        ckpt_dir = "checkpoints"
        best_eval_path = os.path.join(ckpt_dir, f"{args.save_name}_best_eval.pt")
        load_path = best_eval_path if os.path.exists(best_eval_path) else None
        if load_path:
            print(f"    自动找到checkpoint: {load_path}")
        else:
            print(f"    未找到默认checkpoint: {best_eval_path}")
    else:
        print(f"    使用指定的checkpoint: {load_path}")
            
    if load_path:
        print(f"    开始加载checkpoint...")
        maybe_load_checkpoint(model, load_path, device=device, is_main_process=True)
        print(f"    ✓ Checkpoint加载成功: {load_path}")
    else:
        print("[!] 未找到checkpoint，请提供 --load_ckpt")
        return
    
    print(f"[✓] 模型准备完成")

    # 4. Phase A: Efficient Batch Inference
    print(f"\n{'='*80}")
    print(f"[步骤6] 数据归一化与推理准备")
    print(f"{'='*80}")
    print(f"\n[6.1] 归一化测试数据")
    pos_test_n = pos_normalizer.encode(pos_test)
    fx_test_n = fx_normalizer.encode(fx_test) if fx_normalizer is not None else fx_test
    y_test_n = y_normalizer.encode(y_test)
    
    print(f"    pos_test_n shape: {pos_test_n.shape}")
    print(f"    fx_test_n shape: {fx_test_n.shape}")
    print(f"    y_test_n shape: {y_test_n.shape}")
    
    # 如果padding_enabled，统计有效点的归一化值范围
    if padding_enabled and mask_test is not None:
        valid_pos_n = pos_test_n[mask_test.bool()]
        valid_fx_n = fx_test_n[mask_test.bool()]
        valid_y_n = y_test_n[mask_test.bool()]
        print(f"    有效点归一化值范围:")
        print(f"      pos_n: {_fmt_tensor_range(valid_pos_n)}")
        print(f"      fx_n: {_fmt_tensor_range(valid_fx_n)}")
        print(f"      y_n: {_fmt_tensor_range(valid_y_n)}")
    else:
        print(f"    归一化值范围:")
        print(f"      pos_n: {_fmt_tensor_range(pos_test_n)}")
        print(f"      fx_n: {_fmt_tensor_range(fx_test_n)}")
        print(f"      y_n: {_fmt_tensor_range(y_test_n)}")
    
    # pos/fx/y 已是 test split；SATO order/inverse 仍是全量 [B_all, n_orders, N]，
    # 必须按 test_idx 切开，否则 TensorDataset 会 Size mismatch（训练侧同逻辑）。
    sato_test = None
    if sato_indices is not None:
        sato_test = {
            "order": sato_indices["order"][test_idx],
            "inverse": sato_indices["inverse"][test_idx],
        }
        print(
            f"    SATO test indices: order {tuple(sato_test['order'].shape)}, "
            f"inverse {tuple(sato_test['inverse'].shape)}"
        )

    def _build_test_ds():
        return _build_keti_tensor_dataset(
            pos_test_n,
            fx_test_n,
            y_test_n,
            indices=None,
            padding_enabled=padding_enabled,
            mask_all=mask_test,
            node_weight_all=nw_test if use_node_weight else None,
            knn_idx_all=knn_idx_test,
            knn_valid_all=knn_valid_test,
            sato_indices=sato_test,
            t_n=t_test_n if time_input else None,
        )

    sato_in_loader = sato_indices is not None

    print(f"\n[6.2] 创建DataLoader")
    test_ds = _build_test_ds()
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"    Batch size: {args.batch_size}")
    print(f"    总batch数: {len(test_loader)}")
    
    print(f"\n[6.3] 移动归一化器到设备")
    pos_normalizer.cuda() if device.type == "cuda" else None
    if fx_normalizer is not None:
        fx_normalizer.cuda() if device.type == "cuda" else None
    if t_normalizer is not None:
        t_normalizer.cuda() if device.type == "cuda" else None
    y_normalizer.cuda() if device.type == "cuda" else None
    model.eval()
    print(f"    ✓ 归一化器和模型已准备就绪")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = _resolve_eval_dir(args, timestamp)
    os.makedirs(eval_dir, exist_ok=True)

    def _run_attn_rank_phase_and_save():
        rel_tol = float(getattr(args, "attn_rank_rel_tol", 1e-3))
        use_linearno = bool(int(getattr(args, "use_linearno", 0)))
        label = "LinearNO" if use_linearno else "Transolver"
        print(f"    label={label}, rel_tol={rel_tol}, overlay={getattr(args, 'attn_rank_overlay', '') or '(none)'}")
        series = _collect_attn_rank_over_loader(
            model,
            test_loader,
            device,
            padding_enabled=padding_enabled,
            need_knn=need_knn,
            sato_in_loader=sato_in_loader,
            use_time_input=time_input,
            use_node_weight=use_node_weight,
            node_weight_in_attn=node_weight_in_attn,
            rel_tol=rel_tol,
            label=label,
        )
        print(
            f"    n_samples={series['n_samples']}, slice_num={series['slice_num']}, "
            f"layers={series['layers']}, mean_ranks={[round(x, 3) for x in series['mean_ranks']]}"
        )
        single_json = os.path.join(eval_dir, "plot", f"attn_rank_{label}.json")
        os.makedirs(os.path.dirname(single_json), exist_ok=True)
        save_attn_rank_json(series, single_json)
        series_list = _maybe_overlay_rank_series(series, getattr(args, "attn_rank_overlay", ""))
        png, js = save_attn_rank_curves(series_list, eval_dir, tag="attn_matrix_avg_rank", rel_tol=rel_tol)
        print(f"    ✓ rank 曲线: {png}")
        print(f"    ✓ rank json: {js} (本 run 单系列: {single_json})")
        return png, js

    attn_rank_only = bool(int(getattr(args, "attn_rank_only", 0)))
    plot_attn_rank = bool(int(getattr(args, "plot_attn_rank", 0))) or attn_rank_only
    if attn_rank_only:
        print(f"\n{'='*80}")
        print(f"[attn_rank_only] Attention 矩阵平均秩 (全 test) → {eval_dir}")
        print(f"{'='*80}")
        _run_attn_rank_phase_and_save()
        print(f"\n{'='*80}")
        print(f"[✓] attn_rank_only 完成！结果已保存到: {eval_dir}")
        print(f"{'='*80}")
        return

    slice_viz_only = bool(int(getattr(args, "slice_viz_only", 0)))
    if slice_viz_only:
        case_i = int(getattr(args, "slice_viz_case", 0))
        ch_i = int(getattr(args, "slice_viz_channel", 0))
        n_test = int(pos_test_n.shape[0])
        if case_i < 0 or case_i >= n_test:
            raise ValueError(f"slice_viz_case={case_i} out of range [0, {n_test})")
        global_id = int(test_idx[case_i].item()) if torch.is_tensor(test_idx) else int(test_idx[case_i])
        print(f"\n{'='*80}")
        print(f"[slice_viz_only] A soft-grid + B hard-assign → {eval_dir}")
        print(f"{'='*80}")
        print(f"    test_local={case_i}, global_case={global_id}, channel={ch_i}")
        print(f"    layers=all, last_layer_only=False")

        b_pos = pos_test_n[case_i : case_i + 1].to(device)
        b_fx = fx_test_n[case_i : case_i + 1].to(device)
        b_mask = None
        if padding_enabled and mask_test is not None:
            b_mask = mask_test[case_i : case_i + 1].to(device)
        b_nw = None
        if use_node_weight and nw_test is not None:
            b_nw = nw_test[case_i : case_i + 1].to(device)
        b_sato = None
        if sato_indices is not None:
            gid = int(test_idx[case_i].item()) if torch.is_tensor(test_idx) else int(test_idx[case_i])
            b_sato = {
                "order": sato_indices["order"][gid : gid + 1].to(device),
                "inverse": sato_indices["inverse"][gid : gid + 1].to(device),
            }

        if hasattr(model, "enable_attention_recording"):
            model.enable_attention_recording(True, channel_idx=ch_i)
        else:
            raise RuntimeError("model does not support enable_attention_recording")

        with torch.no_grad():
            b_fx_model = b_fx if b_fx.shape[-1] > 0 else None
            _ = model(
                b_pos,
                fx=b_fx_model,
                mask=b_mask,
                sato_indices=b_sato if sato_indices is not None else None,
                quadrature_weights=b_nw,
            )
            attn_list = model.get_last_attention(channel_idx=ch_i)

        if hasattr(model, "enable_attention_recording"):
            model.enable_attention_recording(False, channel_idx=ch_i)

        # 几何用物理坐标（与 paper case 图一致）
        pos_phys = pos_test[case_i].detach().cpu().numpy()
        mask_np = None
        if b_mask is not None:
            mask_np = b_mask[0].detach().cpu().numpy()
            if mask_np.ndim > 1:
                mask_np = mask_np.reshape(-1)

        ch_name = "ch0"
        out_cols = resolve_output_cols(getattr(args, "output_cols", None), output_dim)
        if ch_i < len(out_cols):
            ch_name = str(out_cols[ch_i])
        tag = f"slice_case{case_i}_g{global_id}_{ch_name}"
        print(f"    attn layers recorded: {len(attn_list)}")
        for it in attn_list:
            sw = it.get("slice_weights", None)
            shp = tuple(sw.shape) if isinstance(sw, torch.Tensor) else None
            print(f"      layer={it.get('layer')} slice_weights={shp}")

        plot_dir = save_attention_heatmaps(
            attn_list,
            tag=tag,
            base_dir=eval_dir,
            pos=pos_phys,
            plot_slice_grid=True,
            plot_hard_assign=True,
            last_layer_only=False,
            mask=mask_np,
            batch_index=0,
        )
        print(f"\n{'='*80}")
        print(f"[✓] slice_viz_only 完成！图目录: {plot_dir}")
        print(f"{'='*80}")
        return

    print(f"\n{'='*80}")
    print(f"[步骤7] 模型推理")
    print(f"{'='*80}")
    print(f"    测试样本数: {len(test_idx)}")
    print(f"    开始推理...")
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    inf_start = time.perf_counter()
    all_preds, all_trues = _run_keti_inference(
        model,
        test_loader,
        y_normalizer,
        device,
        padding_enabled=padding_enabled,
        need_knn=need_knn,
        sato_in_loader=sato_in_loader,
        use_time_input=time_input,
        use_node_weight=use_node_weight,
        node_weight_in_attn=node_weight_in_attn,
        num_samples=len(test_idx),
        log_prefix="test",
    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    inf_end = time.perf_counter()
    total_inf_time = inf_end - inf_start
    avg_latency = (total_inf_time / len(test_idx)) * 1000
    print(f"    ✓ 推理完成")
    print(f"    总耗时: {total_inf_time:.2f}秒")
    print(f"    平均延迟: {avg_latency:.2f}ms/样本")

    all_pos_np = pos_test.numpy()
    
    print(f"\n[7.1] 推理结果统计")
    print(f"    all_preds shape: {all_preds.shape}")
    print(f"    all_trues shape: {all_trues.shape}")
    print(f"    all_pos_np shape: {all_pos_np.shape}")
    if padding_enabled and mask_test is not None:
        mask_test_np = mask_test.numpy()
        valid_preds = all_preds[mask_test_np]
        valid_trues = all_trues[mask_test_np]
        print(f"    有效点预测值范围: [{valid_preds.min():.4f}, {valid_preds.max():.4f}]")
        print(f"    有效点真值范围: [{valid_trues.min():.4f}, {valid_trues.max():.4f}]")
    else:
        print(f"    预测值范围: [{all_preds.min():.4f}, {all_preds.max():.4f}]")
        print(f"    真值范围: [{all_trues.min():.4f}, {all_trues.max():.4f}]")

    # 5. Phase B: Detailed Metric Analysis (L2 and Linf)
    print(f"\n{'='*80}")
    print(f"[步骤8] 计算评估指标")
    print(f"{'='*80}")
    num_test, num_channels = len(test_idx), output_dim
    eps = 1e-12
    print(f"    测试样本数: {num_test}")
    print(f"    输出通道数: {num_channels}")
    print(f"    开始计算相对/绝对 L2 和 L-inf 误差...")
    
    output_cols = resolve_output_cols(getattr(args, "output_cols", None), num_channels)
    print(f"    输出通道名: {output_cols}")

    metrics_records = []
    sample_avg_l2 = [] # For ranking best/worst
    
    # Store true/pred peaks for scatter plot
    sample_true_peaks = np.zeros((num_test, num_channels))
    sample_pred_peaks = np.zeros((num_test, num_channels))

    print(f"\n[8.1] 逐样本计算误差")
    for i in range(num_test):
        ch_l2_list = []
        # 如果padding_enabled，只考虑有效点
        if padding_enabled and mask_test is not None:
            valid_mask_i = mask_test[i].numpy()  # [N]
            y_p_valid = all_preds[i, valid_mask_i, :]  # [N_valid, C]
            y_t_valid = all_trues[i, valid_mask_i, :]  # [N_valid, C]
            q_i = (
                None
                if eval_nw_test is None
                else eval_nw_test[i].numpy()[valid_mask_i].astype(np.float64)
            )
            num_valid_points = valid_mask_i.sum()
        else:
            y_p_valid = all_preds[i]  # [N, C]
            y_t_valid = all_trues[i]  # [N, C]
            q_i = (
                None
                if eval_nw_test is None
                else eval_nw_test[i].numpy().astype(np.float64)
            )
            num_valid_points = y_p_valid.shape[0]
        
        for c in range(num_channels):
            y_p = y_p_valid[:, c]
            y_t = y_t_valid[:, c]
            diff = y_p - y_t
            
            # Pointwise RelL2 (legacy)
            abs_l2 = float(np.linalg.norm(diff))
            l2_pt = float(abs_l2 / (np.linalg.norm(y_t) + eps))
            # Quadrature RelL2 (official when eval_quad)
            if q_i is None:
                l2 = l2_pt
            else:
                num = float(np.sum(q_i * diff**2))
                den = float(np.sum(q_i * y_t**2))
                l2 = float(np.sqrt(num + eps) / (np.sqrt(den + eps)))
            # Relative Linf (Max pointwise error relative to max case value)
            abs_linf = float(np.max(np.abs(diff)))
            linf = float(abs_linf / (np.max(np.abs(y_t)) + eps))
            # Point-wise MSE/MAE; NMSE follows official RelL2 definition
            mse = float(np.mean(diff ** 2))
            mae = float(np.mean(np.abs(diff)))
            if q_i is None:
                nmse = float(np.sum(diff ** 2) / (np.sum(y_t ** 2) + eps))
            else:
                nmse = float(np.sum(q_i * diff**2) / (np.sum(q_i * y_t**2) + eps))
            
            ch_l2_list.append(l2)
            col_name = output_cols[c] if c < len(output_cols) else f"Ch{c}"
            
            metrics_records.append({'Case': test_idx[i].item(), 'Channel': col_name, 'Metric': 'RelL2', 'Value': l2})
            metrics_records.append({'Case': test_idx[i].item(), 'Channel': col_name, 'Metric': 'RelL2_pointwise', 'Value': l2_pt})
            metrics_records.append({'Case': test_idx[i].item(), 'Channel': col_name, 'Metric': 'RelLinf', 'Value': linf})
            metrics_records.append({'Case': test_idx[i].item(), 'Channel': col_name, 'Metric': 'AbsL2', 'Value': abs_l2})
            metrics_records.append({'Case': test_idx[i].item(), 'Channel': col_name, 'Metric': 'AbsLinf', 'Value': abs_linf})
            metrics_records.append({'Case': test_idx[i].item(), 'Channel': col_name, 'Metric': 'MSE', 'Value': mse})
            metrics_records.append({'Case': test_idx[i].item(), 'Channel': col_name, 'Metric': 'MAE', 'Value': mae})
            metrics_records.append({'Case': test_idx[i].item(), 'Channel': col_name, 'Metric': 'NMSE', 'Value': nmse})
            
            sample_true_peaks[i, c] = float(np.max(y_t_valid[:, c]))
            sample_pred_peaks[i, c] = float(np.max(y_p_valid[:, c]))
            
        sample_avg_l2.append(np.mean(ch_l2_list))
    
    metrics_df = pd.DataFrame(metrics_records)
    sample_avg_l2 = np.array(sample_avg_l2)
    if eval_quad and eval_nw_test is not None:
        print(
            f"    RelL2 = quadrature ({eval_q_source}); "
            f"RelL2_pointwise also recorded"
        )
    else:
        print("    RelL2 = pointwise")
    
    # 6. Setup Output Directory (eval_dir already created before inference)
    print(f"    评估输出目录: {eval_dir}")
    metrics_csv = os.path.join(eval_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"    ✓ metrics.csv: {metrics_csv}")

    # 8.2 RelL2 vs 物理时间
    print(f"\n{'='*80}")
    print(f"[步骤8.2] RelL2 vs 时间散点图")
    print(f"{'='*80}")
    test_idx_np = test_idx.numpy() if isinstance(test_idx, torch.Tensor) else np.asarray(test_idx)
    train_idx_np = train_idx.numpy() if isinstance(train_idx, torch.Tensor) else np.asarray(train_idx)
    mask_test_np = mask_test.numpy() if padding_enabled and mask_test is not None else None

    print(f"\n[8.2.1] 测试集 {len(test_idx_np)} 帧")
    t_test = _get_time_for_indices(cache, fx, test_idx)
    node_weight_test_np = (
        eval_nw_test.numpy()
        if eval_nw_test is not None
        else (nw_test.numpy() if use_node_weight and nw_test is not None else None)
    )
    # Scatter uses same q as official RelL2 when quadrature enabled
    if eval_quad and eval_nw_test is not None:
        node_weight_test_np = eval_nw_test.numpy()
    elif not eval_quad:
        node_weight_test_np = None
    rel_ch_test, rel_mean_test = compute_sample_rel_l2(
        all_preds, all_trues, mask_test_np, node_weight=node_weight_test_np
    )
    test_plot = plot_rel_vs_time_scatter(
        t_test,
        rel_mean_test,
        eval_dir,
        "rel_vs_time_test_scatter.png",
        f"Test Set RelL2 vs Time (n={len(test_idx_np)})",
    )
    print(f"    ✓ 图1 测试集散点: {test_plot}")
    print(
        f"    RelL2_mean: min={rel_mean_test.min():.4f}, "
        f"median={np.median(rel_mean_test):.4f}, max={rel_mean_test.max():.4f}"
    )

    print(f"\n[8.2.2] 全量 {B} 帧推理")
    if not bool(int(getattr(args, "full_rel_scatter", 1))):
        print("    full_rel_scatter=0 → skip")
    else:
        bs_enc = int(getattr(args, "batch_size", 1) or 1)
        print(
            f"    encode in chunks: B={B} batch_size={bs_enc} "
            f"(avoid full fx.to(cuda))",
            flush=True,
        )
        pos_all_n = _encode_tensor_batched(pos_normalizer, pos, device, bs_enc)
        fx_all_n = (
            _encode_tensor_batched(fx_normalizer, fx, device, bs_enc)
            if fx_normalizer is not None
            else fx
        )
        y_all_n = _encode_tensor_batched(y_normalizer, y, device, bs_enc)
        full_ds = _build_keti_tensor_dataset(
            pos_all_n,
            fx_all_n,
            y_all_n,
            indices=None,
            padding_enabled=padding_enabled,
            mask_all=mask_all,
            node_weight_all=node_weight if use_node_weight else None,
            knn_idx_all=knn_idx_all,
            knn_valid_all=knn_valid_all,
            sato_indices=sato_indices,
            t_n=t_all_n if time_input else None,
        )
        full_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False)
        print(f"    Batch size: {args.batch_size}, 总 batch 数: {len(full_loader)}")
        torch.cuda.synchronize() if device.type == "cuda" else None
        full_inf_start = time.perf_counter()
        all_preds_full, all_trues_full = _run_keti_inference(
            model,
            full_loader,
            y_normalizer,
            device,
            padding_enabled=padding_enabled,
            need_knn=need_knn,
            sato_in_loader=sato_in_loader,
            use_time_input=time_input,
            use_node_weight=use_node_weight,
            node_weight_in_attn=node_weight_in_attn,
            num_samples=B,
            log_prefix="all",
        )
        torch.cuda.synchronize() if device.type == "cuda" else None
        full_inf_time = time.perf_counter() - full_inf_start
        print(f"    ✓ 全量推理完成，耗时 {full_inf_time:.2f}s")

        t_all = _get_time_for_indices(cache, fx, np.arange(B))
        split_all = np.array(["train"] * B, dtype=object)
        split_all[test_idx_np] = "test"
        mask_all_np = mask_all.numpy() if padding_enabled and mask_all is not None else None
        if eval_quad and eval_node_weight is not None:
            node_weight_all_np = eval_node_weight.numpy()
        elif use_node_weight and node_weight is not None:
            node_weight_all_np = node_weight.numpy()
        else:
            node_weight_all_np = None
        if not eval_quad:
            node_weight_all_np = None
        rel_ch_all, rel_mean_all = compute_sample_rel_l2(
            all_preds_full, all_trues_full, mask_all_np, node_weight=node_weight_all_np
        )
        all_plot = plot_rel_vs_time_scatter(
            t_all,
            rel_mean_all,
            eval_dir,
            "rel_vs_time_all_scatter.png",
            f"All Frames RelL2 vs Time (n={B}, train={len(train_idx_np)}, test={len(test_idx_np)})",
            split=split_all,
        )
        print(f"    ✓ 图2 全量散点: {all_plot}")
        train_mask = split_all == "train"
        test_mask = split_all == "test"
        if train_mask.any():
            print(
                f"    train RelL2_mean: avg={rel_mean_all[train_mask].mean():.4f}, "
                f"median={np.median(rel_mean_all[train_mask]):.4f}"
            )
        else:
            print("    train RelL2_mean: (empty)")
        print(
            f"    test  RelL2_mean: avg={rel_mean_all[test_mask].mean():.4f}, "
            f"median={np.median(rel_mean_all[test_mask]):.4f}"
        )
        del all_preds_full, all_trues_full, full_loader, full_ds, pos_all_n, fx_all_n, y_all_n
    
    # 7. Phase C: Violin Distribution Plots
    print(f"\n{'='*80}")
    print(f"[步骤9] 生成可视化图表")
    print(f"{'='*80}")
    print(f"[9.1] 绘制误差分布图（Violin Plots）...")
    plot_violin_distribution(metrics_df, eval_dir, filename="error_distribution_violin.png")
    print(f"    ✓ 误差分布图已保存")

    # 8. Phase D: Extreme Case Analysis (Best & Worst)
    n_plot = max(0, int(getattr(args, "num_plot", 3)))
    sorted_indices = np.argsort(sample_avg_l2)  # Ascending order
    best_indices = sorted_indices[:n_plot] if n_plot > 0 else np.array([], dtype=int)
    worst_indices = (
        sorted_indices[-n_plot:][::-1] if n_plot > 0 else np.array([], dtype=int)
    )
    if n_plot <= 0:
        print("[*] Phase D/E: num_plot=0 → skip best/worst grids and detail plots")
    else:
        print(f"[*] Phase D: Identifying Top {n_plot} Best and Worst Cases...")
        # 使用测试集索引作为case_id（与eval_with_mask.py保持一致）
        best_mask_list = (
            [mask_test[idx].numpy() for idx in best_indices]
            if padding_enabled and mask_test is not None
            else None
        )
        worst_mask_list = (
            [mask_test[idx].numpy() for idx in worst_indices]
            if padding_enabled and mask_test is not None
            else None
        )
        plot_case_grid(
            [all_pos_np[idx] for idx in best_indices],
            [all_preds[idx] for idx in best_indices],
            [all_trues[idx] for idx in best_indices],
            [idx for idx in best_indices],
            [sample_avg_l2[idx] for idx in best_indices],
            output_cols,
            eval_dir,
            filename="best_cases_grid.png",
            title=f"Top {n_plot} Best Cases (Lowest RelL2)",
            padding_mask_list=best_mask_list,
        )
        plot_case_grid(
            [all_pos_np[idx] for idx in worst_indices],
            [all_preds[idx] for idx in worst_indices],
            [all_trues[idx] for idx in worst_indices],
            [idx for idx in worst_indices],
            [sample_avg_l2[idx] for idx in worst_indices],
            output_cols,
            eval_dir,
            filename="worst_cases_grid.png",
            title=f"Top {n_plot} Worst Cases (Highest RelL2)",
            padding_mask_list=worst_mask_list,
        )
        print(f"    最佳样本测试集索引: {best_indices[:n_plot]}")
        print(f"    最差样本测试集索引: {worst_indices[:n_plot]}")
        print(f"    ✓ 样本识别完成")

        print(f"\n[9.3] 绘制最佳/最差样本详细对比图...")
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for s_i in list(best_indices[:3]) + list(worst_indices[:3]):
                test_idx_local = s_i
                prefix = "best" if s_i in best_indices else "worst"
                futures.append(
                    executor.submit(
                        plot_combined_channels_atmospheric,
                        all_pos_np[s_i],
                        all_preds[s_i],
                        all_trues[s_i],
                        eval_dir,
                        prefix,
                        test_idx_local,
                        output_cols,
                        mask_test[s_i].numpy()
                        if padding_enabled and mask_test is not None
                        else None,
                    )
                )
            for f in futures:
                f.result()
        print(f"    ✓ 详细对比图已保存")

    # 9.5: Plot specified case IDs
    if args.case_ids:
        print(f"\n[9.5] 绘制指定case id的对比图...")
        case_specs = [x.strip() for x in args.case_ids.split(",") if x.strip()]
        print(f"    指定的case标识: {case_specs}")
        print(f"    测试集全局case id范围: {test_idx.min().item()} - {test_idx.max().item()}")
        print(f"    测试集内索引范围: 0 - {len(test_idx)-1}")
        
        # 解析case标识：支持两种格式
        # 1. 全局case id: "7" -> 全局case id 7
        # 2. 测试集内索引: "t:7" -> 测试集内第7个样本（对应eval_with_mask.py中的索引）
        found_cases = []
        test_idx_np = test_idx.numpy() if isinstance(test_idx, torch.Tensor) else test_idx
        
        for case_spec in case_specs:
            if case_spec.startswith("t:"):
                # 测试集内索引模式（对应eval_with_mask.py中的best_case/worst_case编号）
                try:
                    test_idx_local = int(case_spec[2:])
                    if 0 <= test_idx_local < len(test_idx):
                        s_i = test_idx_local  # 直接使用测试集内索引
                        global_case_id = test_idx_np[s_i]  # 获取对应的全局case id
                        found_cases.append((global_case_id, s_i, f"t:{test_idx_local}"))
                        case_l2 = sample_avg_l2[s_i]
                        print(f"    ✓ 找到测试集内索引 {test_idx_local} -> 全局case id {global_case_id}，在测试集中的位置: {s_i}，RelL2: {case_l2:.6f}")
                    else:
                        print(f"    [!] 测试集内索引 {test_idx_local} 超出范围 (0-{len(test_idx)-1})")
                except ValueError:
                    print(f"    [!] 无效的测试集索引格式: {case_spec}，应为 t:数字")
            else:
                # 全局case id模式
                try:
                    global_case_id = int(case_spec)
                    # 检查case_id是否在测试集中
                    test_pos = np.where(test_idx_np == global_case_id)[0]
                    if len(test_pos) > 0:
                        s_i = test_pos[0]  # 测试集中的索引（对应all_preds/all_trues的索引）
                        found_cases.append((global_case_id, s_i, str(global_case_id)))
                        # 获取该case的误差信息
                        case_l2 = sample_avg_l2[s_i]
                        print(f"    ✓ 找到全局case id {global_case_id}，在测试集中的位置: {s_i}（测试集内索引: {s_i}），RelL2: {case_l2:.6f}")
                    else:
                        print(f"    [!] 全局case id {global_case_id} 不在测试集中（测试集全局case id范围: {test_idx_np.min()} - {test_idx_np.max()}）")
                except ValueError:
                    print(f"    [!] 无效的case id格式: {case_spec}，应为数字或 t:数字")
        
        if found_cases:
            print(f"    开始绘制 {len(found_cases)} 个指定case的对比图...")
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = []
                for global_case_id, s_i, spec_str in found_cases:
                    # 统一使用测试集索引作为文件名（与eval_with_mask.py保持一致）
                    test_idx_local = s_i
                    futures.append(executor.submit(
                        plot_combined_channels_atmospheric, 
                        all_pos_np[s_i], all_preds[s_i], all_trues[s_i], 
                        eval_dir, "case", test_idx_local, output_cols,  # 使用测试集索引
                        mask_test[s_i].numpy() if padding_enabled and mask_test is not None else None,
                    ))
                for f in futures: 
                    f.result()
            print(f"    ✓ 指定case对比图已保存（文件名前缀: case_*，使用测试集索引）")
        else:
            print(f"    [!] 没有找到任何有效的case id，跳过绘制")
            print(f"    提示: 使用 't:7' 格式指定测试集内索引（对应eval_with_mask.py中的best_case7等）")
            print(f"    提示: 使用 '7' 格式指定全局case id（会自动转换为测试集索引）")

    # 10. Phase F: Peak Value Scatter Plots
    print(f"\n[9.4] 绘制峰值散点图...")
    for c in range(num_channels):
        col_name = output_cols[c] if c < len(output_cols) else f"Ch{c}"
        plot_peak_scatter(sample_true_peaks[:, c], sample_pred_peaks[:, c], eval_dir, "test", col_name)

    # 11. Final Reporting
    report_lines = [
        f"Evaluation Report - {timestamp}",
        f"Model: {load_path}",
        f"Test Set Size: {len(test_idx)}",
        f"RelL2 quadrature: {eval_q_source if (eval_quad and eval_nw_test is not None) else 'none (pointwise)'}",
        "-" * 50
    ]
    for c in range(num_channels):
        col_name = output_cols[c] if c < len(output_cols) else f"Ch{c}"
        ch_l2 = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'RelL2')]['Value']
        ch_l2_pt = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'RelL2_pointwise')]['Value']
        ch_linf = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'RelLinf')]['Value']
        ch_abs_l2 = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'AbsL2')]['Value']
        ch_abs_linf = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'AbsLinf')]['Value']
        ch_mse = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'MSE')]['Value']
        ch_mae = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'MAE')]['Value']
        ch_nmse = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'NMSE')]['Value']
        line = (
            f"Channel {c} ({col_name}): "
            f"Avg RelL2: {ch_l2.mean():.4f}, Avg RelL2_pointwise: {ch_l2_pt.mean():.4f}, "
            f"Avg RelLinf: {ch_linf.mean():.4f}, "
            f"Avg AbsL2: {ch_abs_l2.mean():.6e}, Avg AbsLinf: {ch_abs_linf.mean():.6e}, "
            f"Avg MSE: {ch_mse.mean():.6e}, Avg MAE: {ch_mae.mean():.6e}, Avg NMSE: {ch_nmse.mean():.6e}"
        )
        report_lines.append(line)
        print(line)

    with open(os.path.join(eval_dir, "summary_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    # Cylinder ref-grid: vorticity field
    cache_meta = {}
    try:
        cache_meta = (_load_cache(args.cache_path).get("metadata", {}) or {})
    except Exception:
        pass
    is_refgrid = "ref_grid" in str(cache_meta.get("task", ""))
    if args.cylinder_physics or is_refgrid:
        try:
            from scripts.transolver.viz.cylinder_refgrid_physics import run_cylinder_physics_eval
        except ModuleNotFoundError:
            print("[!] cylinder_refgrid_physics not available, skip step 9.5")
        else:
            print(f"\n{'='*80}")
            print(f"[步骤9.5] 圆柱绕流：涡量场点级误差")
            print(f"{'='*80}")
            is_delta = str(cache_meta.get("y_target", "")).lower() == "delta"
            cache_full = _load_cache(args.cache_path)
            cyl_summary = run_cylinder_physics_eval(
                all_pos_np=all_pos_np,
                all_preds=all_preds,
                all_trues=all_trues,
                fx_test=fx_test.numpy(),
                test_idx=test_idx,
                cache=cache_full,
                eval_dir=eval_dir,
                is_delta=is_delta,
                num_vorticity_plot=int(args.num_vorticity_plot),
            )
            report_lines.append("-" * 50)
            report_lines.append("Cylinder physics:")
            for k, v in cyl_summary.items():
                report_lines.append("  {}: {}".format(k, v))
            with open(os.path.join(eval_dir, "summary_report.txt"), "w") as f:
                f.write("\n".join(report_lines))
            print(f"    ✓ 结果目录: {cyl_summary['out_dir']}")
        
    print(f"    ✓ 峰值散点图已保存")

    # 写入 time.txt（论文用：time_average 留白，memory 为 checkpoint 显存 MB，inference time 为秒/样本）
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    memory_mb = param_bytes / (1024 * 1024)  # bytes -> MB
    inference_time_per_sample_s = avg_latency / 1000.0  # ms -> s
    time_txt_path = os.path.join(eval_dir, "time.txt")
    with open(time_txt_path, "w", encoding="utf-8") as f:
        f.write("time_average = \n")
        f.write(f"memory = {memory_mb:.2f}MB\n")
        f.write(f"inference time = {inference_time_per_sample_s:.6f}s/sample\n")
    print(f"    ✓ time.txt 已保存: {time_txt_path}")

    # 9.6: High Stress Region Error Analysis
    if int(getattr(args, "high_stress_analysis", 0)):
        print(f"\n{'='*80}")
        print(f"[步骤9.6] 高应力区域误差分析")
        print(f"{'='*80}")
        
        # 解析误差阈值 / 相对误差分母
        error_thresholds = [float(x.strip()) for x in args.error_thresholds.split(",") if x.strip()]
        rel_denoms = resolve_rel_err_denoms(
            output_cols,
            override=getattr(args, "rel_err_denoms", "") or None,
        )
        print(f"\n[9.6.1] 分析配置")
        if int(getattr(args, "per_channel_stress", 0)):
            print(f"    模式: 每个通道独立分析（使用各自通道的最大值）")
        else:
            print(f"    模式: 统一筛选（使用通道 {args.stress_channel_idx} 的最大值筛选所有通道）")
            print(f"    应力筛选通道索引: {args.stress_channel_idx} ({output_cols[args.stress_channel_idx] if args.stress_channel_idx < len(output_cols) else f'Ch{args.stress_channel_idx}'})")
        print(f"    应力范围: [{args.stress_lower_ratio:.2f}*max, {args.stress_upper_ratio:.2f}*max]")
        print(f"    误差阈值: {error_thresholds}")
        print(f"    相对误差分母: {list(rel_denoms)}  (local=除当地值, max=除case通道max)")

        # 准备数据列表
        all_pos_list = [all_pos_np[i] for i in range(num_test)]
        all_pred_list = [all_preds[i] for i in range(num_test)]
        all_true_list = [all_trues[i] for i in range(num_test)]
        mask_list = None
        if padding_enabled and mask_test is not None:
            mask_list = [mask_test[i].numpy() for i in range(num_test)]

        # HS quadrature: reuse eval RelL2 q when available
        nw_hs_list = None
        if eval_quad and eval_nw_test is not None:
            nw_hs_list = [eval_nw_test[i].numpy() for i in range(num_test)]
            print(
                f"\n[9.6.2] 高应力区域误差统计（含求积 q={eval_q_source}）..."
            )
        else:
            print(f"\n[9.6.2] 对所有测试样本进行高应力区域误差统计（无求积）...")
        stats = analyze_high_stress_region_error(
            all_pos_list=all_pos_list,
            all_pred_list=all_pred_list,
            all_true_list=all_true_list,
            channel_names=output_cols,
            stress_channel_idx=args.stress_channel_idx,
            lower_ratio=args.stress_lower_ratio,
            upper_ratio=args.stress_upper_ratio,
            error_thresholds=error_thresholds,
            padding_mask_list=mask_list,
            per_channel=bool(int(getattr(args, "per_channel_stress", 0))),
            relative_denoms=rel_denoms,
            node_weight_list=nw_hs_list,
        )

        print(f"\n[9.6.3] 生成统计报告...")
        report_lines = [
            "=" * 80,
            "高应力区域误差分析报告",
            "=" * 80,
            f"分析时间: {timestamp}",
            f"测试样本数: {num_test}",
            f"筛选模式: {'每个通道独立分析' if int(getattr(args, 'per_channel_stress', 0)) else '统一筛选（通道 ' + str(args.stress_channel_idx) + ': ' + (output_cols[args.stress_channel_idx] if args.stress_channel_idx < len(output_cols) else f'Ch{args.stress_channel_idx}') + '）'}",
            f"应力范围: [{args.stress_lower_ratio:.2f}*max, {args.stress_upper_ratio:.2f}*max]",
            f"误差阈值: {error_thresholds}",
            f"相对误差分母: {list(rel_denoms)}",
            "  local = |pred-true| / (|true| + eps)",
            "  max   = |pred-true| / (case_channel_max(|true|) + eps)",
            "RelL2_hs (无求积/pointwise): ||e||_2/||y||_2 on high-stress nodes;",
            "RelL2_hs_q (求积): sqrt(sum q e^2)/sqrt(sum q y^2) on HS nodes;",
            "lt*_pct: point count fraction; lt*_pct_q: mass fraction sum q / sum q_HS;",
            "  case-mean = mean over test cases; pooled = one norm over all HS points",
            f"HS quadrature source: {eval_q_source if nw_hs_list is not None else 'none'}",
            "=" * 80,
            "",
        ]

        csv_rows = []
        for c in range(num_channels):
            ch_stats = stats[c]
            ch_name = ch_stats["channel_name"]
            total_points = ch_stats["total_points"]
            high_stress_points = ch_stats["high_stress_points"]
            hs_l2_mean = ch_stats.get("rel_l2_pointwise_mean", float("nan"))
            hs_l2_med = ch_stats.get("rel_l2_pointwise_median", float("nan"))
            hs_l2_pool = ch_stats.get("rel_l2_pointwise_pooled", float("nan"))
            hs_l2q_mean = ch_stats.get("rel_l2_quad_mean", float("nan"))
            hs_l2q_med = ch_stats.get("rel_l2_quad_median", float("nan"))
            hs_l2q_pool = ch_stats.get("rel_l2_quad_pooled", float("nan"))

            report_lines.append(f"通道 {c} ({ch_name}):")
            report_lines.append(f"  总点数: {total_points:,}")
            report_lines.append(
                f"  高应力区域点数: {high_stress_points:,} "
                f"({100.0 * high_stress_points / max(total_points, 1):.2f}%)"
            )
            report_lines.append(
                f"  RelL2_hs (无求积/pointwise, case-mean): {hs_l2_mean:.6f}"
            )
            report_lines.append(
                f"  RelL2_hs (无求积/pointwise, case-median): {hs_l2_med:.6f}"
            )
            report_lines.append(
                f"  RelL2_hs (无求积/pointwise, pooled): {hs_l2_pool:.6f}"
            )
            report_lines.append(
                f"  RelL2_hs_q (求积, case-mean): {hs_l2q_mean:.6f}"
            )
            report_lines.append(
                f"  RelL2_hs_q (求积, case-median): {hs_l2q_med:.6f}"
            )
            report_lines.append(
                f"  RelL2_hs_q (求积, pooled): {hs_l2q_pool:.6f}"
            )

            print(f"    通道 {c} ({ch_name}):")
            print(f"      总点数: {total_points:,}")
            print(
                f"      高应力区域点数: {high_stress_points:,} "
                f"({100.0 * high_stress_points / max(total_points, 1):.2f}%)"
            )
            print(
                f"      RelL2_hs (无求积) case-mean={hs_l2_mean:.6f} "
                f"median={hs_l2_med:.6f} pooled={hs_l2_pool:.6f}"
            )
            print(
                f"      RelL2_hs_q (求积) case-mean={hs_l2q_mean:.6f} "
                f"median={hs_l2q_med:.6f} pooled={hs_l2q_pool:.6f}"
            )

            if high_stress_points > 0:
                for denom_name in rel_denoms:
                    dstat = ch_stats["denoms"][denom_name]
                    report_lines.append(f"  --- 分母={denom_name} ---")
                    report_lines.append(
                        f"    平均误差: {dstat['mean_error']:.6f} ({dstat['mean_error'] * 100:.2f}%)"
                    )
                    report_lines.append(
                        f"    中位数误差: {dstat['median_error']:.6f} ({dstat['median_error'] * 100:.2f}%)"
                    )
                    report_lines.append(
                        f"    最大误差: {dstat['max_error']:.6f} ({dstat['max_error'] * 100:.2f}%)"
                    )
                    report_lines.append(
                        f"    P95误差: {dstat['p95_error']:.6f} ({dstat['p95_error'] * 100:.2f}%)"
                    )
                    report_lines.append(
                        f"    P99误差: {dstat['p99_error']:.6f} ({dstat['p99_error'] * 100:.2f}%)"
                    )
                    report_lines.append("    误差阈值统计:")
                    for th in error_thresholds:
                        th_stats = dstat["threshold_stats"][th]
                        th_q = dstat.get("threshold_stats_q", {}).get(th, {})
                        q_pct = th_q.get("percentage", float("nan"))
                        report_lines.append(
                            f"      误差 < {th * 100:.0f}%: {th_stats['count']:,} 点 "
                            f"({th_stats['percentage']:.2f}%)"
                            f" | mass_q={q_pct:.2f}%"
                        )

                    lt5 = dstat["threshold_stats"].get(0.05, {}).get("percentage", float("nan"))
                    lt10 = dstat["threshold_stats"].get(0.10, {}).get("percentage", float("nan"))
                    lt5q = (
                        dstat.get("threshold_stats_q", {})
                        .get(0.05, {})
                        .get("percentage", float("nan"))
                    )
                    lt10q = (
                        dstat.get("threshold_stats_q", {})
                        .get(0.10, {})
                        .get("percentage", float("nan"))
                    )
                    print(
                        f"      [{denom_name}] P95={dstat['p95_error'] * 100:.2f}% "
                        f"mean={dstat['mean_error'] * 100:.2f}% "
                        f"lt5={lt5:.2f}% lt5_q={lt5q:.2f}% "
                        f"lt10={lt10:.2f}% lt10_q={lt10q:.2f}%"
                    )

                    row = {
                        "channel": ch_name,
                        "channel_idx": c,
                        "denom": denom_name,
                        "total_points": total_points,
                        "high_stress_points": high_stress_points,
                        "RelL2_hs_pointwise_mean": hs_l2_mean,
                        "RelL2_hs_pointwise_median": hs_l2_med,
                        "RelL2_hs_pointwise_pooled": hs_l2_pool,
                        "RelL2_hs_quad_mean": hs_l2q_mean,
                        "RelL2_hs_quad_median": hs_l2q_med,
                        "RelL2_hs_quad_pooled": hs_l2q_pool,
                        "mean": dstat["mean_error"],
                        "median": dstat["median_error"],
                        "max": dstat["max_error"],
                        "p95": dstat["p95_error"],
                        "p99": dstat["p99_error"],
                    }
                    for th in error_thresholds:
                        th_stats = dstat["threshold_stats"][th]
                        th_q = dstat.get("threshold_stats_q", {}).get(th, {})
                        row[f"lt{int(th * 100)}_count"] = th_stats["count"]
                        row[f"lt{int(th * 100)}_pct"] = th_stats["percentage"]
                        row[f"lt{int(th * 100)}_pct_q"] = th_q.get("percentage", float("nan"))
                    csv_rows.append(row)
            else:
                report_lines.append("  [!] 该通道没有找到高应力区域点")
                print("      [!] 该通道没有找到高应力区域点")

            report_lines.append("")

        stats_file = os.path.join(eval_dir, "high_stress_error_stats.txt")
        with open(stats_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\n    ✓ 统计报告已保存到: {stats_file}")

        if csv_rows:
            import csv as _csv

            csv_path = os.path.join(eval_dir, "high_stress_error_stats.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"    ✓ 统计CSV已保存到: {csv_path}")

        # 可选：对最佳/最差样本绘制详细图（用 primary denom）
        plot_cases = list(best_indices[:3]) + list(worst_indices[:3])
        if not plot_cases:
            print("\n[9.6.4] num_plot=0 → skip high-stress best/worst plots")
        else:
            print(f"\n[9.6.4] 绘制最佳/最差样本的高应力区域误差图...")
            plot_denom = rel_denoms[0]
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = []
                for s_i in plot_cases:
                    prefix = "best" if s_i in best_indices else "worst"
                    mask_i = None
                    if padding_enabled and mask_test is not None:
                        mask_i = mask_test[s_i].numpy()
                    futures.append(
                        executor.submit(
                            plot_high_stress_error_analysis,
                            all_pos_np[s_i],
                            all_preds[s_i],
                            all_trues[s_i],
                            eval_dir,
                            prefix,
                            s_i,
                            output_cols,
                            stress_channel_idx=args.stress_channel_idx,
                            lower_ratio=args.stress_lower_ratio,
                            upper_ratio=args.stress_upper_ratio,
                            padding_mask=mask_i,
                            per_channel=bool(int(getattr(args, "per_channel_stress", 0))),
                            relative_denom=plot_denom,
                        )
                    )
                for f in futures:
                    f.result()
            print(f"    ✓ 高应力区域误差图已保存 (denom={plot_denom})")

    # Phase: attention-matrix average rank (LinearNO Fig.4d-style)
    if plot_attn_rank:
        print(f"\n{'='*80}")
        print(f"[步骤] Attention 矩阵平均秩 (全 test, A=φψᵀ / WWᵀ)")
        print(f"{'='*80}")
        _run_attn_rank_phase_and_save()

    print(f"\n{'='*80}")
    print(f"[步骤10] 评估报告汇总")
    print(f"{'='*80}")
    for c in range(num_channels):
        col_name = output_cols[c] if c < len(output_cols) else f"Ch{c}"
        ch_l2 = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'RelL2')]['Value']
        ch_linf = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'RelLinf')]['Value']
        ch_abs_l2 = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'AbsL2')]['Value']
        ch_abs_linf = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'AbsLinf')]['Value']
        ch_mse = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'MSE')]['Value']
        ch_mae = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'MAE')]['Value']
        ch_nmse = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'NMSE')]['Value']
        avg_l2 = ch_l2.mean()
        avg_linf = ch_linf.mean()
        avg_abs_l2 = ch_abs_l2.mean()
        avg_abs_linf = ch_abs_linf.mean()
        avg_mse = ch_mse.mean()
        avg_mae = ch_mae.mean()
        avg_nmse = ch_nmse.mean()
        print(
            f"    通道 {c} ({col_name}): "
            f"平均RelL2: {avg_l2:.6f}, 平均RelLinf: {avg_linf:.6f}, "
            f"平均AbsL2: {avg_abs_l2:.6e}, 平均AbsLinf: {avg_abs_linf:.6e}, "
            f"平均MSE: {avg_mse:.6e}, 平均MAE: {avg_mae:.6e}, 平均NMSE: {avg_nmse:.6e}"
        )
        ch_l2_pt = metrics_df[(metrics_df['Channel'] == col_name) & (metrics_df['Metric'] == 'RelL2_pointwise')]['Value']
        if len(ch_l2_pt):
            print(f"             RelL2_pointwise: {ch_l2_pt.mean():.6f}")
    
    print(f"\n{'='*80}")
    print(f"[✓] 评估完成！结果已保存到: {eval_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_evaluation()
