from typing import Callable, Optional, Tuple

import torch


LossFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[object],
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ],
    Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]],
]


def build_padding_mask(pos_b: torch.Tensor, fx_b: torch.Tensor, y_b: torch.Tensor, padding_value: float) -> torch.Tensor:
    """
    Build a boolean mask for valid points (True=valid, False=padding).

    A point is considered padding if its pos coordinates are all equal to padding_value (typically 0.0).
    This is more reliable than checking pos/fx/y all together, since coordinates cannot be all zero for valid points.
    """
    pad = torch.tensor(padding_value, device=pos_b.device, dtype=pos_b.dtype)
    # 只检查pos是否全为padding_value（通常是0.0）
    pos_pad = (pos_b == pad).all(dim=-1)  # [B, N]
    return ~pos_pad  # True=valid, False=padding


def run_epoch(
    *,
    model,
    loader,
    device,
    y_normalizer,
    fx_normalizer,
    loss_fn: LossFn,
    optimizer=None,
    max_grad_norm: Optional[float] = None,
    max_batches: int = 0,
    padding_enabled: bool = False,
    padding_value: float = 0.0,
    use_knn: bool = False,
    use_sato: bool = False,
    use_time_input: bool = False,
    use_node_weight: bool = False,
    node_weight_in_attn: bool = True,
    node_weight_in_loss: bool = True,
    lr_scheduler=None,
) -> Tuple[float, Optional[list], float, float, Optional[list], Optional[list], float, float]:
    """
    Shared train/eval epoch runner for KETI MultiNet-style batches.
    
    Args:
        use_knn: If True, loader provides (..., knn_idx, knn_valid)
        use_sato: If True, loader provides (..., order, inverse)
        use_node_weight: If True, loader provides node_weight after mask
        node_weight_in_attn: pass q as quadrature_weights to the model
        node_weight_in_loss: pass q into loss_fn
    """
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    loss_sum = None  # scalar tensor on device
    loss_sum_ch = None  # [C] tensor on device or None
    rel_sum = None
    grad_sum = None
    rel_sum_ch = None
    grad_sum_ch = None
    mse_sum = None
    mae_sum = None
    batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            # Deterministic unpacking based on flags
            idx = 0
            pos_b = batch[idx].to(device, non_blocking=True); idx += 1
            fx_b = batch[idx].to(device, non_blocking=True); idx += 1
            y_b = batch[idx].to(device, non_blocking=True); idx += 1
            
            if padding_enabled:
                mask_b = batch[idx].to(device, non_blocking=True); idx += 1
            else:
                mask_b = None

            if use_node_weight:
                node_weight_b = batch[idx].to(device, non_blocking=True); idx += 1
            else:
                node_weight_b = None
                
            if use_knn:
                knn_idx = batch[idx].to(device, non_blocking=True); idx += 1
                knn_valid = batch[idx].to(device, non_blocking=True); idx += 1
            else:
                knn_idx = None
                knn_valid = None
                
            if use_sato:
                order_b = batch[idx].to(device, non_blocking=True); idx += 1
                inverse_b = batch[idx].to(device, non_blocking=True); idx += 1
                sato_indices = {"order": order_b, "inverse": inverse_b}
            else:
                sato_indices = None

            t_b = None
            if use_time_input:
                t_b = batch[idx].to(device, non_blocking=True); idx += 1

            if is_train:
                optimizer.zero_grad()

            # Forward pass
            model_fx = fx_b if fx_b.shape[-1] > 0 else None
            qw = node_weight_b if (node_weight_b is not None and node_weight_in_attn) else None
            out_n = model(
                pos_b,
                fx=model_fx,
                mask=mask_b,
                knn_idx=knn_idx,
                knn_valid=knn_valid,
                sato_indices=sato_indices,
                T=t_b,
                quadrature_weights=qw,
            )
            out = y_normalizer.decode(out_n)
            y_orig = y_normalizer.decode(y_b)

            loss_nw = node_weight_b if (node_weight_b is not None and node_weight_in_loss) else None
            try:
                loss, loss_ch, metrics = loss_fn(
                    out,
                    y_orig,
                    fx_b,
                    fx_normalizer,
                    mask_b,
                    pos_b,
                    knn_idx,
                    knn_valid,
                    loss_nw,
                )
            except TypeError:
                loss, loss_ch = loss_fn(out, y_orig, fx_b, fx_normalizer)
                metrics = None

            if is_train:
                loss.backward()
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

            # accumulate on-device; avoid .item() in loop
            if loss_sum is None:
                loss_sum = loss.detach()
                loss_sum_ch = loss_ch.detach() if isinstance(loss_ch, torch.Tensor) else None
            else:
                loss_sum = loss_sum + loss.detach()
                if isinstance(loss_ch, torch.Tensor) and loss_sum_ch is not None:
                    loss_sum_ch = loss_sum_ch + loss_ch.detach()

            if metrics is not None:
                rel = metrics.get("rel")
                grad = metrics.get("grad")
                rel_ch = metrics.get("rel_ch")
                grad_ch = metrics.get("grad_ch")
                if rel is not None:
                    rel_sum = rel.detach() if rel_sum is None else rel_sum + rel.detach()
                if grad is not None:
                    grad_sum = grad.detach() if grad_sum is None else grad_sum + grad.detach()
                if isinstance(rel_ch, torch.Tensor):
                    rel_sum_ch = rel_ch.detach() if rel_sum_ch is None else rel_sum_ch + rel_ch.detach()
                if isinstance(grad_ch, torch.Tensor):
                    grad_sum_ch = grad_ch.detach() if grad_sum_ch is None else grad_sum_ch + grad_ch.detach()
                mse = metrics.get("mse")
                mae = metrics.get("mae")
                if mse is not None:
                    mse_sum = mse.detach() if mse_sum is None else mse_sum + mse.detach()
                if mae is not None:
                    mae_sum = mae.detach() if mae_sum is None else mae_sum + mae.detach()

            batches += 1
            if max_batches and batches >= max_batches:
                break

    if batches == 0 or loss_sum is None:
        return float("nan"), None, float("nan"), float("nan"), None, None, float("nan"), float("nan")

    avg = (loss_sum / batches).item()
    avg_ch = (loss_sum_ch / batches).detach().cpu().tolist() if isinstance(loss_sum_ch, torch.Tensor) else None
    rel_avg = (rel_sum / batches).item() if rel_sum is not None else float("nan")
    grad_avg = (grad_sum / batches).item() if grad_sum is not None else float("nan")
    mse_avg = (mse_sum / batches).item() if mse_sum is not None else float("nan")
    mae_avg = (mae_sum / batches).item() if mae_sum is not None else float("nan")
    rel_avg_ch = (rel_sum_ch / batches).detach().cpu().tolist() if isinstance(rel_sum_ch, torch.Tensor) else None
    grad_avg_ch = (grad_sum_ch / batches).detach().cpu().tolist() if isinstance(grad_sum_ch, torch.Tensor) else None
    return avg, avg_ch, rel_avg, grad_avg, rel_avg_ch, grad_avg_ch, mse_avg, mae_avg
