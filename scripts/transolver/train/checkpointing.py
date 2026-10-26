import json
import os
from typing import Any, Optional

import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Return the underlying model when wrapped by DDP/DataParallel.
    """
    return model.module if hasattr(model, "module") else model


def save_state_dict(model: nn.Module, path: str, is_main_process: bool = True) -> None:
    """
    Save model state_dict to path. No-op if not main process.
    """
    if not is_main_process:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), path)


def _normalize_fx_map(fx_map: Any) -> dict[int, Optional[int]]:
    """
    Convert yaml/json fx_map to {new_idx: old_idx|None}.
    None / null / "null" / "random" / "zero" → None (new channel).
    """
    if fx_map is None:
        return {}
    if not isinstance(fx_map, dict):
        raise TypeError(f"transfer.fx_map must be a dict, got {type(fx_map)}")

    out: dict[int, Optional[int]] = {}
    for k, v in fx_map.items():
        new_idx = int(k)
        if v is None:
            out[new_idx] = None
            continue
        if isinstance(v, str):
            token = v.strip().lower()
            if token in ("", "null", "none", "random", "new", "zero"):
                # concrete init handled by new_init; map value None means "no old col"
                out[new_idx] = None
                continue
            out[new_idx] = int(v)
            continue
        out[new_idx] = int(v)
    return out


def _resolve_transfer_spec(transfer: Optional[dict], args=None) -> Optional[dict]:
    """
    Build a normalized transfer spec, or None if transfer is disabled/absent.
    """
    if transfer is None and args is not None:
        transfer = getattr(args, "transfer", None)
    if not transfer:
        return None
    if isinstance(transfer, str):
        transfer = json.loads(transfer)
    if not isinstance(transfer, dict):
        raise TypeError(f"transfer must be a dict, got {type(transfer)}")

    enabled = transfer.get("enabled", 1)
    if enabled in (0, "0", False, "false", "False"):
        return None

    pos_feat_dim = transfer.get("pos_feat_dim", None)
    if pos_feat_dim is None and args is not None:
        unified = int(getattr(args, "unified_pos", 1))
        ref = int(getattr(args, "ref", 8))
        if unified:
            pos_feat_dim = ref * ref
        else:
            pos_feat_dim = 2
    if pos_feat_dim is None:
        raise ValueError("transfer.pos_feat_dim is required when args cannot derive it")

    new_init = str(transfer.get("new_init", "random")).strip().lower()
    if new_init not in ("random", "zero"):
        raise ValueError(f"transfer.new_init must be 'random' or 'zero', got {new_init!r}")

    fx_map = _normalize_fx_map(transfer.get("fx_map"))
    if not fx_map:
        raise ValueError("transfer.enabled but transfer.fx_map is empty")

    return {
        "pos_feat_dim": int(pos_feat_dim),
        "new_init": new_init,
        "fx_map": fx_map,
    }


def _expand_linear_weight_by_fx_map(
    old_w: torch.Tensor,
    new_w_template: torch.Tensor,
    *,
    pos_feat_dim: int,
    fx_map: dict[int, Optional[int]],
    new_init: str,
) -> tuple:
    """
    Remap preprocess linear_pre input columns:

      [pos_feat | fx_old...]  →  [pos_feat | fx_new...]

    fx_map: new_fx_idx -> old_fx_idx | None (new channel).
    Returns (migrated_weight, n_mapped, n_new, old_fx, new_fx).
    """
    if old_w.ndim != 2 or new_w_template.ndim != 2:
        raise ValueError(
            f"expected 2D linear weights, got old={tuple(old_w.shape)} new={tuple(new_w_template.shape)}"
        )
    out_o, in_o = old_w.shape
    out_n, in_n = new_w_template.shape
    if out_o != out_n:
        raise ValueError(
            f"linear out_features mismatch: old={out_o} new={out_n} (cannot remap)"
        )
    if in_o < pos_feat_dim or in_n < pos_feat_dim:
        raise ValueError(
            f"pos_feat_dim={pos_feat_dim} larger than in_features old={in_o} new={in_n}"
        )

    old_fx = in_o - pos_feat_dim
    new_fx = in_n - pos_feat_dim
    expected = set(range(new_fx))
    got = set(fx_map.keys())
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise ValueError(
            f"transfer.fx_map must cover new fx indices 0..{new_fx - 1}; "
            f"missing={missing}, extra={extra}"
        )

    for new_i, old_i in fx_map.items():
        if old_i is None:
            continue
        if old_i < 0 or old_i >= old_fx:
            raise ValueError(
                f"fx_map[{new_i}]={old_i} out of old fx range [0, {old_fx})"
            )

    if new_init == "random":
        migrated = new_w_template.detach().clone()
    else:
        migrated = torch.zeros_like(new_w_template)

    # pos features: require equal width; copy 1:1
    migrated[:, :pos_feat_dim] = old_w[:, :pos_feat_dim]

    n_mapped = 0
    n_new = 0
    for new_i, old_i in sorted(fx_map.items()):
        col = pos_feat_dim + new_i
        if old_i is None:
            n_new += 1
            if new_init == "zero":
                migrated[:, col] = 0
            # random: already from template
            continue
        migrated[:, col] = old_w[:, pos_feat_dim + old_i]
        n_mapped += 1

    return migrated, n_mapped, n_new, old_fx, new_fx


def _is_preprocess_linear_pre_weight(key: str) -> bool:
    return ("preprocess.linear_pre" in key) and key.endswith(".weight")


def load_state_dict_into(
    model: nn.Module,
    state_dict: dict,
    *,
    transfer: Optional[dict] = None,
    args=None,
    is_main_process: bool = True,
) -> dict:
    """
    Load state_dict into model.

    - Matching shapes: copy.
    - preprocess.linear_pre.*.weight with transfer.fx_map: column remap.
    - Any other shape mismatch: raise (no silent skip).
    """
    core = unwrap_model(model)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and not any(
        torch.is_tensor(v) for v in state_dict.values()
    ):
        state_dict = state_dict["state_dict"]

    spec = _resolve_transfer_spec(transfer, args=args)
    model_sd = core.state_dict()
    filtered = {}
    migrated = []
    missing_in_model = []

    for key, value in state_dict.items():
        if key not in model_sd:
            missing_in_model.append(key)
            continue
        if not torch.is_tensor(value):
            raise TypeError(f"checkpoint value for {key} is not a tensor: {type(value)}")

        tgt = model_sd[key]
        if tuple(tgt.shape) == tuple(value.shape):
            filtered[key] = value
            continue

        if _is_preprocess_linear_pre_weight(key):
            if spec is None:
                raise RuntimeError(
                    f"shape mismatch for {key}: {tuple(value.shape)} -> {tuple(tgt.shape)}; "
                    f"set global transfer.fx_map to remap fx input columns"
                )
            expanded, n_mapped, n_new, old_fx, new_fx = _expand_linear_weight_by_fx_map(
                value,
                tgt,
                pos_feat_dim=spec["pos_feat_dim"],
                fx_map=spec["fx_map"],
                new_init=spec["new_init"],
            )
            filtered[key] = expanded
            migrated.append(
                (
                    key,
                    f"fx {old_fx}→{new_fx} via fx_map; mapped={n_mapped}, "
                    f"new={n_new} init={spec['new_init']}, "
                    f"pos_feat_dim={spec['pos_feat_dim']}, "
                    f"shape {tuple(value.shape)}→{tuple(expanded.shape)}",
                )
            )
            continue

        raise RuntimeError(
            f"Unhandled shape mismatch for {key}: "
            f"{tuple(value.shape)} -> {tuple(tgt.shape)} "
            f"(only preprocess.linear_pre.*.weight can be remapped via transfer.fx_map)"
        )

    incompatible = core.load_state_dict(filtered, strict=False)
    # missing_keys = in model but not in filtered/ckpt → left at init; OK for brand-new params
    # unexpected should be empty because we only load filtered keys
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys after filtered load: {incompatible.unexpected_keys[:12]}"
        )

    if is_main_process:
        print(
            f"[*] Checkpoint load: exact={len(filtered) - len(migrated)}/{len(state_dict)}, "
            f"migrated={len(migrated)}, missing_in_ckpt={len(incompatible.missing_keys)}"
        )
        if spec is not None:
            print(
                f"    transfer: pos_feat_dim={spec['pos_feat_dim']}, "
                f"new_init={spec['new_init']}, fx_map_size={len(spec['fx_map'])}"
            )
        for key, reason in migrated:
            print(f"    migrate {key}: {reason}")
        if missing_in_model:
            print(
                f"    [!] ckpt keys not in model (ignored): {missing_in_model[:8]}"
                + (f" ... (+{len(missing_in_model) - 8})" if len(missing_in_model) > 8 else "")
            )
        if incompatible.missing_keys:
            preview = incompatible.missing_keys[:8]
            print(f"    missing_in_ckpt (keep init): {preview}")
            if len(incompatible.missing_keys) > 8:
                print(f"    ... and {len(incompatible.missing_keys) - 8} more")

    return {
        "kept": len(filtered),
        "migrated": migrated,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "transfer": spec,
    }


def maybe_load_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    device,
    is_main_process: bool = True,
    transfer: Optional[dict] = None,
    args=None,
) -> bool:
    """
    Load state_dict from ckpt_path if it exists.
    Returns True if loaded, False otherwise.
    """
    if not ckpt_path:
        return False
    if not os.path.exists(ckpt_path):
        if is_main_process:
            print(f"[WARN] Checkpoint not found: {ckpt_path}")
        return False
    if is_main_process:
        print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    load_state_dict_into(
        model,
        state_dict,
        transfer=transfer,
        args=args,
        is_main_process=is_main_process,
    )
    return True
