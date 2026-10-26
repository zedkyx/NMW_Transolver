"""Learning-rate schedulers and early-stopping helpers for KETI training."""

from __future__ import annotations

import torch


def is_meaningful_improvement(
    best: float,
    new: float,
    min_delta_rel: float = 0.001,
) -> bool:
    """True when new improves best by more than min_delta_rel (relative)."""
    if best == float("inf") or best <= 0.0:
        return True
    return new < best * (1.0 - min_delta_rel)


def build_lr_scheduler(optimizer, args, steps_per_epoch: int | None = None):
    """
    Build LR scheduler from args.

    Supported lr_scheduler: none, cosine, step, plateau, onecycle.
    onecycle uses PyTorch defaults (pct_start=0.3, div_factor=25,
    final_div_factor=1e4, anneal_strategy='cos') and must be stepped
    once per optimizer step. Returns (scheduler, mode) with mode in
    {none, epoch, plateau, batch}.
    """
    name = str(getattr(args, "lr_scheduler", "none") or "none").lower()
    lr_min = float(getattr(args, "lr_min", 1e-6))

    if name in ("none", "off", "false", "0"):
        return None, "none"

    if name in ("onecycle", "onecyclelr", "one_cycle"):
        if steps_per_epoch is None:
            steps_per_epoch = getattr(args, "steps_per_epoch", None)
        steps = int(steps_per_epoch or 0)
        if steps < 1:
            raise ValueError("onecycle requires steps_per_epoch >= 1 (len of train loader)")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(args.lr),
            epochs=int(args.epochs),
            steps_per_epoch=steps,
        )
        return scheduler, "batch"

    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(args.epochs),
            eta_min=lr_min,
        )
        return scheduler, "epoch"

    if name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(getattr(args, "lr_step_size", 500)),
            gamma=float(getattr(args, "lr_gamma", 0.5)),
        )
        return scheduler, "epoch"

    if name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(getattr(args, "lr_plateau_factor", 0.5)),
            patience=int(getattr(args, "lr_plateau_patience", 50)),
            min_lr=lr_min,
            threshold=float(getattr(args, "early_stop_min_delta_rel", 0.001)),
            threshold_mode="rel",
        )
        return scheduler, "plateau"

    raise ValueError(f"Unknown lr_scheduler: {name!r} (use none/cosine/step/plateau/onecycle)")


def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])
