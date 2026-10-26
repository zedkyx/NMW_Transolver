"""
Backward-compatible re-exports for legacy imports.

Existing training scripts import helpers from `scripts.transolver.keti_utils`.
We keep this file as a thin compatibility layer while moving reusable logic into
smaller modules under `scripts/transolver/train/` and `scripts/transolver/viz/`.
    """

from scripts.transolver.train.metrics import relative_l2
from scripts.transolver.viz.loss_curves import (
    plot_loss_curves,
    plot_loss_curves_channels,
    save_loss_curves_npy,
)
from scripts.transolver.viz.samples import plot_samples

__all__ = [
    "relative_l2",
    "plot_loss_curves",
    "plot_loss_curves_channels",
    "save_loss_curves_npy",
    "plot_samples",
]
