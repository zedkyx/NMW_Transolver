"""Science-style matplotlib helpers (.cursorules §7)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PAPER_DPI = 600


def apply_paper_rcparams() -> None:
    plt.rcParams["font.sans-serif"] = ["Arial", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["axes.linewidth"] = 1.2


def style_axes(ax) -> None:
    ax.tick_params(width=1.2, length=4)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")


def style_colorbar(cbar, label: str | None = None) -> None:
    if label is not None:
        cbar.set_label(label, fontweight="bold", fontsize=10)
    elif cbar.ax.get_ylabel():
        cbar.ax.yaxis.label.set_fontweight("bold")
    elif cbar.ax.get_xlabel():
        cbar.ax.xaxis.label.set_fontweight("bold")
    cbar.ax.tick_params(width=1.2, length=3)
    for label_obj in list(cbar.ax.get_xticklabels()) + list(cbar.ax.get_yticklabels()):
        label_obj.set_fontweight("bold")


def style_figure(fig) -> None:
    for ax in fig.axes:
        # colorbar axes also get tick bolding; safe for line plots
        style_axes(ax)


def save_paper_fig(fig, out_path, *, dpi: int = PAPER_DPI, caption: str | None = None) -> Path:
    """Save at paper dpi; optional caption sidecar (.txt next to png)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    style_figure(fig)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if caption:
        out_path.with_suffix(".txt").write_text(caption.strip() + "\n", encoding="utf-8")
    return out_path
