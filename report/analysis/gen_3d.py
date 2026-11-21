"""The "how small is it" figure: a ghost square (the 16-bit original), a filled square
(the quantised winner), and dotted squares (how many more copies fit in the original)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import model_viz as mv

PLOTS = Path(__file__).resolve().parents[1] / "plots"
GIB = 2 ** 30


def fig_squares(key: str) -> None:
    mats = mv.build_model(key)
    assign = mv.assign_types(mats, mv.WINNER[key])
    b16 = sum(r * c * 2 for (_, r, c) in mats)
    bq = sum(r * c * mv.BPW[t] / 8 for (_, r, c), t in zip(mats, assign))
    ratio = bq / b16
    saved = 100 * (1 - ratio)
    s = ratio ** 0.5
    n = int(1 // s)  # whole copies per side that fit in the ghost square

    fig, ax = plt.subplots(figsize=(3.2, 3.9))
    ax.set_aspect("equal")
    ax.axis("off")

    # ghost square: the 16-bit original
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#e8edf3", edgecolor="#94a3b8", linewidth=1.4, zorder=1))

    # dotted squares: the copies that would fit (all but the first)
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            ax.add_patch(Rectangle((i * s, j * s), s, s, fill=False, edgecolor="#64748b",
                                   linestyle=(0, (3, 2)), linewidth=1.0, zorder=2))

    # filled square: the quantised winner
    ax.add_patch(Rectangle((0, 0), s, s, facecolor="#2563eb", edgecolor="#1e3a8a", linewidth=1.0, zorder=3))

    ax.text(0.5, -0.13, f"{b16 / GIB:.1f} GiB  →  {bq / GIB:.2f} GiB   (−{saved:.0f}%)",
            ha="center", va="top", fontsize=10, color="#1e293b")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.10)
    fig.savefig(PLOTS / f"layers3d_{key}.png", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    for key in ("1.5b", "14b"):
        fig_squares(key)
    print("wrote square plots")
