"""The "how small is it" figure: a ghost square (the 16-bit original), a filled square
(the quantised winner), and dotted squares (how many more copies fit in the original)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


PLOTS = Path(__file__).resolve().parents[1] / "plots"
GIB = 2 ** 30


# Tensor bytes of the 16-bit original and of the winner, read from the GGUF files on the build server
SIZES = {"1.5b": (3_087_718_400, 581_259_776), "7b": (15_231_899_648, 4_827_504_640),
         "14b": (29_541_748_736, 4_531_470_336)}


def fig_squares(key: str) -> None:
    b16, bq = SIZES[key]
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
    for key in SIZES:
        fig_squares(key)
    print("wrote square plots")
