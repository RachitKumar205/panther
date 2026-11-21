"""Model-structure visualisations: space saved and a 3D layer map coloured by bit-width.

The per-matrix type assignment is reconstructed from the recorded type counts in
runs/.../README.md (the full per-matrix genome was not copied to this machine), so the
colours are a schematic of the winner's mix; the architecture sizes and bit-widths are exact.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

PLOTS = Path(__file__).resolve().parents[1] / "plots"
PLOTS.mkdir(exist_ok=True)

# bits per weight (gguf GGML_QUANT_SIZES)
BPW = {
    "IQ1_S": 1.5625, "IQ1_M": 1.75, "IQ2_XXS": 2.0625, "IQ2_XS": 2.3125,
    "IQ2_S": 2.5, "Q2_K": 2.625, "IQ3_XXS": 3.0625, "Q3_K": 3.4375,
    "IQ4_XS": 4.25, "Q4_K": 4.5, "Q5_K": 5.5, "Q6_K": 6.5625, "Q8_0": 8.5,
}
ROLES = ["q", "k", "v", "o", "gate", "up", "down"]


def build_model(name: str) -> list[tuple[str, int, int]]:
    """(matrix name, rows, cols) in run order; the word table and output layer included."""
    if name == "1.5b":
        hidden, layers, inter, vocab, kv = 1536, 28, 8960, 151936, 256
        tied = True
    else:  # 14b
        hidden, layers, inter, vocab, kv = 5120, 48, 13824, 152064, 1024
        tied = False
    mats = [("embd", vocab, hidden)]
    for l in range(layers):
        mats += [
            (f"L{l}.q", hidden, hidden),
            (f"L{l}.k", kv, hidden),
            (f"L{l}.v", kv, hidden),
            (f"L{l}.o", hidden, hidden),
            (f"L{l}.gate", inter, hidden),
            (f"L{l}.up", inter, hidden),
            (f"L{l}.down", hidden, inter),
        ]
    if not tied:
        mats.append(("output", vocab, hidden))
    return mats


WINNER = {
    "1.5b": {"IQ3_XXS": 192, "Q2_K": 4, "IQ2_XS": 1},
    "14b": {"IQ2_XXS": 93, "IQ2_S": 69, "IQ3_XXS": 51, "IQ2_XS": 41, "IQ4_XS": 20, "IQ1_S": 16},
}


def assign_types(mats: list, counts: dict) -> list[str]:
    """Expand the recorded counts to one type per matrix (padding with the dominant
    type), then hand the higher-bit types to the larger matrices. Schematic only."""
    types: list[str] = []
    for t, c in sorted(counts.items(), key=lambda kv: -BPW[kv[0]]):
        types += [t] * c
    dominant = max(counts, key=counts.get)
    while len(types) < len(mats):
        types.append(dominant)
    types = types[: len(mats)]
    order = sorted(range(len(mats)), key=lambda i: -(mats[i][1] * mats[i][2]))
    out = [None] * len(mats)
    for i, t in zip(order, types):
        out[i] = t
    return out


def sizes(mats: list, assign: list[str]) -> tuple[float, float, float]:
    """(original GB, quantised GB, weighted mean bpw)."""
    orig = sum(r * c * 16 / 8 for _, r, c in mats) / 1e9
    quant = sum(r * c * BPW[t] / 8 for (_, r, c), t in zip(mats, assign)) / 1e9
    total = sum(r * c for _, r, c in mats)
    mean_bpw = sum(r * c * BPW[t] for (_, r, c), t in zip(mats, assign)) / total
    return orig, quant, mean_bpw


# --------------------------------------------------------------------------- #
# Figure A: space saved                                                        #
# --------------------------------------------------------------------------- #
def fig_space() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0), sharex=True)
    names = [("1.5b", "Qwen2.5-1.5B\n@ 1 GiB"), ("14b", "Qwen2.5-14B\n@ 5 GiB")]
    for ax, (key, label) in zip(axes, names):
        mats = build_model(key)
        assign = assign_types(mats, WINNER[key])
        orig, quant, bpw = sizes(mats, assign)
        saved = 100 * (orig - quant) / orig
        ax.barh([1], [orig], color="#cbd5e1", height=0.5, label="16-bit original")
        ax.barh([0], [quant], color="#1a73e8", height=0.5, label="quantised winner")
        ax.set_xlim(0, orig * 1.18)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["ours", "original"], fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.text(quant + 0.02 * orig, 0, f"{quant:.1f} GiB · {bpw:.1f} bpw", va="center", fontsize=7, color="#1a73e8")
        ax.text(orig + 0.02 * orig, 1, f"{orig:.1f} GiB", va="center", fontsize=7, color="#334155")
        ax.text(0.98 * orig, -0.42, f"−{saved:.0f}%", va="bottom", ha="right", fontsize=8, color="#0f766e")
        ax.grid(axis="x", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(PLOTS / "space.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure B: 3D layer map, colour = bit-width                                  #
# --------------------------------------------------------------------------- #
def fig_3d() -> None:
    fig = plt.figure(figsize=(3.4, 5.2))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(1.5, 8.5)
    titles = {"1.5b": "Qwen2.5-1.5B @ 1 GiB", "14b": "Qwen2.5-14B @ 5 GiB"}
    xtick_step = {"1.5b": 4, "14b": 8}

    for idx, key in enumerate(("1.5b", "14b")):
        ax = fig.add_subplot(2, 1, idx + 1, projection="3d")
        mats = build_model(key)
        layers = (len(mats) - (2 if key == "14b" else 1)) // 7
        assign = assign_types(mats, WINNER[key])

        for (name, r, c), t in zip(mats, assign):
            if name in ("embd", "output"):
                continue
            layer_str, role = name.split(".")
            x = int(layer_str[1:])
            y = ROLES.index(role)
            size = r * c * BPW[t] / 8 / 1e6  # MB
            ax.bar3d(x, y, 0, 0.85, 0.85, size, color=cmap(norm(BPW[t])), shade=True, alpha=0.95)

        ax.set_xlabel("layer", fontsize=8, labelpad=3)
        ax.set_xticks(range(0, layers, xtick_step[key]))
        ax.tick_params(axis="x", labelsize=6, pad=0)
        ax.set_yticks(range(7))
        ax.set_yticklabels(ROLES, fontsize=6)
        ax.tick_params(axis="y", labelsize=6, pad=0)
        ax.set_zlabel("MB", fontsize=8, labelpad=3)
        ax.tick_params(axis="z", labelsize=6, pad=0)
        ax.view_init(elev=28, azim=-62)
        ax.set_title(titles[key], fontsize=9, pad=4)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=fig.axes, shrink=0.6, aspect=25, pad=0.06)
    cbar.set_label("bits per weight", fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.05, hspace=0.28)
    fig.savefig(PLOTS / "layers3d.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    for key in ("1.5b", "14b"):
        mats = build_model(key)
        assign = assign_types(mats, WINNER[key])
        orig, quant, bpw = sizes(mats, assign)
        print(f"{key}: original {orig:.2f} GiB, quantised {quant:.2f} GiB, "
              f"saved {100*(orig-quant)/orig:.1f}%, mean {bpw:.2f} bpw")
    fig_space()
    print("wrote space.png (3D renders are produced by gen_3d.py + the Three.js renderer)")
