"""Analyse the Panther contests and produce the report figures.

The held-out (verdict) numbers are copied from each contest's verdict.json on the build server
(data/contests/<model>@<budget>/verdict.json, 22 Nov 2026). Answer flips are score - 10 x KL, the
fitness definition, since verdict.json records the score and the KL.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS = Path(__file__).resolve().parents[1] / "plots"
PLOTS.mkdir(exist_ok=True)

TASKS = ["MMLU", "HellaSwag", "ARC", "WinoGrande", "GSM8K"]

DATA = {
    "1.5b@1": {
        "label": "Qwen2.5-1.5B-Instruct @ 1 GiB",
        "short": "1.5B @ 1 GiB",
        "preset": "IQ2_S",
        "acc": {
            "ours": [53.8, 63.3, 43.8, 62.5, 33.4],
            "preset": [45.9, 56.1, 36.8, 59.0, 4.4],
            "original": [62.2, 67.5, 46.5, 63.9, 35.6],
        },
        "kl": {"ours": 0.194, "preset": 0.494},
        "flips": {"ours": 21.36, "preset": 27.49},
        "same_top": {"ours": 79.4, "preset": 67.7},
        "relative": 71.9,
        "ci": (67.5, 76.2),
    },
    "7b@5": {
        "label": "Qwen2.5-7B-Instruct @ 5 GiB",
        "short": "7B @ 5 GiB",
        "preset": "Q4_K_M",
        "acc": {
            "ours": [75.4, 79.3, 52.7, 71.0, 75.8],
            "preset": [74.7, 78.8, 54.1, 70.5, 77.4],
            "original": [75.5, 79.3, 53.2, 70.8, 77.2],
        },
        "kl": {"ours": 0.0129, "preset": 0.0154},
        "flips": {"ours": 6.34, "preset": 7.27},
        "same_top": {"ours": 95.1, "preset": 94.4},
        "relative": 87.1,
        "ci": (75.5, 98.6),
    },
    "14b@5": {
        "label": "Qwen2.5-14B-Instruct @ 5 GiB",
        "short": "14B @ 5 GiB",
        "preset": "IQ2_XXS",
        "acc": {
            "ours": [73.6, 77.9, 53.7, 75.5, 77.2],
            "preset": [69.8, 74.8, 50.9, 68.8, 61.8],
            "original": [80.5, 83.4, 61.5, 74.4, 82.8],
        },
        "kl": {"ours": 0.237, "preset": 0.395},
        "flips": {"ours": 15.11, "preset": 22.30},
        "same_top": {"ours": 79.7, "preset": 73.7},
        "relative": 66.6,
        "ci": (61.7, 71.5),
    },
}

COLORS = {"original": "#94a3b8", "preset": "#e8710a", "ours": "#1a73e8"}
ROLE_LABEL = {"original": "original", "preset": "preset", "ours": "ours (Panther)"}
METRIC_LABEL = {
    "kl": "KL divergence",
    "flips": "answer flips (%)",
    "same_top": "same top word (%)",
}


def _fmt(v, d=1):
    s = f"{v:.{d}f}"
    return s[:-2] if s.endswith(".0") else s


def summary() -> None:
    print("\n== derived summary ==")
    for contest, d in DATA.items():
        m = {r: float(np.mean(d["acc"][r])) for r in ("original", "preset", "ours")}
        gains = [o - p for o, p in zip(d["acc"]["ours"], d["acc"]["preset"])]
        print(f"{d['label']}: mean acc ours {m['ours']:.2f} vs preset {m['preset']:.2f} (+{m['ours']-m['preset']:.2f}), original {m['original']:.2f}")
        print("   gains (pts): " + ", ".join(f"{t}: {v:+.1f}" for t, v in zip(TASKS, gains)))
        print(f"   relative {d['relative']} [{d['ci'][0]}, {d['ci'][1]}]  (WIN)")


# --------------------------------------------------------------------------- #
# Accuracy: one figure per model (no title; the report caption names the model) #
# --------------------------------------------------------------------------- #
def fig_accuracy(key: str) -> None:
    d = DATA[key]
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    x = np.arange(len(TASKS))
    width = 0.26
    top = max(d["acc"]["original"])
    for i, role in enumerate(("original", "preset", "ours")):
        bars = ax.bar(x + (i - 1) * width, d["acc"][role], width, label=ROLE_LABEL[role], color=COLORS[role])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.02, _fmt(b.get_height()),
                    ha="center", va="bottom", fontsize=6, rotation=90, color="#334155")
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=9, rotation=12, ha="right")
    ax.set_ylabel("accuracy (%)", fontsize=10)
    ax.set_ylim(0, top * 1.3)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(PLOTS / f"accuracy_{key}.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# One figure per quality metric (no title)                                       #
# --------------------------------------------------------------------------- #
def fig_metric(fname: str, key: str, series: list[str], fmt_d: int) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    contests = list(DATA.values())
    names = [d["short"] for d in contests]
    x = np.arange(len(contests))
    width = 0.28
    n = len(series)
    top = max(d[key][role] for d in contests for role in series)
    for j, role in enumerate(series):
        vals = [d[key][role] for d in contests]
        bars = ax.bar(x + (j - (n - 1) / 2) * width, vals, width, label=ROLE_LABEL[role], color=COLORS[role])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.02,
                    _fmt(b.get_height(), fmt_d), ha="center", va="bottom", fontsize=8, color="#334155")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(METRIC_LABEL[key], fontsize=9)
    ax.set_ylim(0, top * 1.3)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=8, frameon=False, ncol=n, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(PLOTS / f"{fname}.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Verdict                                                                       #
# --------------------------------------------------------------------------- #
def fig_verdict() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    contests = list(DATA.values())
    names = [d["short"] for d in contests]
    rel = [d["relative"] for d in contests]
    lo = [d["ci"][0] for d in contests]
    hi = [d["ci"][1] for d in contests]
    yerr = np.array([[r - l for r, l in zip(rel, lo)], [h - r for h, r in zip(hi, rel)]])
    y = np.arange(len(contests))[::-1]
    ax.errorbar(rel, y, xerr=yerr, fmt="o", color="#1a73e8", capsize=5, markersize=6)
    ax.axvline(100, color="#e8710a", linestyle="--", linewidth=1)
    ax.text(100.5, -0.02, "preset (100)", color="#e8710a", fontsize=7, va="bottom")
    for r, yy, name in zip(rel, y, names):
        ax.text(r + 0.8, yy + 0.16, f"{r:.0f}  WIN", fontsize=8, color="#1a73e8", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("relative score (ours ÷ preset × 100)", fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(55, 105)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(PLOTS / "verdict.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    summary()
    for key in DATA:
        fig_accuracy(key)
    fig_metric("kl", "kl", ["ours", "preset"], 3)
    fig_metric("flips", "flips", ["ours", "preset"], 1)
    fig_metric("sametop", "same_top", ["ours", "preset"], 1)
    fig_verdict()
    print("\nwrote plots to", PLOTS)
