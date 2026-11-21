"""Analyse the two Panther runs (1.5B@1 and 14B@5) and produce report figures.

The held-out (verdict) numbers are transcribed from runs/panther-runs-21nov/README.md,
and the per-task multiple-choice accuracies are cross-checked against the raw
`benchmark-results.csv` files with pandas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parents[1] / "runs" / "panther-runs-21nov"
PLOTS = Path(__file__).resolve().parents[1] / "plots"
PLOTS.mkdir(exist_ok=True)

TASKS = ["MMLU", "HellaSwag", "ARC", "WinoGrande", "GSM8K"]

DATA = {
    "1.5b@1": {
        "label": "Qwen2.5-1.5B-Instruct @ 1 GiB",
        "short": "1.5B @ 1 GiB",
        "preset": "IQ2_S",
        "acc": {
            "ours": [53.6, 63.0, 41.8, 61.1, 23.7],
            "preset": [45.6, 57.2, 38.0, 57.8, 1.3],
            "original": [60.1, 67.7, 46.5, 63.9, 41.0],
        },
        "kl": {"ours": 0.26, "preset": 0.51},
        "flips": {"ours": 20.21, "preset": 28.81},
        "same_top": {"ours": 78.2, "preset": 66.9},
        "ppl": {"ours": 9.50, "preset": 12.44, "original": 7.42},
        "relative": 67.1,
        "ci": (62.5, 71.6),
    },
    "14b@5": {
        "label": "Qwen2.5-14B-Instruct @ 5 GiB",
        "short": "14B @ 5 GiB",
        "preset": "IQ2_XXS",
        "acc": {
            "ours": [69.2, 78.1, 54.2, 75.8, 71.0],
            "preset": [65.9, 75.0, 53.7, 69.0, 58.7],
            "original": [80.4, 83.8, 61.5, 74.3, 82.3],
        },
        "kl": {"ours": 0.30, "preset": 0.45},
        "flips": {"ours": 16.58, "preset": 21.92},
        "same_top": {"ours": 78.6, "preset": 72.9},
        "ppl": {"ours": 6.54, "preset": 7.42, "original": 5.03},
        "relative": 74.1,
        "ci": (68.3, 80.0),
    },
}

COLORS = {"original": "#94a3b8", "preset": "#e8710a", "ours": "#1a73e8"}
ROLE_LABEL = {"original": "original", "preset": "preset", "ours": "ours (Panther)"}
METRIC_LABEL = {
    "kl": "KL divergence",
    "flips": "answer flips (%)",
    "same_top": "same top word (%)",
    "ppl": "perplexity",
}


def _fmt(v, d=1):
    s = f"{v:.{d}f}"
    return s[:-2] if s.endswith(".0") else s


def cross_check_csv() -> None:
    """Verify the four multiple-choice tasks against the raw CSVs."""
    print("== cross-check against benchmark-results.csv ==")
    for contest, d in DATA.items():
        df = pd.read_csv(RUNS / contest / "benchmark-results.csv")

        def role(w):
            w = str(w)
            if "winner" in w:
                return "ours"
            if "preset" in w:
                return "preset"
            if "original" in w:
                return "original"
            return None

        df["role"] = df["what"].map(role)
        verdict = df[
            (df["split"] == "poc_test")
            & (df["kind"].isin(["task", "task.json"]))
            & (df["role"].notna())
        ]
        csv_acc = verdict.groupby(["role", "task"])["accuracy_pct"].mean()
        csv_task = {"MMLU": "mmlu", "HellaSwag": "hellaswag", "ARC": "arc_challenge", "WinoGrande": "winogrande"}
        print(f"\n{contest}:")
        for r in ["original", "preset", "ours"]:
            row = {t: round(csv_acc.get((r, csv_task[t]), float("nan")), 1) for t in TASKS[:4]}
            expect = dict(zip(TASKS[:4], d["acc"][r][:4]))
            match = all(abs(row[t] - expect[t]) < 0.11 for t in TASKS[:4] if not np.isnan(row[t]))
            print(f"  {r:8s} csv={row}  expect={expect}  {'OK' if match else 'MISMATCH'}")


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
                    ha="center", va="bottom", fontsize=7, color="#334155")
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
    cross_check_csv()
    summary()
    for key in DATA:
        fig_accuracy(key)
    fig_metric("kl", "kl", ["ours", "preset"], 2)
    fig_metric("flips", "flips", ["ours", "preset"], 1)
    fig_metric("sametop", "same_top", ["ours", "preset"], 1)
    fig_metric("ppl", "ppl", ["original", "preset", "ours"], 2)
    fig_verdict()
    print("\nwrote plots to", PLOTS)
