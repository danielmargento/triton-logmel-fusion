"""Render the two headline figures from the measured results.

Writes results/speedup_by_config.png and results/tail_latency.png. The values
are the measured numbers from a single benchmark run on the RTX 4090, recorded
in results/benchmarks.txt.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# fp16 fused speedup over reference, per (batch, duration) cell.
CONFIGS = ["1/1s", "1/10s", "1/30s", "8/1s", "8/10s", "8/30s", "32/1s", "32/10s", "32/30s"]
FP16_SPEEDUP = [1.24, 1.15, 0.90, 1.08, 1.15, 1.16, 1.16, 1.09, 1.06]

# Per-request p99 tail latency (ms) under concurrent load, fp16, 1s clips.
CONCURRENCY = ["1", "8", "32"]
REF_P99 = [0.7902, 0.8815, 0.8285]
FUSED_P99 = [0.6574, 0.7854, 0.7469]


def speedup_figure() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2a9d8f" if v >= 1.0 else "#e76f51" for v in FP16_SPEEDUP]
    ax.bar(CONFIGS, FP16_SPEEDUP, color=colors)
    ax.axhline(1.0, color="#264653", linewidth=1, linestyle="--")
    ax.set_ylabel("speedup over reference")
    ax.set_xlabel("batch / audio duration")
    ax.set_title("fp16 fused frontend speedup, RTX 4090")
    ax.set_ylim(0, 1.4)
    for i, v in enumerate(FP16_SPEEDUP):
        ax.text(i, v + 0.02, f"{v:.2f}x", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "speedup_by_config.png"), dpi=140)


def tail_figure() -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(CONCURRENCY))
    width = 0.38
    ax.bar([i - width / 2 for i in x], REF_P99, width, label="reference", color="#e76f51")
    ax.bar([i + width / 2 for i in x], FUSED_P99, width, label="fused", color="#2a9d8f")
    ax.set_xticks(list(x))
    ax.set_xticklabels(CONCURRENCY)
    ax.set_ylabel("per-request p99 latency (ms)")
    ax.set_xlabel("concurrent requests in flight")
    ax.set_title("Tail latency under concurrency, fp16 1s clips")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "tail_latency.png"), dpi=140)


if __name__ == "__main__":
    speedup_figure()
    tail_figure()
    print("wrote results/speedup_by_config.png and results/tail_latency.png")
