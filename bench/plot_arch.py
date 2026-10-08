"""Render the cross-architecture speedup figure from the measured results.

Writes results/arch_speedup.png. Values are the fp16 fused-over-reference speedup
measured on each GPU by bench/arch_report.py, recorded in results/architectures.txt.
This runs on CPU and needs no GPU.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Ordered oldest to newest tensor-core generation.
GPUS = ["A100\n(Ampere)", "A40\n(Ampere)", "RTX 4090\n(Ada)", "L4\n(Ada)", "H100\n(Hopper)", "B200\n(Blackwell)"]
SMALL = [1.09, 1.47, 1.24, 1.34, 0.91, 0.97]   # fp16 batch 1, 1 s
MID = [1.07, 1.13, 1.18, 1.13, 1.06, 0.92]     # fp16 batch 8, 10 s
LARGE = [1.25, 1.21, 1.06, 1.12, 1.13, 1.11]   # fp16 batch 32, 30 s


def main() -> None:
    x = range(len(GPUS))
    width = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([i - width for i in x], SMALL, width, label="streaming (b1, 1s)", color="#2a9d8f")
    ax.bar(list(x), MID, width, label="mixed (b8, 10s)", color="#e9c46a")
    ax.bar([i + width for i in x], LARGE, width, label="batched (b32, 30s)", color="#e76f51")
    ax.axhline(1.0, color="#264653", linewidth=1, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(GPUS)
    ax.set_ylabel("fp16 speedup over reference")
    ax.set_title("Fused log-mel speedup across GPU architectures")
    ax.set_ylim(0, 1.6)
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "arch_speedup.png"), dpi=140)
    print("wrote results/arch_speedup.png")


if __name__ == "__main__":
    main()
