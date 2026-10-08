"""Baseline latency of the reference log-mel frontend across the grid.

This is the number the fused kernel must beat. It times the decomposed reference
pipeline, which runs the ordinary PyTorch path with the mixed-precision
boundaries the hardware forces, and reports p50 and p99 for every cell of the
benchmark grid.
"""

import torch

from timing import benchmark
from logmel import LogMelReference, whisper_config

BATCH_SIZES = [1, 8, 32]
DURATIONS_S = [1, 10, 30]
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def main() -> None:
    config = whisper_config()
    print(f"{'dtype':>5} {'batch':>6} {'dur_s':>6} {'p50_ms':>10} {'p99_ms':>10}")
    for name, dtype in DTYPES.items():
        frontend = LogMelReference(config, dtype=dtype)
        for batch in BATCH_SIZES:
            for duration in DURATIONS_S:
                wave = torch.randn(batch, duration * config.sample_rate, device="cuda", dtype=dtype)
                result = benchmark(lambda w=wave: frontend(w))
                print(f"{name:>5} {batch:>6} {duration:>6} {result.p50_ms:>10.4f} {result.p99_ms:>10.4f}")


if __name__ == "__main__":
    main()
