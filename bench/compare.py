"""Fused frontend latency against the reference, across the grid.

Both paths run the same fp32 FFT, so the ratio isolates what fusing the post-FFT
stages buys. The win tracks the post-FFT share: large where the
pipeline is launch bound, small where it is bandwidth bound.
"""

import torch

from timing import benchmark
from logmel import whisper_config
from logmel.reference import LogMelReference
from logmel.triton_fused import FusedLogMel

BATCH_SIZES = [1, 8, 32]
DURATIONS_S = [1, 10, 30]
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def main() -> None:
    config = whisper_config()
    print(f"{'dtype':>5} {'batch':>6} {'dur_s':>6} {'ref_ms':>9} {'fused_ms':>9} {'speedup':>8}")
    for name, dtype in DTYPES.items():
        reference = LogMelReference(config, dtype=dtype)
        fused = FusedLogMel(config, dtype=dtype)
        for batch in BATCH_SIZES:
            for duration in DURATIONS_S:
                wave = torch.randn(batch, duration * config.sample_rate, device="cuda", dtype=dtype)
                ref_ms = benchmark(lambda w=wave: reference(w)).p50_ms
                fused_ms = benchmark(lambda w=wave: fused(w)).p50_ms
                print(f"{name:>5} {batch:>6} {duration:>6} {ref_ms:>9.4f} {fused_ms:>9.4f} {ref_ms / fused_ms:>7.2f}x")


if __name__ == "__main__":
    main()
