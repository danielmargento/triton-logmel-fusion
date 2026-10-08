"""Compare the maximally fused kernel against the two-kernel fused path and the reference.

The fully fused kernel folds in the magnitude and the per-utterance max reduction
that the two-kernel path leaves outside, so this shows what that extra fusion buys.
"""

import torch

from timing import benchmark
from logmel import whisper_config
from logmel.reference import LogMelReference
from logmel.triton_fused import FusedLogMel
from logmel.triton_fused_full import FullyFusedLogMel

CELLS = [(1, 1), (8, 10), (32, 30)]


def main() -> None:
    config = whisper_config()
    print(f"GPU {torch.cuda.get_device_name(0)}")
    for dtype_name, dtype in {"fp16": torch.float16, "fp32": torch.float32}.items():
        reference = LogMelReference(config, dtype=dtype)
        fused = FusedLogMel(config, dtype=dtype)
        full = FullyFusedLogMel(config, dtype=dtype)
        for batch, duration in CELLS:
            wave = torch.randn(batch, duration * config.sample_rate, device="cuda", dtype=dtype)
            for fn in (reference, fused, full):
                fn(wave)
            torch.cuda.synchronize()
            ref_ms = benchmark(lambda w=wave: reference(w)).p50_ms
            fused_ms = benchmark(lambda w=wave: fused(w)).p50_ms
            full_ms = benchmark(lambda w=wave: full(w)).p50_ms
            print(f"  {dtype_name} b{batch:<2} {duration:>2}s   ref {ref_ms:7.4f}   fused {fused_ms:7.4f} ({ref_ms/fused_ms:.2f}x)   "
                  f"full {full_ms:7.4f} ({ref_ms/full_ms:.2f}x)")


if __name__ == "__main__":
    main()
