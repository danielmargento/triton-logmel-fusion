"""fp32 frontend latency for the reference, the Triton kernel, and the CUDA C kernel.

All three run the same fp32 FFT, so the difference is the post-FFT implementation.
The Triton kernel uses tensor cores through tl.dot, while the CUDA C kernel is a
plain one-thread-per-output matmul, so this shows what the tensor-core path buys.
"""

import torch

from timing import benchmark
from logmel import whisper_config
from logmel.reference import LogMelReference
from logmel.triton_fused import FusedLogMel
from logmel.cuda_fused import CudaFusedLogMel

BATCH_SIZES = [1, 8, 32]
DURATIONS_S = [1, 10, 30]


def main() -> None:
    config = whisper_config()
    reference = LogMelReference(config, dtype=torch.float32)
    triton_fused = FusedLogMel(config, dtype=torch.float32)
    cuda_fused = CudaFusedLogMel(config)

    print(f"{'batch':>6} {'dur_s':>6} {'ref_ms':>9} {'triton_ms':>10} {'cuda_ms':>9} {'triton_sp':>10} {'cuda_sp':>9}")
    for batch in BATCH_SIZES:
        for duration in DURATIONS_S:
            wave = torch.randn(batch, duration * config.sample_rate, device="cuda")
            ref_ms = benchmark(lambda w=wave: reference(w)).p50_ms
            tri_ms = benchmark(lambda w=wave: triton_fused(w)).p50_ms
            cuda_ms = benchmark(lambda w=wave: cuda_fused(w)).p50_ms
            print(f"{batch:>6} {duration:>6} {ref_ms:>9.4f} {tri_ms:>10.4f} {cuda_ms:>9.4f} "
                  f"{ref_ms / tri_ms:>9.2f}x {ref_ms / cuda_ms:>8.2f}x")


if __name__ == "__main__":
    main()
