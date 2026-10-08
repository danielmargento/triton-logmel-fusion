"""Per-architecture report of the fused frontend.

Prints the GPU and, for representative cells, the fused Triton speedup over the
reference in fp16 and fp32. Run once on each GPU to build the cross-hardware
comparison. Kernels are pre-warmed before timing so autotuning does not pollute
the first measured cell.
"""

import torch

from timing import benchmark
from logmel import whisper_config
from logmel.reference import LogMelReference
from logmel.triton_fused import FusedLogMel

CELLS = [(1, 1), (8, 10), (32, 30)]
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def main() -> None:
    config = whisper_config()
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU {name} | capability {cap[0]}.{cap[1]}")

    for dtype_name, dtype in DTYPES.items():
        reference = LogMelReference(config, dtype=dtype)
        fused = FusedLogMel(config, dtype=dtype)
        for batch, duration in CELLS:
            wave = torch.randn(batch, duration * config.sample_rate, device="cuda", dtype=dtype)
            fused(wave)            # trigger autotuning and compilation
            reference(wave)
            torch.cuda.synchronize()
            ref_ms = benchmark(lambda w=wave: reference(w)).p50_ms
            fused_ms = benchmark(lambda w=wave: fused(w)).p50_ms
            print(f"  {dtype_name} b{batch:<2} {duration:>2}s   "
                  f"ref {ref_ms:7.4f} ms   fused {fused_ms:7.4f} ms   {ref_ms / fused_ms:4.2f}x")


if __name__ == "__main__":
    main()
