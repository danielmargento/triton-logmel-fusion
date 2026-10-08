"""Profiling target that launches one backend's fused mel+log kernel repeatedly.

Used under Nsight Systems and Nsight Compute. The power spectrum is built once
before the measured region so the profile captures only the fused kernel, not the
FFT. Usage: python bench/_profile_target.py [reference|triton|cuda]
"""

import sys

import torch

from logmel import whisper_config
from logmel.reference import LogMelReference
from logmel.triton_fused import FusedLogMel
from logmel.cuda_fused import CudaFusedLogMel

LAUNCHES = 20


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "triton"
    config = whisper_config()
    wave = torch.randn(8, 10 * config.sample_rate, device="cuda")

    reference = LogMelReference(config, dtype=torch.float32)
    power = reference.power_spectrum(wave).contiguous()

    if backend == "triton":
        model = FusedLogMel(config, dtype=torch.float32)
        step = lambda: model.mel_log(power)
    elif backend == "cuda":
        model = CudaFusedLogMel(config)
        step = lambda: model.mel_log(power)
    else:
        step = lambda: reference.mel_project(power)

    for _ in range(10):  # warmup, outside the region of interest
        step()
    torch.cuda.synchronize()

    for _ in range(LAUNCHES):
        step()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
