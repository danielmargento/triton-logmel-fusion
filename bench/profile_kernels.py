"""Kernel launch counts for the reference and fused frontends.

Launch overhead is the primary cost in the overhead-bound regime, so the number
of GPU operations dispatched per call is the concrete measure of what fusion
removes. Counts come from torch.profiler over a single call after warmup, and
cover only the post-FFT work, which is the part the kernel changes.
"""

import torch
from torch.profiler import ProfilerActivity, profile

from logmel import whisper_config
from logmel.reference import LogMelReference
from logmel.triton_fused import FusedLogMel

CELLS = [(1, 1), (32, 30)]


def _device_time(event) -> float:
    value = getattr(event, "self_device_time_total", None)
    if value is None:
        value = getattr(event, "self_cuda_time_total", 0.0)
    return value or 0.0


def count_device_ops(fn) -> int:
    """Count GPU operations dispatched by one call, after warming up."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(event.count for event in prof.key_averages() if _device_time(event) > 0)


def main() -> None:
    config = whisper_config()
    print(f"{'dtype':>5} {'batch':>6} {'dur_s':>6} {'ref_ops':>8} {'fused_ops':>10} {'reduction':>10}")
    for name, dtype in {"fp32": torch.float32, "fp16": torch.float16}.items():
        reference = LogMelReference(config, dtype=dtype)
        fused = FusedLogMel(config, dtype=dtype)
        for batch, duration in CELLS:
            wave = torch.randn(batch, duration * config.sample_rate, device="cuda", dtype=dtype)
            # Count only the post-FFT work, the stages the kernel replaces.
            power = reference.power_spectrum(wave)
            ref_ops = count_device_ops(lambda p=power: reference.log_normalize(reference.mel_project(p)))
            fused_ops = count_device_ops(lambda p=power: fused.normalize(fused.mel_log(p)))
            reduction = 100.0 * (ref_ops - fused_ops) / ref_ops
            print(f"{name:>5} {batch:>6} {duration:>6} {ref_ops:>8} {fused_ops:>10} {reduction:>9.0f}%")


if __name__ == "__main__":
    main()
