"""GPU latency measurement built on CUDA events.

Host wall-clock timing cannot see GPU execution directly. Kernel launches are
asynchronous, so Python returns before the work runs and a timer around the call
measures launch overhead rather than compute. CUDA events are timestamps
recorded on the GPU stream itself, so the interval between a start and an end
event is the on-device time.

The harness reports median and tail latency rather than a mean. Under concurrent
load an inference team cares about the tail, and a mean averages it away.
"""

from dataclasses import dataclass

import torch


@dataclass
class TimingResult:
    p50_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    samples: int

    def __str__(self) -> str:
        return (
            f"p50={self.p50_ms:.4f}ms  p99={self.p99_ms:.4f}ms  "
            f"mean={self.mean_ms:.4f}ms  min={self.min_ms:.4f}ms  n={self.samples}"
        )


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Return the q-quantile of an already-sorted list by linear interpolation."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_vals) - 1)
    frac = pos - low
    return sorted_vals[low] * (1.0 - frac) + sorted_vals[high] * frac


def benchmark(fn, warmup: int = 25, iters: int = 100, flush_l2: bool = True) -> TimingResult:
    """Measure the per-call latency distribution of a GPU workload.

    fn is a no-argument callable that issues exactly one unit of GPU work. Warmup
    calls are run and discarded so one-time costs (context setup, library plan
    creation, Triton autotuning, clock ramp) stay out of the measurement. With
    flush_l2 set, a buffer larger than the GPU L2 cache is overwritten before
    each timed call so the workload reads from memory rather than a cache warmed
    by the previous iteration.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires a CUDA device")

    # 256 MB exceeds any current GPU L2, so overwriting it evicts cached inputs.
    flusher = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda") if flush_l2 else None

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        if flusher is not None:
            flusher.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return TimingResult(
        p50_ms=_percentile(times, 0.50),
        p99_ms=_percentile(times, 0.99),
        mean_ms=sum(times) / len(times),
        min_ms=times[0],
        samples=iters,
    )


if __name__ == "__main__":
    # Sanity check the harness against a workload with a known cost profile.
    # A 4096-cube fp16 matmul is compute bound and should show a tight p50 to p99
    # spread once warmed up.
    torch.manual_seed(0)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

    result = benchmark(lambda: torch.mm(a, b))
    flops = 2 * 4096**3
    tflops = flops / (result.p50_ms * 1e-3) / 1e12
    print(result)
    print(f"matmul throughput at p50: {tflops:.1f} TFLOP/s")
