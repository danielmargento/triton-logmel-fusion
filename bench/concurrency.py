"""Tail latency of the frontend under concurrent load.

The frontend is a small share of a Whisper forward pass, so its
value is not mean latency but the tail it adds under concurrency. Streaming
servers run many short requests at once, and a frontend built from many small
kernels contends for the scheduler. This issues overlapping requests across CUDA
streams and reports per-request p50 and p99, where fewer kernels should show up
as a lower tail.

The cell is the streaming case: one short clip per request, fp16, the regime a
streaming server runs.
"""

import torch

from logmel import whisper_config
from logmel.reference import LogMelReference
from logmel.triton_fused import FusedLogMel

CONCURRENCY = [1, 8, 32]
DURATION_S = 1
REQUESTS = 400


def _percentile(sorted_vals, q):
    pos = q * (len(sorted_vals) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_vals) - 1)
    return sorted_vals[low] + (sorted_vals[high] - sorted_vals[low]) * (pos - low)


def tail_under_load(frontend, wave, concurrency):
    """Return per-request p50 and p99 with `concurrency` requests in flight."""
    streams = [torch.cuda.Stream() for _ in range(concurrency)]
    for _ in range(3 * concurrency):  # warm every stream
        frontend(wave)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(REQUESTS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(REQUESTS)]
    for i in range(REQUESTS):
        stream = streams[i % concurrency]
        with torch.cuda.stream(stream):
            starts[i].record(stream)
            frontend(wave)
            ends[i].record(stream)
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return _percentile(times, 0.50), _percentile(times, 0.99)


def main() -> None:
    config = whisper_config()
    wave = torch.randn(1, DURATION_S * config.sample_rate, device="cuda", dtype=torch.float16)
    reference = LogMelReference(config, dtype=torch.float16)
    fused = FusedLogMel(config, dtype=torch.float16)

    print(f"{'concurrency':>11} {'ref_p50':>8} {'ref_p99':>8} {'fused_p50':>10} {'fused_p99':>10} {'p99_gain':>9}")
    for concurrency in CONCURRENCY:
        ref_p50, ref_p99 = tail_under_load(reference, wave, concurrency)
        fused_p50, fused_p99 = tail_under_load(fused, wave, concurrency)
        gain = ref_p99 / fused_p99
        print(f"{concurrency:>11} {ref_p50:>8.4f} {ref_p99:>8.4f} {fused_p50:>10.4f} {fused_p99:>10.4f} {gain:>8.2f}x")


if __name__ == "__main__":
    main()
