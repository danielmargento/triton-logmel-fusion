# Fused log-mel frontend in Triton

Speech recognition models turn a waveform into a log-mel spectrogram before the
encoder runs. In a standard PyTorch pipeline that frontend is a chain of separate
GPU kernels, and each one reads its input from global memory and writes its
output back. On short clips and small batches the fixed cost of launching those
kernels and the memory traffic between them dominate the runtime, which is the
regime a streaming or high-concurrency transcription server lives in.

This repository fuses the stages after the FFT, the mel filterbank projection,
the log, and the per-utterance normalization, into a single Triton kernel, so the
intermediate results stay in registers instead of making round trips through
memory. The FFT stays on cuFFT. Around the kernel there is a hand-written CUDA C
version for comparison, a decomposed PyTorch pipeline that serves as both the
correctness oracle and the baseline, and a CUDA-event measurement harness. The
kernel is benchmarked across six GPUs, profiled with Nsight, and run inside a
Whisper serving path through vLLM.

## Headline results

Measured on an RTX 4090 unless noted. Speedup is the reference latency divided by
the fused latency.

| Result | Number |
| --- | --- |
| fp16 frontend speedup, streaming (batch 1, 1 s) | 1.24x, or 1.37x with the fully fused kernel |
| fp16 frontend speedup, batched (batch 32, 30 s) | 1.06x, or 1.22x fully fused |
| post-FFT GPU operations removed | 55 to 67 percent, from 9 to 11 down to 3 to 5 |
| p99 tail latency under concurrency | 1.11 to 1.20x lower |
| hand-written CUDA kernel vs the tensor-core path | 5 to 6x slower (Nsight) |
| frontend share of a Whisper encoder pass | 0.3 to 4 percent |
| numerical agreement with the reference | fp32 to 1e-7, fp16 to 1.6e-3 |

In summary: the fusion helps in the launch-bound streaming regime and for tail
latency under load, ties the baseline in fp32 where cuBLAS is near optimal, and is
a small fraction of end-to-end Whisper latency. The gain is in the frontend and
the tail, not in end-to-end throughput.

## What runs after the FFT, and what gets fused

The frontend is framing, a windowed FFT per frame, the magnitude, a projection
onto the perceptually spaced mel scale, a log, and a per-utterance normalization.
The FFT is compute heavy and well served by cuFFT, so it stays there. The stages
after it are arithmetic dominated by memory traffic and kernel launches, which is
what the fused kernel targets.

Two precision facts shape the kernel. cuFFT rejects a half-precision FFT for a
window whose size is not a power of two, and the window here is 400, so the FFT
runs in fp32. The log floor of 1e-10 underflows in fp16, so the log runs in fp32.
The kernel loads fp16, accumulates the matmul in fp32, and takes the log in fp32,
which is both correct and the only workable fp16 path.

## Experiments

### Speedup across GPU architectures

The fused kernel measured against the reference on six GPUs across four
tensor-core generations. fp16 speedups, three regimes per GPU.

| GPU | generation | streaming (b1, 1 s) | mixed (b8, 10 s) | batched (b32, 30 s) |
| --- | --- | --- | --- | --- |
| A100 80GB | Ampere | 1.09x | 1.07x | 1.25x |
| A40 | Ampere | 1.47x | 1.13x | 1.21x |
| RTX 4090 | Ada | 1.24x | 1.18x | 1.06x |
| L4 | Ada | 1.34x | 1.13x | 1.12x |
| H100 | Hopper | 0.91x | 1.06x | 1.13x |
| B200 | Blackwell | 0.97x | 0.92x | 1.11x |

The win is largest on bandwidth-limited cards such as the A40 and L4 and smallest
on the newest flagships, where the H100 and B200 fall below parity on short clips
because their cuBLAS and tensor cores are close to optimal. A custom fusion helps
most where the vendor baseline is weakest, so a single-GPU speedup means little
without naming the GPU. Full data in `results/architectures.txt`.

### Fewer kernel launches

The post-FFT work measured with torch.profiler, counting GPU operations per call.

| precision | cell | reference | fused | reduction |
| --- | --- | --- | --- | --- |
| fp32 | b1, 1 s | 9 | 3 | 67 percent |
| fp32 | b32, 30 s | 9 | 4 | 56 percent |
| fp16 | b1, 1 s | 11 | 4 | 64 percent |
| fp16 | b32, 30 s | 11 | 5 | 55 percent |

### Where the FFT dominates

Splitting frontend time into the fp32 FFT and the post-FFT stages shows the
ceiling on what fusion can touch. The post-FFT share is large on small inputs and
small on large batched inputs, which is why the speedups track it.

| precision | cell | post-FFT share of frontend time |
| --- | --- | --- |
| fp32 | b1, 1 s | 69 percent |
| fp32 | b8, 10 s | 38 percent |
| fp32 | b32, 30 s | 15 percent |
| fp16 | b1, 1 s | 82 percent |
| fp16 | b32, 30 s | 17 percent |

### Tail latency under concurrency

Per-request p99 with requests in flight across CUDA streams, fp16, 1 s clips. The
tail is the number a streaming server cares about.

| concurrent requests | reference p99 (ms) | fused p99 (ms) | improvement |
| --- | --- | --- | --- |
| 1 | 0.79 | 0.66 | 1.20x |
| 8 | 0.88 | 0.79 | 1.12x |
| 32 | 0.83 | 0.75 | 1.11x |

### Triton against a hand-written CUDA kernel

The same fusion written in plain CUDA C as a one-thread-per-output kernel, timed
per launch with Nsight Systems on an 8 by 10 s workload.

| kernel | per-launch time | note |
| --- | --- | --- |
| cuBLAS matmul (reference) | 20 us | plus separate log and clamp kernels |
| Triton fused | 16 to 24 us | matmul and log in one kernel, uses tensor cores |
| CUDA C, one thread per output | 114 us | no tensor cores |

The naive kernel runs 5 to 6x slower than both cuBLAS and the Triton kernel, the
gap coming from the lack of tensor cores. It still wins on tiny inputs, where one
fused launch beats the baseline's many and the matmul is too small for tensor
cores to matter. Nsight Compute, the hardware-counter profiler, is blocked on the
shared cloud GPUs by ERR_NVGPUCTRPERM, so counter-level metrics are not collected.
See `results/nsight.txt`.

### Everything after the FFT in one kernel

A second Triton kernel folds in the two stages the two-kernel path leaves outside.
It forms the magnitude from the complex FFT and accumulates the per-utterance
maximum with a GPU atomic, so only the FFT runs outside it. On the RTX 4090 this
raises the fp16 win, with the fp32 path regressing because its IEEE matmul is slow
and the extra FFT handling is overhead there.

| precision | cell | two-kernel fused | fully fused |
| --- | --- | --- | --- |
| fp16 | b1, 1 s | 1.08x | 1.37x |
| fp16 | b32, 30 s | 1.06x | 1.22x |

### End-to-end share of a Whisper pass

Timing the frontend against the Whisper encoder on a 30 s window shows how much a
frontend speedup can move.

| model | precision | batch | frontend share of the forward pass |
| --- | --- | --- | --- |
| base | fp32 | 1 | 1.8 percent |
| base | fp16 | 8 | 3.5 percent |
| small | fp32 | 8 | 0.3 percent |
| small | fp16 | 1 | 1.6 percent |

The frontend is a small part of the pipeline, so even the best frontend speedup
moves end-to-end latency by under one percent. That gap between the microbenchmark
and the full pass is the point of measuring both.

### Serving through vLLM

Whisper small served end to end through vLLM on an A100 reaches about 476
audio-seconds of transcription per wall-second over batched requests, with vLLM
handling batching and paged attention. The frontend is the small part measured
above. See `results/vllm.txt`.

## Repository layout

```
src/logmel/reference.py          Decomposed PyTorch pipeline. Correctness oracle and baseline.
src/logmel/triton_fused.py       Triton kernels: fused mel and log, plus normalization.
src/logmel/triton_fused_full.py  One Triton kernel for every post-FFT stage, magnitude to max.
src/logmel/cuda_fused.py         The same fusion in hand-written CUDA C.
bench/timing.py                  CUDA-event harness: warmup, p50 and p99, L2 flush.
bench/baseline.py                Baseline latency across the grid.
bench/breakdown.py               FFT against post-FFT split.
bench/compare.py                 Fused kernel against the reference.
bench/compare_backends.py        Reference against Triton against CUDA C.
bench/compare_full.py            Two-kernel fused against fully fused.
bench/profile_kernels.py         GPU operation counts.
bench/concurrency.py             Per-request tail latency under load.
bench/whisper_e2e.py             Frontend share of a Whisper pass.
bench/arch_report.py             Per-GPU speedup, for the architecture study.
bench/vllm_whisper.py            Whisper served through vLLM.
bench/run_all.py                 Runs the full suite.
tests/test_correctness.py        Fused kernels against the oracle across the grid.
results/                         Captured results, Nsight profiles, and the environment.
```

## Reproducing

Hardware and versions are pinned in `results/environment.txt`: RTX 4090, CUDA
12.8, PyTorch 2.8.0, Triton 3.4.0, torchaudio 2.8.0. On a CUDA machine with those
installed:

```
pip install -e .
PYTHONPATH=src python bench/run_all.py           # full benchmark suite
PYTHONPATH=src python -m pytest tests/            # correctness
PYTHONPATH=src python bench/compare_backends.py   # reference, Triton, CUDA C
```

The CUDA C backend compiles at import through PyTorch's extension loader, so it
needs a CUDA toolkit and ninja. A benchmark number is meaningless without the
hardware it ran on, so the GPU and versions are recorded next to every result.

## Correctness and limitations

The fused kernels match the reference to a relative error of 1e-7 in fp32 and
1.6e-3 in fp16, checked across the grid by `tests/test_correctness.py`. The fp16
tolerance reflects the half-precision matmul. The fp32 agreement is exact because
the matmul runs in IEEE precision.

Three limitations. In fp32 the fused kernel ties the baseline rather than beating
it, because cuBLAS is near optimal and fp32 is not where the fusion helps. The
frontend is a small fraction of a full Whisper pass, so the end-to-end effect is
under one percent. And the win depends on the GPU, ranging from 1.47x on an A40 to
below parity on an H100, so the fusion is most worthwhile on bandwidth-limited
hardware and for tail latency under concurrency.
