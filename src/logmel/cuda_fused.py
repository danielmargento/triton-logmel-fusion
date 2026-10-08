"""Raw CUDA C version of the fused post-FFT log-mel stages.

This mirrors the Triton kernel in hand-written CUDA C, compiled at import time
through PyTorch's extension loader. It exists for comparison: the Triton kernel
uses tensor cores through tl.dot, while this version is a straightforward
one-thread-per-output kernel, which makes the tensor-core advantage visible in
the Nsight profiles and in the timings.

The CUDA path runs in fp32. The mel projection accumulates in fp32 and the log
runs in fp32, the same numerics the fp32 reference uses.
"""

import functools

import torch
import torchaudio
import torchaudio.functional as audio_f
from torch.utils.cpp_extension import load_inline

from .reference import MelConfig

_CPP = """
torch::Tensor mel_log(torch::Tensor power, torch::Tensor fb);
torch::Tensor normalize(torch::Tensor log_spec, torch::Tensor peak);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// out[b, m, t] = log10(max(sum_f fb[f, m] * power[b, f, t], 1e-10)).
// One thread per output element. The reduction over frequency bins runs in a
// register accumulator, so the intermediate mel value never touches memory.
__global__ void mel_log_kernel(const float* __restrict__ power,
                               const float* __restrict__ fb,
                               float* __restrict__ out,
                               int n_freqs, int n_mels, int frames, long total) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int t = idx % frames;
    int m = (idx / frames) % n_mels;
    long b = idx / ((long)frames * n_mels);

    const float* p = power + (b * n_freqs) * (long)frames + t;
    float acc = 0.0f;
    for (int f = 0; f < n_freqs; ++f) {
        acc += fb[(long)f * n_mels + m] * p[(long)f * frames];
    }
    acc = fmaxf(acc, 1e-10f);
    out[idx] = log10f(acc);
}

// Per-utterance normalization: clamp to peak - 8, shift, and scale.
__global__ void normalize_kernel(const float* __restrict__ log_spec,
                                 const float* __restrict__ peak,
                                 float* __restrict__ out,
                                 long per_batch, long total) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long b = idx / per_batch;
    float v = fmaxf(log_spec[idx], peak[b] - 8.0f);
    out[idx] = (v + 4.0f) / 4.0f;
}

torch::Tensor mel_log(torch::Tensor power, torch::Tensor fb) {
    const int batch = power.size(0);
    const int n_freqs = power.size(1);
    const int frames = power.size(2);
    const int n_mels = fb.size(1);
    auto out = torch::empty({batch, n_mels, frames}, power.options());
    const long total = (long)batch * n_mels * frames;
    const int threads = 256;
    const long blocks = (total + threads - 1) / threads;
    mel_log_kernel<<<blocks, threads>>>(
        power.data_ptr<float>(), fb.data_ptr<float>(), out.data_ptr<float>(),
        n_freqs, n_mels, frames, total);
    return out;
}

torch::Tensor normalize(torch::Tensor log_spec, torch::Tensor peak) {
    const int batch = log_spec.size(0);
    const int n_mels = log_spec.size(1);
    const int frames = log_spec.size(2);
    auto out = torch::empty_like(log_spec);
    const long per_batch = (long)n_mels * frames;
    const long total = (long)batch * per_batch;
    const int threads = 256;
    const long blocks = (total + threads - 1) / threads;
    normalize_kernel<<<blocks, threads>>>(
        log_spec.data_ptr<float>(), peak.data_ptr<float>(), out.data_ptr<float>(),
        per_batch, total);
    return out;
}
"""


@functools.lru_cache(maxsize=1)
def _extension():
    return load_inline(
        name="logmel_cuda",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=["mel_log", "normalize"],
        verbose=False,
    )


class CudaFusedLogMel:
    """Log-mel frontend whose post-FFT stages run in hand-written CUDA C (fp32)."""

    def __init__(self, config: MelConfig = MelConfig(), device: str = "cuda"):
        self.config = config
        self.dtype = torch.float32
        self.ext = _extension()

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=config.n_fft,
            win_length=config.n_fft,
            hop_length=config.hop_length,
            power=2.0,
            center=True,
        ).to(device, torch.float32)

        self.mel_fb = audio_f.melscale_fbanks(
            n_freqs=config.n_freqs,
            f_min=0.0,
            f_max=config.sample_rate / 2,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate,
            norm="slaney",
            mel_scale="slaney",
        ).to(device, torch.float32).contiguous()

    def mel_log(self, power: torch.Tensor) -> torch.Tensor:
        return self.ext.mel_log(power.contiguous(), self.mel_fb)

    def normalize(self, log_spec: torch.Tensor) -> torch.Tensor:
        peak = log_spec.amax(dim=(-2, -1)).contiguous()
        return self.ext.normalize(log_spec.contiguous(), peak)

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        power = self.spectrogram(wave.to(torch.float32)).contiguous()
        return self.normalize(self.mel_log(power))


if __name__ == "__main__":
    from .reference import LogMelReference, whisper_config

    cfg = whisper_config()
    torch.manual_seed(0)
    ref = LogMelReference(cfg, dtype=torch.float32)
    cuda = CudaFusedLogMel(cfg)
    for batch, duration in [(1, 1), (8, 10), (32, 30)]:
        wave = torch.randn(batch, duration * cfg.sample_rate, device="cuda")
        a = cuda(wave)
        b = ref(wave)
        rel = ((a - b).abs() / b.abs().clamp_min(1e-6)).max().item()
        status = "PASS" if torch.allclose(a, b, rtol=1e-3, atol=1e-4) else "FAIL"
        print(f"[{status}] fp32 batch={batch:>2} dur={duration:>2}s  max_rel_err={rel:.2e}")
