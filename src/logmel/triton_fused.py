"""Triton kernels for the log-mel stages that run after the FFT.

Two kernels live here. _mel_log_kernel fuses the mel filterbank matmul and the
log into one pass, holding the intermediate mel values in registers so they never
reach global memory. _normalize_kernel applies the per-utterance normalization
given a precomputed maximum. The FFT and the magnitude stay on cuFFT, outside
this file.

The matmul accumulates in fp32 and the log runs in fp32 even for fp16 input. The
mel values span a wide range and the 1e-10 log floor underflows in fp16.
"""

import math

import torch
import triton
import triton.language as tl
import torchaudio
import torchaudio.functional as audio_f

from .reference import MelConfig

_LOG10_E = 1.0 / math.log(10.0)


def _autotune_configs():
    # Tile shapes for a tall, skinny batched matmul. The mel and frequency
    # dimensions are small and fixed, so the frame tile width and the pipeline
    # depth are the knobs worth varying. Shared memory bounds the fp32 tiles.
    return [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    ]


@triton.autotune(configs=_autotune_configs(), key=["n_freqs", "n_mels", "frames"])
@triton.jit
def _mel_log_kernel(
    power_ptr,          # (batch, n_freqs, frames)
    fb_ptr,             # (n_freqs, n_mels)
    out_ptr,            # (batch, n_mels, frames), fp32
    n_freqs,
    n_mels,
    frames,
    stride_pb, stride_pf, stride_pt,
    stride_ff, stride_fm,
    stride_ob, stride_om, stride_ot,
    LOG10_E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(0)
    offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)   # mel rows
    offs_n = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)   # frame columns
    offs_k = tl.arange(0, BLOCK_K)                                # frequency bins

    power_base = power_ptr + batch_id * stride_pb
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, n_freqs, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < n_freqs
        fb_tile = tl.load(
            fb_ptr + offs_m[:, None] * stride_fm + k[None, :] * stride_ff,
            mask=(offs_m[:, None] < n_mels) & k_mask[None, :],
            other=0.0,
        )
        power_tile = tl.load(
            power_base + k[:, None] * stride_pf + offs_n[None, :] * stride_pt,
            mask=k_mask[:, None] & (offs_n[None, :] < frames),
            other=0.0,
        )
        # ieee keeps the accumulation true fp32, matching the reference matmul
        # rather than the faster, lower-precision TF32 default.
        acc += tl.dot(fb_tile, power_tile, input_precision="ieee")

    log_spec = tl.log(tl.maximum(acc, 1e-10)) * LOG10_E

    out = out_ptr + batch_id * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_ot
    tl.store(out, log_spec, mask=(offs_m[:, None] < n_mels) & (offs_n[None, :] < frames))


@triton.jit
def _normalize_kernel(
    log_ptr,       # (batch, n_mels * frames) fp32, contiguous
    peak_ptr,      # (batch,) fp32, per-utterance maximum
    out_ptr,       # (batch, n_mels * frames), output dtype
    per_batch,
    BLOCK: tl.constexpr,
):
    batch_id = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < per_batch
    base = batch_id * per_batch + offs

    x = tl.load(log_ptr + base, mask=mask, other=0.0)
    peak = tl.load(peak_ptr + batch_id)
    normalized = (tl.maximum(x, peak - 8.0) + 4.0) / 4.0
    tl.store(out_ptr + base, normalized, mask=mask)


class FusedLogMel:
    """Log-mel frontend whose post-FFT stages run in two Triton kernels.

    Lines up with LogMelReference stage for stage, so the two compare directly.
    cuFFT produces the power spectrum in fp32, the fused kernel does the mel
    projection and log, and a second kernel normalizes.
    """

    def __init__(self, config: MelConfig = MelConfig(), dtype: torch.dtype = torch.float32,
                 device: str = "cuda"):
        self.config = config
        self.dtype = dtype

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
        ).to(device, dtype)

    def mel_log(self, power: torch.Tensor) -> torch.Tensor:
        """Fused mel projection and log, returning fp32 log-mel values."""
        power = power.to(self.dtype)
        batch, n_freqs, frames = power.shape
        n_mels = self.config.n_mels
        out = torch.empty((batch, n_mels, frames), device=power.device, dtype=torch.float32)

        grid = lambda meta: (batch, triton.cdiv(n_mels, meta["BLOCK_M"]), triton.cdiv(frames, meta["BLOCK_N"]))
        _mel_log_kernel[grid](
            power, self.mel_fb, out,
            n_freqs, n_mels, frames,
            power.stride(0), power.stride(1), power.stride(2),
            self.mel_fb.stride(0), self.mel_fb.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            LOG10_E=_LOG10_E,
        )
        return out

    def normalize(self, log_spec: torch.Tensor) -> torch.Tensor:
        """Per-utterance normalization in one kernel, given the fp32 log-mel."""
        batch, n_mels, frames = log_spec.shape
        peak = log_spec.amax(dim=(-2, -1))
        out = torch.empty_like(log_spec, dtype=self.dtype)
        per_batch = n_mels * frames
        grid = (batch, triton.cdiv(per_batch, 1024))
        _normalize_kernel[grid](log_spec, peak, out, per_batch, BLOCK=1024)
        return out

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        power = self.spectrogram(wave.to(torch.float32))
        log_spec = self.mel_log(power)
        return self.normalize(log_spec)


if __name__ == "__main__":
    # Check the fused output against the reference across the grid.
    from .reference import LogMelReference, whisper_config

    cfg = whisper_config()
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float16):
        ref = LogMelReference(cfg, dtype=dtype)
        fused = FusedLogMel(cfg, dtype=dtype)
        for batch, duration in [(1, 1), (8, 10), (32, 30)]:
            wave = torch.randn(batch, duration * cfg.sample_rate, device="cuda", dtype=dtype)
            a = fused(wave).float()
            b = ref(wave).float()
            rel = ((a - b).abs() / b.abs().clamp_min(1e-6)).max().item()
            name = "fp32" if dtype == torch.float32 else "fp16"
            status = "pass" if torch.allclose(a, b, rtol=1e-2, atol=1e-3) else "fail"
            print(f"[{status}] {name} batch={batch:>2} dur={duration:>2}s  max_rel_err={rel:.2e}")
