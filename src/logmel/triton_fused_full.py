"""A single Triton kernel for every log-mel stage after the FFT.

Where triton_fused.py starts from the power spectrum and runs a separate kernel
for the per-utterance maximum, this kernel takes the raw complex FFT from
torch.stft and, in one pass, forms the magnitude, projects onto the mel scale,
takes the log, and accumulates the per-utterance maximum with a GPU atomic. Only
the FFT stays outside, on cuFFT. A small second kernel then normalizes using that
maximum.

The payoff is that the power spectrum is never materialized and the maximum needs
no standalone reduction.
"""

import math

import torch
import triton
import triton.language as tl

from .reference import MelConfig
from .triton_fused import _normalize_kernel

_LOG10_E = 1.0 / math.log(10.0)
_NEG_INF = float("-inf")


@triton.jit
def _mag_mel_log_kernel(
    spec_ptr,           # complex FFT viewed as real: (batch, n_freqs, frames, 2)
    fb_ptr,             # (n_freqs, n_mels)
    out_ptr,            # (batch, n_mels, frames), fp32
    peak_ptr,           # (batch,) fp32, per-utterance maximum, updated atomically
    n_freqs,
    n_mels,
    frames,
    stride_sb, stride_sf, stride_st,
    stride_ff, stride_fm,
    stride_ob, stride_om, stride_ot,
    LOG10_E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(0)
    offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    spec_base = spec_ptr + batch_id * stride_sb
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, n_freqs, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < n_freqs
        fb_tile = tl.load(
            fb_ptr + offs_m[:, None] * stride_fm + k[None, :] * stride_ff,
            mask=(offs_m[:, None] < n_mels) & k_mask[None, :],
            other=0.0,
        )
        # Real and imaginary parts sit adjacent in memory (the size-2 last dim),
        # so the imaginary element is one step past the real one.
        base = spec_base + k[:, None] * stride_sf + offs_n[None, :] * stride_st
        kn_mask = k_mask[:, None] & (offs_n[None, :] < frames)
        real = tl.load(base, mask=kn_mask, other=0.0)
        imag = tl.load(base + 1, mask=kn_mask, other=0.0)
        power = real * real + imag * imag
        acc += tl.dot(fb_tile, power.to(fb_tile.dtype), input_precision="ieee")

    log_spec = tl.log(tl.maximum(acc, 1e-10)) * LOG10_E
    store_mask = (offs_m[:, None] < n_mels) & (offs_n[None, :] < frames)
    out = out_ptr + batch_id * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_ot
    tl.store(out, log_spec, mask=store_mask)

    # Fold this tile into the per-utterance maximum. Masked lanes use a sentinel
    # below any real value so they never win the atomic.
    tile_max = tl.max(tl.where(store_mask, log_spec, -1e30))
    tl.atomic_max(peak_ptr + batch_id, tile_max)


class FullyFusedLogMel:
    """Log-mel frontend where every post-FFT stage runs inside Triton."""

    def __init__(self, config: MelConfig = MelConfig(), dtype: torch.dtype = torch.float32,
                 device: str = "cuda", block_m: int = 64, block_n: int = 64, block_k: int = 64):
        import torchaudio.functional as audio_f

        self.config = config
        self.dtype = dtype
        self.device = device
        self.block_m, self.block_n, self.block_k = block_m, block_n, block_k
        self.window = torch.hann_window(config.n_fft, device=device)
        self.mel_fb = audio_f.melscale_fbanks(
            n_freqs=config.n_freqs,
            f_min=0.0,
            f_max=config.sample_rate / 2,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate,
            norm="slaney",
            mel_scale="slaney",
        ).to(device, dtype)

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            wave.to(torch.float32),
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec).contiguous()     # (batch, n_freqs, frames, 2)
        batch, n_freqs, frames, _ = spec.shape
        n_mels = self.config.n_mels

        log_spec = torch.empty((batch, n_mels, frames), device=spec.device, dtype=torch.float32)
        peak = torch.full((batch,), _NEG_INF, device=spec.device, dtype=torch.float32)

        grid = (batch, triton.cdiv(n_mels, self.block_m), triton.cdiv(frames, self.block_n))
        _mag_mel_log_kernel[grid](
            spec, self.mel_fb, log_spec, peak,
            n_freqs, n_mels, frames,
            spec.stride(0), spec.stride(1), spec.stride(2),
            self.mel_fb.stride(0), self.mel_fb.stride(1),
            log_spec.stride(0), log_spec.stride(1), log_spec.stride(2),
            LOG10_E=_LOG10_E,
            BLOCK_M=self.block_m, BLOCK_N=self.block_n, BLOCK_K=self.block_k,
            num_warps=4, num_stages=2,
        )

        out = torch.empty_like(log_spec, dtype=self.dtype)
        per_batch = n_mels * frames
        norm_grid = (batch, triton.cdiv(per_batch, 1024))
        _normalize_kernel[norm_grid](log_spec, peak, out, per_batch, BLOCK=1024)
        return out


if __name__ == "__main__":
    # Check the fully fused output against the reference across the grid.
    from .reference import LogMelReference, whisper_config

    cfg = whisper_config()
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float16):
        ref = LogMelReference(cfg, dtype=dtype)
        full = FullyFusedLogMel(cfg, dtype=dtype)
        for batch, duration in [(1, 1), (8, 10), (32, 30)]:
            wave = torch.randn(batch, duration * cfg.sample_rate, device="cuda", dtype=dtype)
            a = full(wave).float()
            b = ref(wave).float()
            rel = ((a - b).abs() / b.abs().clamp_min(1e-6)).max().item()
            name = "fp32" if dtype == torch.float32 else "fp16"
            status = "pass" if torch.allclose(a, b, rtol=2e-2, atol=2e-3) else "fail"
            print(f"[{status}] {name} batch={batch:>2} dur={duration:>2}s  max_rel_err={rel:.2e}")
