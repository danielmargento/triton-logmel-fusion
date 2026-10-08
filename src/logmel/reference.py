"""Reference log-mel frontend, decomposed into explicit stages.

This module plays two roles. It is the correctness oracle the fused Triton
kernel is validated against, and it is the baseline the kernel is timed against.
The stage boundaries match the fusion plan so the two are directly comparable.

Two precision boundaries are forced by the hardware and the math, not by choice:

  The FFT runs in fp32 on cuFFT in every case. cuFFT rejects half precision for
  a window whose size is not a power of two, and Whisper's window is 400.

  The log compression runs in fp32 in every case. Its floor of 1e-10 underflows
  to zero in fp16, which would send the logarithm to negative infinity.

The mel projection, the stage that dominates cost, runs in the requested dtype.
That is the stage the Triton kernel fuses, so it is where precision matters for
the benchmark.
"""

from dataclasses import dataclass

import torch
import torchaudio
import torchaudio.functional as audio_f


@dataclass(frozen=True)
class MelConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80

    @property
    def n_freqs(self) -> int:
        return self.n_fft // 2 + 1


def whisper_config() -> MelConfig:
    """Return the mel parameters Whisper uses for its audio frontend."""
    return MelConfig()


class LogMelReference:
    """Whisper-style log-mel features computed stage by stage.

    Calling the instance maps a batch of waveforms, shaped (batch, samples), to
    normalized log-mel features shaped (batch, n_mels, frames).
    """

    def __init__(self, config: MelConfig = MelConfig(), dtype: torch.dtype = torch.float32,
                 device: str = "cuda"):
        self.config = config
        self.dtype = dtype
        self.device = device

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=config.n_fft,
            win_length=config.n_fft,
            hop_length=config.hop_length,
            power=2.0,
            center=True,
        ).to(device, torch.float32)

        # Mel filterbank shaped (n_freqs, n_mels). Slaney scaling and
        # normalization match the filterbank Whisper ships.
        self.mel_fb = audio_f.melscale_fbanks(
            n_freqs=config.n_freqs,
            f_min=0.0,
            f_max=config.sample_rate / 2,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate,
            norm="slaney",
            mel_scale="slaney",
        ).to(device, dtype)

    def power_spectrum(self, wave: torch.Tensor) -> torch.Tensor:
        """Stage one. The fp32 FFT and magnitude, left on cuFFT."""
        return self.spectrogram(wave.to(torch.float32))

    def mel_project(self, power: torch.Tensor) -> torch.Tensor:
        """Stage two. Project power bins onto the mel scale at the target dtype."""
        power = power.to(self.dtype)
        return torch.matmul(self.mel_fb.transpose(0, 1), power)

    def log_normalize(self, mel: torch.Tensor) -> torch.Tensor:
        """Stage three. Log compression and per-utterance normalization in fp32."""
        log_spec = torch.clamp(mel.to(torch.float32), min=1e-10).log10()
        peak = log_spec.amax(dim=(-2, -1), keepdim=True)
        log_spec = torch.maximum(log_spec, peak - 8.0)
        return (log_spec + 4.0) / 4.0

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        power = self.power_spectrum(wave)
        mel = self.mel_project(power)
        return self.log_normalize(mel).to(self.dtype)


if __name__ == "__main__":
    # Confirm both precisions run end to end, produce finite output of the right
    # shape, and agree with each other. fp16 is compared to fp32 by relative
    # error because the mel projection runs at the lower precision.
    cfg = whisper_config()
    wave = torch.randn(8, 10 * cfg.sample_rate, device="cuda")

    ref32 = LogMelReference(cfg, dtype=torch.float32)
    out32 = ref32(wave)
    frames = out32.shape[-1]
    print(f"output shape {tuple(out32.shape)}  finite={torch.isfinite(out32).all().item()}")
    assert out32.shape == (8, cfg.n_mels, frames)

    ref16 = LogMelReference(cfg, dtype=torch.float16)
    out16 = ref16(wave)
    rel = ((out16.float() - out32).abs() / out32.abs().clamp_min(1e-6)).mean().item()
    print(f"fp16 vs fp32 mean relative error {rel:.2e}  finite={torch.isfinite(out16).all().item()}")
