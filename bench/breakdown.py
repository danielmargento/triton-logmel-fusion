"""Stage breakdown of the reference frontend.

Splits frontend latency into the fp32 FFT, which stays on cuFFT, and the post
FFT stages the kernel fuses. The post FFT share is the ceiling on what fusion
can remove, so it says directly where fusion has headroom and where it does not.
Three cells span the regimes: overhead bound, mixed, and bandwidth bound.
"""

import torch

from timing import benchmark
from logmel import LogMelReference, whisper_config

CELLS = [(1, 1), (8, 10), (32, 30)]
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def main() -> None:
    config = whisper_config()
    header = f"{'dtype':>5} {'batch':>6} {'dur_s':>6} {'fft_ms':>9} {'full_ms':>9} {'post_ms':>9} {'post_%':>7}"
    print(header)
    for name, dtype in DTYPES.items():
        frontend = LogMelReference(config, dtype=dtype)
        for batch, duration in CELLS:
            wave = torch.randn(batch, duration * config.sample_rate, device="cuda", dtype=dtype)
            fft_ms = benchmark(lambda w=wave: frontend.power_spectrum(w)).p50_ms
            full_ms = benchmark(lambda w=wave: frontend(w)).p50_ms
            post_ms = full_ms - fft_ms
            share = 100.0 * post_ms / full_ms
            print(f"{name:>5} {batch:>6} {duration:>6} {fft_ms:>9.4f} {full_ms:>9.4f} {post_ms:>9.4f} {share:>6.1f}%")


if __name__ == "__main__":
    main()
