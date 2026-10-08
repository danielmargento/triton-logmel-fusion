"""Frontend share of a Whisper forward pass.

A microbenchmark speedup on the frontend only matters in proportion to the
frontend's share of the work. This times the log-mel frontend against the
Whisper encoder on a 30 second chunk, the unit Whisper always processes, so the
end-to-end ceiling on any frontend optimization is explicit.

Half precision inference uses autocast rather than model.half(), because
Whisper's custom LayerNorm keeps fp32 weights and a blanket half cast breaks it.
"""

import contextlib

import torch
import whisper

from timing import benchmark
from logmel import LogMelReference, whisper_config

MODELS = ["base", "small"]
BATCH_SIZES = [1, 8]
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
ENCODER_FRAMES = 3000  # Whisper always runs on a 30 second, 3000 frame window


def main() -> None:
    config = whisper_config()
    print(f"{'model':>6} {'dtype':>5} {'batch':>6} {'front_ms':>9} {'enc_ms':>9} {'front_%':>8}")
    for model_name in MODELS:
        model = whisper.load_model(model_name, device="cuda").eval().float()
        n_mels = model.dims.n_mels
        encoder = model.encoder
        for name, dtype in DTYPES.items():
            frontend = LogMelReference(config, dtype=dtype)
            autocast = (
                torch.autocast("cuda", dtype=torch.float16)
                if dtype == torch.float16
                else contextlib.nullcontext()
            )
            for batch in BATCH_SIZES:
                wave = torch.randn(batch, 30 * config.sample_rate, device="cuda", dtype=dtype)
                mel = torch.randn(batch, n_mels, ENCODER_FRAMES, device="cuda")

                def run_encoder(m=mel):
                    with autocast:
                        return encoder(m)

                with torch.no_grad():
                    front_ms = benchmark(lambda w=wave: frontend(w)).p50_ms
                    enc_ms = benchmark(run_encoder).p50_ms
                share = 100.0 * front_ms / (front_ms + enc_ms)
                print(f"{model_name:>6} {name:>5} {batch:>6} {front_ms:>9.4f} {enc_ms:>9.4f} {share:>7.2f}%")


if __name__ == "__main__":
    main()
