"""Throughput of Whisper served by vLLM.

This situates the log-mel frontend in an inference engine. vLLM runs the
whole Whisper pipeline, its own mel frontend, the encoder, and autoregressive
decoding, with batching and paged attention. Measuring serving throughput shows
the scale at which the frontend operates in production, where it is a small
fraction of the work, consistent with the microbenchmark finding in this repo.
"""

import time

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset

MODEL = "openai/whisper-small"
N_REQUESTS = 64


def main() -> None:
    audio = AudioAsset("mary_had_lamb").audio_and_sample_rate
    waveform, sample_rate = audio
    clip_seconds = len(waveform) / sample_rate

    llm = LLM(
        model=MODEL,
        max_model_len=448,
        limit_mm_per_prompt={"audio": 1},
        enforce_eager=True,          # skip CUDA graph capture, which hangs on this host
        gpu_memory_utilization=0.6,
        max_num_seqs=16,
    )
    prompt = {"prompt": "<|startoftranscript|>", "multi_modal_data": {"audio": audio}}
    sampling = SamplingParams(temperature=0.0, max_tokens=200)

    llm.generate([prompt] * 4, sampling)  # warmup

    start = time.perf_counter()
    outputs = llm.generate([prompt] * N_REQUESTS, sampling)
    elapsed = time.perf_counter() - start

    print(f"model {MODEL}")
    print(f"requests {N_REQUESTS}, clip {clip_seconds:.1f} s each")
    print(f"wall time {elapsed:.2f} s")
    print(f"throughput {N_REQUESTS / elapsed:.1f} req/s, "
          f"{N_REQUESTS * clip_seconds / elapsed:.1f} audio-s per s")
    print("sample transcription:", outputs[0].outputs[0].text.strip()[:80])


if __name__ == "__main__":
    main()
