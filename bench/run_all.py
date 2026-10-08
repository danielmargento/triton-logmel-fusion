"""Run the full benchmark suite and print every result under a header.

This is the single reproducible command referenced by the README. Run it from
the repository root with the source directory on the path:

    PYTHONPATH=src python bench/run_all.py
"""

import baseline
import breakdown
import compare
import profile_kernels
import concurrency
import whisper_e2e


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    banner("Baseline frontend latency across the grid")
    baseline.main()

    banner("Stage breakdown: FFT vs post-FFT")
    breakdown.main()

    banner("Fused kernel vs reference")
    compare.main()

    banner("Post-FFT GPU operation counts")
    profile_kernels.main()

    banner("Tail latency under concurrent load")
    concurrency.main()

    banner("Frontend share of a Whisper forward pass")
    whisper_e2e.main()


if __name__ == "__main__":
    main()
