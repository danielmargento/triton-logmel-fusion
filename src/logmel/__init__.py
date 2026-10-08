"""Log-mel frontend implementations and shared configuration.

Exposes the reference pipeline and the mel configuration. The Triton and CUDA
kernels live in their own modules and are imported directly where they are used.
"""

from .reference import LogMelReference, MelConfig, whisper_config

__all__ = ["LogMelReference", "MelConfig", "whisper_config"]
