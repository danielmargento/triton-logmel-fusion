"""Correctness of the fused frontend against the reference oracle.

The fused kernel must match the decomposed PyTorch pipeline across the grid. The
fp32 path is held to a tight tolerance because its matmul runs in true IEEE
precision; the fp16 path is looser because its matmul runs at half precision.
"""

import pytest
import torch

from logmel import whisper_config
from logmel.reference import LogMelReference
from logmel.triton_fused import FusedLogMel

CASES = [(1, 1), (8, 10), (32, 30), (1, 30), (8, 1)]


@pytest.mark.parametrize("dtype,rtol,atol", [
    (torch.float32, 1e-4, 1e-5),
    (torch.float16, 2e-2, 2e-3),
])
@pytest.mark.parametrize("batch,duration", CASES)
def test_fused_matches_reference(dtype, rtol, atol, batch, duration):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    config = whisper_config()
    torch.manual_seed(0)
    wave = torch.randn(batch, duration * config.sample_rate, device="cuda", dtype=dtype)

    reference = LogMelReference(config, dtype=dtype)(wave).float()
    fused = FusedLogMel(config, dtype=dtype)(wave).float()

    torch.testing.assert_close(fused, reference, rtol=rtol, atol=atol)
