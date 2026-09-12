# SPDX-License-Identifier: Apache-2.0
"""Cache views must retain their shared-storage addressing across IPC."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.contiguity import (
    attempt_permute_to_contiguous_view,
)


def test_attempt_permute_accepts_cudaipc_strided_view_metadata():
    base = torch.arange(64, dtype=torch.float32)
    view = base[1:20:3]
    assert not view.is_contiguous()
    assert view.storage_offset() != 0

    out = attempt_permute_to_contiguous_view(view)

    assert out.shape == view.shape
    assert out.stride() == view.stride()
    assert out.storage_offset() == view.storage_offset()
    assert out.data_ptr() == view.data_ptr()
    torch.testing.assert_close(out, view)


@pytest.mark.parametrize("shape", [(8, 1, 16, 576), (4, 2, 32, 128)])
@pytest.mark.parametrize("offset", [0, 7])
def test_mla_padded_block_view_preserves_addressing(
    shape: tuple[int, int, int, int],
    offset: int,
) -> None:
    """Padded [B,H,N,C] views retain offsets, strides, and storage aliasing."""
    blocks, heads, tokens, channels = shape
    stride = (heads * tokens * channels + 64, tokens * channels, channels, 1)
    storage = torch.arange(offset + blocks * stride[0], dtype=torch.float32)
    view = storage.as_strided(shape, stride, offset)
    result = attempt_permute_to_contiguous_view(view)
    assert isinstance(result, torch.Tensor)
    assert result.shape == shape
    assert result.stride() == stride
    assert result.storage_offset() == offset
    assert result.data_ptr() == view.data_ptr()
    torch.testing.assert_close(result, view)
    result[-1, -1, -1, -1] = -123
    assert view[-1, -1, -1, -1] == -123
