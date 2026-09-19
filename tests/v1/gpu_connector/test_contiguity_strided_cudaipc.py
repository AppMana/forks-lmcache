# SPDX-License-Identifier: Apache-2.0
"""Cache views must retain their shared-storage addressing across IPC.

vLLM's per-layer views of a shared KV pool are dim-0 padded (every page
strides by the pool's widest row) and may start at a nonzero storage offset
(MTP layers, CPU offload). CudaIPC carries shape, stride and storage_offset,
so the registration path must keep them; only views the transfer kernels
cannot address (interior padding) are rejected.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.contiguity import (
    attempt_permute_to_contiguous_view,
)


def test_dim0_padded_view_with_storage_offset_keeps_addressing():
    blocks, row, pad, offset = 4, 32, 8, 7
    base = torch.arange(offset + blocks * (row + pad), dtype=torch.float32)
    view = base.as_strided((blocks, 8, 4), (row + pad, 4, 1), offset)
    assert not view.is_contiguous()

    out = attempt_permute_to_contiguous_view(view)

    assert out.shape == view.shape
    assert out.stride() == view.stride()
    assert out.storage_offset() == offset
    assert out.data_ptr() == view.data_ptr()
    torch.testing.assert_close(out, view)


def test_interior_padded_view_is_rejected():
    """The kernels honour dim-0 padding only; anything else would be
    transferred at the wrong addresses."""
    base = torch.arange(4 * 80, dtype=torch.float32)
    # Rows are tight and dim-0 is padded, but the head axis is padded too.
    view = base.as_strided((4, 2, 4, 4), (80, 32, 4, 1))

    with pytest.raises(ValueError, match="interior-dim padding"):
        attempt_permute_to_contiguous_view(view)


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
