import torch

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
