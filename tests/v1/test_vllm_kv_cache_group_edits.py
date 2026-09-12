# SPDX-License-Identifier: Apache-2.0
"""Registration-time edits of vLLM's ``[B, H, N, C]`` per-layer KV views.

vLLM views every per-layer KV cache as ``[num_blocks, num_head_slots,
num_states, content]`` (RFC #42082). MLA layers store one latent vector per
state, so their view is ``[B, 1, N, C]``; LMCache's MLA format is the rank-3
``[num_blocks, block_size, head_size]``, which is the same view with the
singleton head-slot axis dropped -- exactly what vLLM's own MLA backends do in
``bind_kv_cache``. These tests need the real vLLM spec classes.
"""

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# Third Party
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

# First Party
from lmcache.integration.vllm.kv_cache_group_edits import (  # noqa: E402
    apply_kv_cache_group_edits,
)

NB = 8
# DeepSeek V4 int8_ds_mla: 528-byte packed rows, 64 states per page, every
# page padded to the pool's widest row so dim-0 carries a stride larger than
# the page's own bytes.
STATES = 64
ROW = 528
PAGE_STRIDE = 87104


def _padded_page_view(num_states: int, row: int) -> torch.Tensor:
    raw = torch.zeros(NB * PAGE_STRIDE, dtype=torch.int8)
    return raw.as_strided(
        (NB, 1, num_states, row), (PAGE_STRIDE, num_states * row, row, 1)
    ).view(torch.uint8)


def _dsv4_int8_config() -> KVCacheConfig:
    compressed = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        tokens_per_state=4,
        cache_dtype_str="int8_ds_mla",
        alignment=528,
        model_version="deepseek_v4",
        state_content_bytes=528,
    )
    swa = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=128,
        cache_dtype_str="int8_ds_mla",
        alignment=528,
        model_version="deepseek_v4",
        state_content_bytes=528,
    )
    return KVCacheConfig(
        num_blocks=NB,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["mla.0", "mla.1"], compressed),
            KVCacheGroupSpec(["swa.0"], swa),
        ],
    )


def test_mla_views_drop_singleton_head_slot_axis():
    config = _dsv4_int8_config()
    assert not config.has_mamba_layers
    caches = {
        "mla.0": _padded_page_view(STATES, ROW),
        "mla.1": _padded_page_view(STATES, ROW),
        "swa.0": _padded_page_view(STATES, ROW),
    }

    edited = apply_kv_cache_group_edits(config, caches)

    for name, original in caches.items():
        view = edited[name]
        assert tuple(view.shape) == (NB, STATES, ROW), name
        assert view.stride() == (PAGE_STRIDE, ROW, 1), name
        assert view.dtype == torch.uint8
        assert view.data_ptr() == original.data_ptr(), name
        assert view.untyped_storage().data_ptr() == (
            original.untyped_storage().data_ptr()
        ), name


def test_non_mla_fused_views_are_left_alone():
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16
    )
    config = KVCacheConfig(
        num_blocks=NB,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["attn.0"], spec)],
    )
    # One KV head with K and V fused into the content axis is also
    # [B, 1, N, C]; only the spec tells it apart from an MLA latent.
    fused = torch.zeros(NB, 1, 16, 2 * 64, dtype=torch.float16)

    edited = apply_kv_cache_group_edits(config, {"attn.0": fused})

    assert edited["attn.0"] is fused
