# SPDX-License-Identifier: Apache-2.0
"""The KV layout hint is read from vLLM's resolved KV cache layout.

vLLM resolves one ``KVCacheLayout`` per engine and exposes it as
``cache_config.get_resolved_kv_cache_layout()``; it no longer exports
``get_kv_cache_layout``, so the hint must come from the config the connector
already holds.
"""

# Standard
from enum import Enum
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.utils import (
    translate_vllm_kv_cache_layout,
    try_get_vllm_kv_cache_layout,
    vllm_layout_hints,
)


class KVCacheLayout(Enum):
    LBNHC = (0, 1, 2, 3, 4)
    LBHNC = (0, 1, 3, 2, 4)
    BLNHC = (1, 0, 2, 3, 4)
    BLHNC = (1, 0, 3, 2, 4)
    BHLNC = (1, 3, 0, 2, 4)
    LHBNC = (0, 3, 1, 2, 4)


def _config(layout):
    return SimpleNamespace(
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: layout)
    )


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        (KVCacheLayout.LBNHC, "NHD"),
        (KVCacheLayout.LBHNC, "HND"),
        (KVCacheLayout.BLNHC, "BLNHC"),
        (KVCacheLayout.BLHNC, "BLHNC"),
    ],
)
def test_layout_from_resolved_config(layout, expected):
    assert try_get_vllm_kv_cache_layout(_config(layout)) == expected
    assert vllm_layout_hints(_config(layout)) == {"kv_layout": expected}


@pytest.mark.parametrize("name", ["NHD", "HND", "LBNHC", "LBHNC", "BLNHC", "BLHNC"])
def test_layout_names_translate(name):
    assert translate_vllm_kv_cache_layout(name) in ("NHD", "HND", "BLNHC", "BLHNC")


@pytest.mark.parametrize("layout", [KVCacheLayout.BHLNC, KVCacheLayout.LHBNC])
def test_heads_outermost_layouts_are_rejected(layout):
    """Per-block content fragmented per head cannot be transferred."""
    with pytest.raises(NotImplementedError):
        try_get_vllm_kv_cache_layout(_config(layout))


def test_unresolved_layout_raises_from_vllm():
    def unresolved():
        raise ValueError("KV cache layout has not been resolved")

    config = SimpleNamespace(
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=unresolved)
    )
    with pytest.raises(ValueError, match="not been resolved"):
        try_get_vllm_kv_cache_layout(config)


def test_config_without_resolver_falls_back_without_raising():
    """A tree predating vllm#51718 is queried through its legacy helper."""
    config = SimpleNamespace(cache_config=SimpleNamespace())
    assert try_get_vllm_kv_cache_layout(config) in (
        "NHD",
        "HND",
        "BLNHC",
        "BLHNC",
        None,
    )
