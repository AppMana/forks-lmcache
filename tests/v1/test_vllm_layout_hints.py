# SPDX-License-Identifier: Apache-2.0
"""The NHD/HND layout hint is read from vLLM's resolved KV cache layout.

vLLM resolves one ``KVCacheLayout`` per engine (``cache_config.kv_cache_layout``,
e.g. ``LBNHC``) and no longer exposes ``get_kv_cache_layout``; the hint must
come from the config the connector already holds.
"""

# Standard
from enum import Enum
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.utils import (
    try_get_vllm_kv_cache_layout,
    vllm_layout_hints,
)


def _config(layout):
    return SimpleNamespace(cache_config=SimpleNamespace(kv_cache_layout=layout))


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        ("LBNHC", "NHD"),
        ("LBHNC", "HND"),
        ("BLNHC", "NHD"),
        ("BLHNC", "HND"),
        ("BHLNC", "HND"),
        ("NHD", "NHD"),
        ("HND", "HND"),
    ],
)
def test_layout_from_resolved_config(layout, expected):
    assert try_get_vllm_kv_cache_layout(_config(layout)) == expected
    assert vllm_layout_hints(_config(layout)) == {"kv_layout": expected}


def test_layout_accepts_enum_member():
    class KVCacheLayout(Enum):
        LBHNC = (0, 1, 2, 3, 4)

    assert try_get_vllm_kv_cache_layout(_config(KVCacheLayout.LBHNC)) == "HND"


def test_layout_without_nhd_hnd_equivalent_yields_no_hint():
    assert try_get_vllm_kv_cache_layout(_config("LHBNC")) is None
    assert vllm_layout_hints(_config("LHBNC")) == {}


def test_unresolved_layout_falls_back_without_raising(monkeypatch):
    monkeypatch.delitem(__import__("sys").modules, "vllm", raising=False)
    assert try_get_vllm_kv_cache_layout(_config(None)) in ("NHD", "HND", None)
    assert try_get_vllm_kv_cache_layout(None) in ("NHD", "HND", None)
