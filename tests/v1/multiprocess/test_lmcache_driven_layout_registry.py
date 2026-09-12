# SPDX-License-Identifier: Apache-2.0
"""Regression tests for GPU transfer layout registration lifetime."""

# Standard
from typing import Any, cast
from unittest.mock import MagicMock, patch
import sys
import types

# Third Party
import pytest
import torch


class _FakeKVLayerGroupsManager:
    """Minimal manager stub: one full-attention object group."""

    num_object_groups: int = 1

    def get_attn_desc(self) -> Any:
        """One full-attention object group."""
        # First Party
        from lmcache.v1.distributed.api import AttnWindowDesc

        return AttnWindowDesc(num_chunks_in_sw=[-1])


class _FakeGPUContext:
    """Small stand-in for GPUCacheContext used by registration tests."""

    num_layers: int = 2
    kv_layer_groups_manager: _FakeKVLayerGroupsManager = _FakeKVLayerGroupsManager()

    def close(self) -> None:
        """No-op teardown (real GPUCacheContext.close deregisters its GDS buffer)."""


class _FakeDeviceHostFuncDispatcher:
    """No-op dispatcher to avoid starting native completion threads."""

    def register(self, kind: str, handler: object, payload_type: object) -> None:
        """Record no native callback registration."""

    def start(self) -> None:
        """Start no background thread."""

    def stop(self) -> None:
        """Stop no background thread."""


@pytest.fixture
def stub_native_storage_ops() -> Any:
    """Stub native modules so MP server imports work in source-only test runs."""
    module = types.ModuleType("lmcache.native_storage_ops")
    module_any = cast(Any, module)
    module_any.TTLLock = type("TTLLock", (), {})
    module_any.Bitmap = type("Bitmap", (), {})
    module_any.PeriodicEventNotifier = type("PeriodicEventNotifier", (), {})
    with patch.dict(
        sys.modules,
        {
            "lmcache.native_storage_ops": module,
            "cupy": MagicMock(),
        },
    ):
        yield


def test_unregister_one_shared_gpu_layout_keeps_registry_until_last_instance(
    monkeypatch: pytest.MonkeyPatch,
    stub_native_storage_ops: Any,
) -> None:
    """Unregistering one shared GPU instance must not remove the shared layout."""
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry
    from lmcache.v1.multiprocess.modules import (
        lmcache_driven_transfer as lmcache_driven_transfer_mod,
    )

    layout_desc = MemoryLayoutDesc(
        shapes=[torch.Size([2, 16, 32])],
        dtypes=[torch.float32],
    )
    ctx = MagicMock()
    ctx.chunk_size = 16
    ctx.layout_desc_registry = LayoutDescRegistry()

    def fake_create_cache_context(
        kv_caches: object,
        lmcache_tokens_per_chunk: int,
        layout_hints: object = None,
        engine_group_infos: object = (),
        engine_type: object = None,
        separate_object_groups: bool = False,
    ) -> _FakeGPUContext:
        """Return a fake cache context without touching CUDA or wrappers."""
        return _FakeGPUContext()

    def fake_layout_desc(
        gpu_context: _FakeGPUContext,
        num_tokens: int,
        object_group_id: int = 0,
    ) -> MemoryLayoutDesc:
        """Return the shared layout descriptor used by both registrations."""
        return layout_desc

    monkeypatch.setattr(
        lmcache_driven_transfer_mod,
        "DeviceHostFuncDispatcher",
        _FakeDeviceHostFuncDispatcher,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer_mod,
        "create_cache_context",
        fake_create_cache_context,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer_mod,
        "get_layout_desc",
        fake_layout_desc,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer_mod.torch_dev,
        "empty_cache",
        lambda: None,
        raising=False,
    )

    module = lmcache_driven_transfer_mod.LMCacheDrivenTransferModule(ctx)
    module.register_kv_cache(1, [], "shared-model", 1, EngineType.VLLM, {}, [], 0)
    module.register_kv_cache(2, [], "shared-model", 1, EngineType.VLLM, {}, [], 0)
    assert ctx.layout_desc_registry.find("shared-model", 1) is layout_desc

    module.unregister_kv_cache(1)

    assert ctx.layout_desc_registry.find("shared-model", 1) is layout_desc

    module.unregister_kv_cache(2)
    assert ctx.layout_desc_registry.find("shared-model", 1) is None


def _layout() -> Any:
    """A minimal layout descriptor for registry tests."""
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc

    return MemoryLayoutDesc(shapes=[torch.Size([2, 4])], dtypes=[torch.float16])


def test_registry_attn_desc_roundtrip() -> None:
    """register stores the attention-window descriptor; find_attn_desc reads it."""
    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    registry = LayoutDescRegistry()
    registry.register(
        "m", 2, _layout(), attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1, 2])
    )

    assert registry.find_attn_desc("m", 2).num_chunks_in_sw == [-1, 2]


def test_registry_attn_desc_raises_when_unregistered() -> None:
    """find_attn_desc raises for an unknown (model, world_size) pair."""
    # First Party
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    registry = LayoutDescRegistry()

    with pytest.raises(ValueError, match="No attention-window descriptor"):
        registry.find_attn_desc("missing", 1)


def test_registry_windows_default_single_group_when_omitted() -> None:
    """A registration without windows resolves to a single full-attention group."""
    # First Party
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    registry = LayoutDescRegistry()
    registry.register("m", 1, _layout())

    assert registry.find_attn_desc("m", 1).num_chunks_in_sw == [-1]


def test_registry_windows_updated_on_reregister() -> None:
    """Re-registering the same pair refreshes the stored windows."""
    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    registry = LayoutDescRegistry()
    registry.register(
        "m", 1, _layout(), attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1])
    )
    registry.register(
        "m", 1, _layout(), attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1, 4])
    )

    assert registry.find_attn_desc("m", 1).num_chunks_in_sw == [-1, 4]


def test_registry_group_layout_descs_roundtrip() -> None:
    """register stores one layout per object group; find_group_layout_descs
    reads them back in object-group order and find still returns group 0's."""
    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    group0 = _layout()
    group1 = MemoryLayoutDesc(shapes=[torch.Size([2, 1])], dtypes=[torch.float16])
    registry = LayoutDescRegistry()
    registry.register(
        "m",
        2,
        group0,
        attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1, 2]),
        group_layout_descs=[group0, group1],
    )

    assert registry.find_group_layout_descs("m", 2) == [group0, group1]
    assert registry.find("m", 2) is group0


def test_registry_group_layout_descs_default_single_group() -> None:
    """A registration without per-group layouts covers one object group."""
    # First Party
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    layout = _layout()
    registry = LayoutDescRegistry()
    registry.register("m", 1, layout)

    assert registry.find_group_layout_descs("m", 1) == [layout]


def test_registry_group_layout_descs_must_cover_every_window() -> None:
    """One layout per attention window, and group 0's layout is layout_desc."""
    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    other = MemoryLayoutDesc(shapes=[torch.Size([2, 1])], dtypes=[torch.float16])
    registry = LayoutDescRegistry()
    with pytest.raises(ValueError, match="object group"):
        registry.register(
            "m",
            1,
            _layout(),
            attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1, 2]),
            group_layout_descs=[_layout()],
        )
    with pytest.raises(ValueError, match="object group"):
        registry.register(
            "m",
            1,
            _layout(),
            attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1, 2]),
            group_layout_descs=[other, _layout()],
        )


def test_registry_group_layout_descs_raise_when_unregistered() -> None:
    """find_group_layout_descs raises for an unknown (model, world_size) pair."""
    # First Party
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    with pytest.raises(ValueError, match="No layout"):
        LayoutDescRegistry().find_group_layout_descs("missing", 1)


class _FakeTwoGroupKVLayerGroupsManager(_FakeKVLayerGroupsManager):
    """Manager stub with a full-attention and a sliding-window object group."""

    num_object_groups: int = 2

    def get_attn_desc(self) -> Any:
        # First Party
        from lmcache.v1.distributed.api import AttnWindowDesc

        return AttnWindowDesc(num_chunks_in_sw=[-1, 2])


class _FakeTwoGroupGPUContext(_FakeGPUContext):
    kv_layer_groups_manager: _FakeKVLayerGroupsManager = (
        _FakeTwoGroupKVLayerGroupsManager()
    )


def test_gpu_registration_registers_one_layout_per_object_group(
    monkeypatch: pytest.MonkeyPatch,
    stub_native_storage_ops: Any,
) -> None:
    """Registering a hybrid KV cache records every object group's layout."""
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry
    from lmcache.v1.multiprocess.modules import (
        lmcache_driven_transfer as lmcache_driven_transfer_mod,
    )

    layouts = [
        MemoryLayoutDesc(shapes=[torch.Size([2, 16, 32])], dtypes=[torch.float32]),
        MemoryLayoutDesc(shapes=[torch.Size([2, 4, 32])], dtypes=[torch.float32]),
    ]
    ctx = MagicMock()
    ctx.chunk_size = 16
    ctx.layout_desc_registry = LayoutDescRegistry()

    def fake_create_cache_context(*args: object, **kwargs: object) -> _FakeGPUContext:
        return _FakeTwoGroupGPUContext()

    def fake_layout_desc(
        gpu_context: _FakeGPUContext,
        num_tokens: int,
        object_group_id: int,
    ) -> MemoryLayoutDesc:
        return layouts[object_group_id]

    monkeypatch.setattr(
        lmcache_driven_transfer_mod,
        "DeviceHostFuncDispatcher",
        _FakeDeviceHostFuncDispatcher,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer_mod,
        "create_cache_context",
        fake_create_cache_context,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer_mod, "get_layout_desc", fake_layout_desc
    )
    monkeypatch.setattr(
        lmcache_driven_transfer_mod.torch_dev,
        "empty_cache",
        lambda: None,
        raising=False,
    )

    module = lmcache_driven_transfer_mod.LMCacheDrivenTransferModule(ctx)
    module.register_kv_cache(1, [], "hybrid-model", 1, EngineType.VLLM, {}, [], 0)

    assert ctx.layout_desc_registry.find("hybrid-model", 1) is layouts[0]
    assert ctx.layout_desc_registry.find_group_layout_descs("hybrid-model", 1) == (
        layouts
    )


def _rank(world_size: int, worker_id: int) -> int:
    # First Party
    from lmcache.v1.distributed.api import ObjectKey

    return ObjectKey.ComputeKVRank(world_size, worker_id, world_size, worker_id)


def test_registry_tracks_kv_ranks_per_registration() -> None:
    """Rank-carrying registrations are counted per kv_rank and released one
    instance at a time; find_kv_ranks reports the ranks still registered."""
    # First Party
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    registry = LayoutDescRegistry()
    assert registry.find_kv_ranks("m", 2) == set()
    registry.register("m", 2, _layout(), kv_rank=_rank(2, 0))
    registry.register("m", 2, _layout(), kv_rank=_rank(2, 0))
    assert registry.find_kv_ranks("m", 2) == {_rank(2, 0)}

    registry.unregister("m", 2, kv_rank=_rank(2, 0))
    assert registry.find_kv_ranks("m", 2) == {_rank(2, 0)}
    registry.unregister("m", 2, kv_rank=_rank(2, 0))
    assert registry.find_kv_ranks("m", 2) == set()
    assert registry.find("m", 2) is None


def test_registry_accepts_second_rank_with_the_same_layout() -> None:
    """Ranks holding identical layouts (tensor parallelism without MLA) share
    one server and are all local to it."""
    # First Party
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    registry = LayoutDescRegistry()
    registry.register("m", 2, _layout(), kv_rank=_rank(2, 0))
    registry.register("m", 2, _layout(), kv_rank=_rank(2, 1))

    assert registry.find_kv_ranks("m", 2) == {_rank(2, 0), _rank(2, 1)}


def test_registry_rejects_second_rank_with_a_different_layout() -> None:
    """Two ranks with different layouts cannot share one (model_name,
    world_size) entry; the registration fails instead of overwriting."""
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    other = MemoryLayoutDesc(shapes=[torch.Size([2, 1])], dtypes=[torch.float16])
    registry = LayoutDescRegistry()
    registry.register("m", 2, _layout(), kv_rank=_rank(2, 0))

    with pytest.raises(ValueError, match="one LMCache server per rank"):
        registry.register("m", 2, other, kv_rank=_rank(2, 1))

    assert registry.find_kv_ranks("m", 2) == {_rank(2, 0)}
    assert registry.find("m", 2) == _layout()


def test_registry_rank_less_registration_cannot_replace_ranked_layout() -> None:
    """A registration without a rank (blend, engine-driven) neither adds a
    rank nor trips the per-rank layout check."""
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    other = MemoryLayoutDesc(shapes=[torch.Size([2, 1])], dtypes=[torch.float16])
    registry = LayoutDescRegistry()
    registry.register("m", 2, _layout(), kv_rank=_rank(2, 0))
    with pytest.raises(ValueError, match="one LMCache server per rank"):
        registry.register("m", 2, other)

    assert registry.find_kv_ranks("m", 2) == {_rank(2, 0)}


def test_register_kv_cache_protocol_carries_worker_id() -> None:
    """REGISTER_KV_CACHE ends with the registering worker's kv worker id."""
    # First Party
    from lmcache.v1.multiprocess.protocol import RequestType, get_payload_classes

    payload_classes = get_payload_classes(RequestType.REGISTER_KV_CACHE)
    assert len(payload_classes) == 8
    assert payload_classes[7] is int


def test_gpu_registration_records_the_workers_kv_rank(
    monkeypatch: pytest.MonkeyPatch,
    stub_native_storage_ops: Any,
) -> None:
    """Registering a KV cache records the worker's kv_rank under the pair and
    unregistering the instance releases it."""
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry
    from lmcache.v1.multiprocess.modules import (
        lmcache_driven_transfer as lmcache_driven_transfer_mod,
    )

    layout_desc = MemoryLayoutDesc(
        shapes=[torch.Size([2, 16, 32])], dtypes=[torch.float32]
    )
    ctx = MagicMock()
    ctx.chunk_size = 16
    ctx.layout_desc_registry = LayoutDescRegistry()

    monkeypatch.setattr(
        lmcache_driven_transfer_mod,
        "DeviceHostFuncDispatcher",
        _FakeDeviceHostFuncDispatcher,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer_mod,
        "create_cache_context",
        lambda *args, **kwargs: _FakeGPUContext(),
    )
    monkeypatch.setattr(
        lmcache_driven_transfer_mod,
        "get_layout_desc",
        lambda gpu_context, num_tokens, object_group_id: layout_desc,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer_mod.torch_dev,
        "empty_cache",
        lambda: None,
        raising=False,
    )

    module = lmcache_driven_transfer_mod.LMCacheDrivenTransferModule(ctx)
    module.register_kv_cache(7, [], "m", 2, EngineType.VLLM, {}, [], 1)

    assert ctx.layout_desc_registry.find_kv_ranks("m", 2) == {_rank(2, 1)}

    module.unregister_kv_cache(7)
    assert ctx.layout_desc_registry.find_kv_ranks("m", 2) == set()
