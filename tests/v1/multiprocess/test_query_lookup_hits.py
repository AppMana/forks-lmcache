# SPDX-License-Identifier: Apache-2.0
"""Tests for the key rows the LOOKUP handler submits to the storage manager."""

# Standard
import time
from unittest.mock import MagicMock

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    GroupedObjectKeys,
    ObjectKey,
    PrefetchHandle,
    PrefetchLockMode,
    PrefetchResult,
    ipc_key_to_object_keys,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.token_codec import pack_token_ids

# ============================================================================
# Grouped key layout
# ============================================================================


def _lookup_key(world_size: int) -> IPCCacheServerKey:
    """A lookup-side IPC key (worker_id None -> expand over all workers)."""
    return IPCCacheServerKey(
        model_name="m",
        world_size=world_size,
        num_kv_readers=1,
        worker_id=None,
        token_bytes=pack_token_ids((0,)),
        start=0,
        end=0,
        request_id="r",
    )


def _captured_lookup_key_groups(
    world_size: int, num_groups: int, chunk_hashes: list[bytes]
) -> list[GroupedObjectKeys]:
    """Drive the public ``lookup()`` and return the key rows it submits.

    The engine context is mocked so ``lookup()`` runs end-to-end; the rows
    are recovered from the ``submit_prefetch_task`` call rather than by
    reaching into a private helper.
    """
    ctx = MagicMock()
    # No previous lookup to settle: ``lookup`` claims any unreleased
    # result before ``begin_lookup`` resets it.
    _sess = ctx.session_manager.get_or_create.return_value
    _sess.take_prefetch_release.return_value = None
    ctx.chunk_size = 16
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find.return_value = MagicMock()  # non-None layout
    # lookup() requires a registered per-group layout map (it error-returns
    # without one); one entry per object group, keyed by object_group_id.
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {
        gid: MagicMock() for gid in range(num_groups)
    }
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1] * num_groups
    )
    ctx.layout_desc_registry.find_kv_ranks.return_value = set()
    ctx.token_hasher.compute_packed_chunk_hashes.return_value = chunk_hashes

    module = LookupModule(ctx)
    module.lookup(_lookup_key(world_size=world_size), tp_size=1)

    ctx.storage_manager.submit_prefetch_task.assert_called_once()
    spec = ctx.storage_manager.submit_prefetch_task.call_args.args[0]
    assert spec.fetching_policy == "prefix"
    return spec.key_groups


def test_lookup_submits_one_row_per_group_and_rank_group_major():
    """lookup() submits one chunk-ordered key row per (object group, kv_rank),
    group-major / rank-minor, each row carrying its own group id."""
    rows = _captured_lookup_key_groups(
        world_size=2, num_groups=2, chunk_hashes=[b"c0", b"c1"]
    )

    # 2 groups * 2 ranks rows, each of 2 chunks.
    assert len(rows) == 4
    assert [row.object_group_id for row in rows] == [0, 0, 1, 1]
    for row in rows:
        assert [k.chunk_hash for k in row.keys] == [b"c0", b"c1"]
        assert {k.object_group_id for k in row.keys} == {row.object_group_id}
        assert len({k.kv_rank for k in row.keys}) == 1
    # The two rank rows of one group address distinct kv_ranks.
    assert rows[0].keys[0].kv_rank != rows[1].keys[0].kv_rank
    # Every group's rows use the same rank order.
    assert [r.keys[0].kv_rank for r in rows[:2]] == [
        r.keys[0].kv_rank for r in rows[2:]
    ]


def test_lookup_single_group_matches_single_group_expansion():
    """With one object group the submitted keys are exactly the single-group
    expansion, one row per kv_rank (the object-group-separation-disabled /
    non-hybrid case)."""
    chunk_hashes = [b"c0", b"c1"]
    rows = _captured_lookup_key_groups(
        world_size=2, num_groups=1, chunk_hashes=chunk_hashes
    )

    expected = ipc_key_to_object_keys(_lookup_key(world_size=2), chunk_hashes, [0])[0]
    assert len(rows) == 2
    submitted = [k for row in rows for k in row.keys]
    assert sorted(submitted, key=repr) == sorted(expected, key=repr)


def test_lookup_hashing_stops_at_key_end() -> None:
    """LOOKUP must not hash chunks beyond the IPC key's requested range."""
    ctx = MagicMock()
    # No previous lookup to settle: ``lookup`` claims any unreleased
    # result before ``begin_lookup`` resets it.
    _sess = ctx.session_manager.get_or_create.return_value
    _sess.take_prefetch_release.return_value = None
    ctx.chunk_size = 16
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find.return_value = MagicMock()
    ctx.token_hasher.compute_packed_chunk_hashes.return_value = []
    key = IPCCacheServerKey(
        model_name="m",
        world_size=1,
        num_kv_readers=1,
        worker_id=None,
        token_bytes=pack_token_ids(range(32)),
        start=0,
        end=16,
        request_id="r",
    )

    LookupModule(ctx).lookup(key, tp_size=1)

    ctx.token_hasher.compute_packed_chunk_hashes.assert_called_once_with(
        pack_token_ids(range(32)), end=16
    )


def test_lookup_submits_every_object_groups_layout():
    """lookup() hands the prefetch the registry's per-group layouts so each
    key's L1 buffer is sized by its own object group."""
    ctx = MagicMock()
    _sess = ctx.session_manager.get_or_create.return_value
    _sess.take_prefetch_release.return_value = None
    ctx.chunk_size = 16
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find.return_value = MagicMock()  # non-None layout
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1, 2]
    )
    ctx.layout_desc_registry.find_kv_ranks.return_value = set()
    group_layouts = {0: MagicMock(), 1: MagicMock()}
    ctx.layout_desc_registry.find_group_layout_descs.return_value = group_layouts
    ctx.token_hasher.compute_packed_chunk_hashes.return_value = [b"c0", b"c1"]

    module = LookupModule(ctx)
    module.lookup(_lookup_key(world_size=1), tp_size=1)

    ctx.storage_manager.submit_prefetch_task.assert_called_once()
    spec = ctx.storage_manager.submit_prefetch_task.call_args.args[0]
    assert [row.layout_desc for row in spec.key_groups] == [
        group_layouts[0],
        group_layouts[1],
    ]


# ============================================================================
# Rank-local lookups (one LMCache server per pipeline stage)
# ============================================================================


def _rank(world_size: int, worker_id: int) -> int:
    return ObjectKey.ComputeKVRank(world_size, worker_id, world_size, worker_id)


def _handle_for_spec(spec, external_request_id: str = "") -> PrefetchHandle:
    """A storage-manager handle sized like the submitted spec."""
    return PrefetchHandle(
        prefetch_request_id=0,
        external_request_id=external_request_id,
        total_requested_keys=len(spec.key_groups) * spec.group_size,
        submit_time=time.monotonic(),
        sliding_windows=tuple(row.sliding_window_size for row in spec.key_groups),
    )


def _split_lookup_ctx(local_worker_ids: set[int], world_size: int = 2):
    """A mocked engine context whose registry holds ``local_worker_ids`` and
    two full-attention object groups; ``submit_prefetch_task`` returns a
    handle sized for each submitted spec."""
    ctx = MagicMock()
    ctx.chunk_size = 16
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find.return_value = MagicMock()
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1, -1]
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {
        0: MagicMock(),
        1: MagicMock(),
    }
    ctx.layout_desc_registry.find_kv_ranks.return_value = {
        _rank(world_size, w) for w in local_worker_ids
    }
    ctx.token_hasher.compute_packed_chunk_hashes.return_value = [b"c0", b"c1"]
    ctx.storage_manager.submit_prefetch_task.side_effect = _handle_for_spec
    session = ctx.session_manager.get_or_create.return_value
    session.take_prefetch_release.return_value = None
    session.prefetch_hit_chunks = -1
    session.prefetch_locked_gids = ()
    session.lookup_ipc_key = None
    return ctx


def _submitted_specs(ctx) -> list:
    return [c.args[0] for c in ctx.storage_manager.submit_prefetch_task.call_args_list]


def test_lookup_prefetches_local_ranks_and_checks_foreign_ranks_for_existence():
    """Keys of the ranks registered in this server are prefetched; keys of
    the other ranks are only checked for existence on L2, in a separate
    request that loads and locks nothing."""
    ctx = _split_lookup_ctx(local_worker_ids={0})
    module = LookupModule(ctx)
    module.lookup(_lookup_key(world_size=2), tp_size=1)

    local, foreign = _submitted_specs(ctx)
    assert [row.object_group_id for row in local.key_groups] == [0, 1]
    for row in local.key_groups:
        assert [k.chunk_hash for k in row.keys] == [b"c0", b"c1"]
        assert {k.kv_rank for k in row.keys} == {_rank(2, 0)}
    assert [row.object_group_id for row in foreign.key_groups] == [0, 1]
    for row in foreign.key_groups:
        assert [k.chunk_hash for k in row.keys] == [b"c0", b"c1"]
        assert {k.kv_rank for k in row.keys} == {_rank(2, 1)}
    assert local.lock_mode is PrefetchLockMode.LOCK
    group_layouts = ctx.layout_desc_registry.find_group_layout_descs.return_value
    assert [row.layout_desc for row in local.key_groups] == [
        group_layouts[0],
        group_layouts[1],
    ]
    assert foreign.lock_mode is PrefetchLockMode.EXISTS


def test_lookup_with_every_rank_local_submits_one_prefetch():
    ctx = _split_lookup_ctx(local_worker_ids={0, 1})
    module = LookupModule(ctx)
    module.lookup(_lookup_key(world_size=2), tp_size=1)

    (spec,) = _submitted_specs(ctx)
    assert len(spec.key_groups) * spec.group_size == 8


def test_lookup_with_no_registered_rank_treats_every_rank_as_local():
    ctx = _split_lookup_ctx(local_worker_ids=set())
    module = LookupModule(ctx)
    module.lookup(_lookup_key(world_size=2), tp_size=1)

    (spec,) = _submitted_specs(ctx)
    assert len(spec.key_groups) * spec.group_size == 8


def _bitmap(size: int, leading_ones: int):
    # First Party
    from lmcache.lmcache_native import Bitmap

    return Bitmap(size, leading_ones)


def _prefetch_result(row_bits: list) -> PrefetchResult:
    return PrefetchResult(
        hit_cells=row_bits,
        l1_hit_cells=[_bitmap(len(b), 0) for b in row_bits],
        l2_hit_cells=row_bits,
    )


def test_query_prefetch_status_hits_the_chunks_every_rank_holds():
    """The hit length is the number of leading chunks present for the local
    ranks AND existing on L2 for the foreign ranks; local keys prefetched
    beyond that prefix are released again."""
    ctx = _split_lookup_ctx(local_worker_ids={0})
    module = LookupModule(ctx)
    module.lookup(_lookup_key(world_size=2), tp_size=1)
    local_groups = _submitted_specs(ctx)[0].key_groups

    # Return complete results in local-then-foreign polling order.
    ctx.storage_manager.query_prefetch_status.side_effect = [
        _prefetch_result([_bitmap(2, 2), _bitmap(2, 2)]),
        _prefetch_result([_bitmap(2, 1), _bitmap(2, 1)]),
    ]

    assert module.query_prefetch_status("r") == 1

    # Chunk 1 of each local group was prefetched and locked for nothing.
    ctx.storage_manager.finish_read_prefetched.assert_called_once_with(
        [local_groups[0].keys[1], local_groups[1].keys[1]], read_locks=1
    )
    session = ctx.session_manager.get_or_create.return_value
    session.record_prefetch_result.assert_called_once_with(1, (0, 1), {})


def test_free_lookup_locks_releases_only_local_rank_keys():
    """Foreign ranks' keys were never locked in this server's L1."""
    ctx = _split_lookup_ctx(local_worker_ids={0})
    session = ctx.session_manager.get_or_create.return_value
    session.prefetch_hit_chunks = 2
    module = LookupModule(ctx)
    key = IPCCacheServerKey(
        model_name="m",
        world_size=2,
        num_kv_readers=1,
        worker_id=None,
        token_bytes=pack_token_ids(range(32)),
        start=0,
        end=32,
        request_id="r",
    )
    module.free_lookup_locks(key, tp_size=1)

    released = ctx.storage_manager.finish_read_prefetched.call_args.args[0]
    assert len(released) == 4
    assert {k.kv_rank for k in released} == {_rank(2, 0)}


def test_split_lookup_keeps_consumed_local_result_while_foreign_is_pending():
    ctx = _split_lookup_ctx(local_worker_ids={0})
    module = LookupModule(ctx)
    module.lookup(_lookup_key(world_size=2), tp_size=1)
    ctx.storage_manager.query_prefetch_status.side_effect = [
        _prefetch_result([_bitmap(2, 2), _bitmap(2, 2)]),
        None,
        _prefetch_result([_bitmap(2, 2), _bitmap(2, 2)]),
    ]
    assert module.query_prefetch_status("r") is None
    assert module.query_prefetch_status("r") == 2
    assert ctx.storage_manager.query_prefetch_status.call_count == 3
    ctx.storage_manager.finish_read_prefetched.assert_not_called()


def test_split_lookup_releases_every_local_lock_when_foreign_prefix_misses():
    ctx = _split_lookup_ctx(local_worker_ids={0})
    module = LookupModule(ctx)
    module.lookup(_lookup_key(world_size=2), tp_size=1)
    local_groups = _submitted_specs(ctx)[0].key_groups
    ctx.storage_manager.query_prefetch_status.side_effect = [
        _prefetch_result([_bitmap(2, 2), _bitmap(2, 2)]),
        _prefetch_result([_bitmap(2, 0), _bitmap(2, 0)]),
    ]
    assert module.query_prefetch_status("r") == 0
    ctx.storage_manager.finish_read_prefetched.assert_called_once_with(
        [*local_groups[0].keys, *local_groups[1].keys], read_locks=1
    )


def test_split_lookup_truncation_keeps_sliding_windows_servable():
    """With a sliding-window group, the shorter common prefix only counts
    chunks whose window keys are still locked; the rest are released."""
    ctx = _split_lookup_ctx(local_worker_ids={0})
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1, 1]
    )
    ctx.token_hasher.compute_packed_chunk_hashes.return_value = [b"c0", b"c1", b"c2"]
    module = LookupModule(ctx)
    module.lookup(_lookup_key(world_size=2), tp_size=1)
    local_groups = _submitted_specs(ctx)[0].key_groups
    # Local prefetch retained the full-attention prefix (chunks 0, 1, 2) and
    # the window chunk of the sliding group at chunk 2 (index 2); the foreign
    # ranks only hold chunks 0-1.
    sw_row = _bitmap(3, 0)
    sw_row.batched_set([2])
    ctx.storage_manager.query_prefetch_status.side_effect = [
        _prefetch_result([_bitmap(3, 3), sw_row]),
        _prefetch_result([_bitmap(3, 2), _bitmap(3, 2)]),
    ]

    # Chunk 1's sliding-window key (index 1) was released at the longer
    # hit, so the servable common prefix is chunk 0 alone.
    assert module.query_prefetch_status("r") == 0
    released = ctx.storage_manager.finish_read_prefetched.call_args.args[0]
    assert released == [*local_groups[0].keys, local_groups[1].keys[2]]
