# Pipeline-parallel sidecars

Pipeline stages can hold different layer counts and object sizes. Each stage
uses its own LMCache server; all servers share Redis L2. GPU prefix caching
remains independent.

REGISTER_KV_CACHE carries the KV worker index as its eighth field (-1 for
rank-less legacy callers). Client and server must be upgraded together. The
layout registry counts active registrations per packed KV rank and rejects
incompatible layouts before replacing an existing descriptor. Unregister and
liveness cleanup release the corresponding rank registration.

The scheduler's server prefetches only locally registered ranks. For foreign
ranks, PrefetchMode.EXISTS submits the adapter's batched lookup and releases its
locks without allocating L1 buffers or issuing GETs. The reported prefix is the
minimum complete chunk count across local and foreign ranks and all object
groups. Local loads beyond that common prefix are released. Polling preserves
one side's consumed completion while the other side is pending.

Only the scheduler's sidecar receives LOOKUP. Worker retrieval without a local
lookup acquires its own read locks, loading L2 misses using that worker's object
group layouts. Incomplete loads release their successful locks and return a
failure so vLLM can recompute. The transfer callback releases successful loads
after the GPU copy finishes. Redis presence is advisory: a key can disappear
between lookup and retrieval, so retrieval failures remain recoverable.

For pods with localhost sidecars, the existing lmcache.mp.host and
lmcache.mp.port settings stay valid. To run stages on one host, supply
lmcache.mp.pp_server_urls as a list of one URL per pipeline stage. The scheduler
uses stage 0; workers use their pipeline stage's URL. This selects independent
pipeline sidecars, not the tensor-sharded multi-server mode, and does not
support data parallelism. Without shared L2, foreign stages conservatively
miss because the scheduler cannot inspect their private L1 caches.

Tests cover incompatible registrations, rank cleanup, all-rank prefix
intersection, partial asynchronous completion, release of surplus local locks,
and an actual controller EXISTS followed by a normal load of foreign-sized
objects. Local end-to-end verification uses a two-stage DSV4 mini checkpoint,
two sidecars, and Redis; a restart empties both GPU and L1 caches before replay.

The repeatable fork regression gate is `tests/run_appmana_fork.sh`. Run it in
the vLLM/LMCache image with the repository mounted and native extensions
available. Supply `REDIS_HOST` and `REDIS_PORT` for a disposable Redis database
(the integration tests flush it). The gate covers strided MLA storage aliases,
worker-rank preservation during registration and reconnect, per-object Redis
sizing and malformed replies, pipeline prefix intersection, lock cleanup, and
MQ failure responses. Require the RESP tests to pass, not skip, for a release.
