#!/usr/bin/env bash
# Regression gate for the AppMana DSV4 multiprocess fork.
set -euo pipefail
cd "$(dirname "$0")/.."
# RESP integration tests flush their target database. Require an explicitly
# supplied, disposable Redis endpoint; never default to a cluster service.
: "${REDIS_HOST:?Set REDIS_HOST to a disposable test Redis instance}"
: "${REDIS_PORT:?Set REDIS_PORT to its isolated test port}"
command -v redis-cli >/dev/null
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping | grep -qx PONG
python3 -c 'import torch; from lmcache.lmcache_redis import LMCacheRedisClient'
python3 -m pytest -q -rs \
 lmcache/integration/vllm/tests/test_mm_hash_utils.py \
 tests/v1/gpu_connector/test_contiguity_strided_cudaipc.py \
 tests/v1/test_vllm_layout_hints.py \
 tests/v1/test_vllm_mp_adapter.py \
 tests/v1/multiprocess/test_lmcache_driven_layout_registry.py \
 tests/v1/multiprocess/test_query_lookup_hits.py \
 tests/v1/multiprocess/test_free_locks.py \
 tests/v1/multiprocess/test_mq.py \
 tests/v1/multiprocess/test_worker_liveness.py \
 tests/v1/distributed/test_prefetch_controller.py \
 tests/v1/distributed/test_resp_l2_adapter_integration.py "$@"
