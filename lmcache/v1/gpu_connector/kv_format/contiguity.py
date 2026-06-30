# SPDX-License-Identifier: Apache-2.0
"""Contiguous-view recovery for raw engine KV caches.

Metadata-only (zero-copy) preprocessing that makes a tensor's ``.shape``
reflect its physical layout. Engine- and format-agnostic: ``detect_format``
runs it before detection, and CUDA-IPC callers use it standalone.
"""

# mypy: disable-error-code="union-attr"
# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache

logger = init_logger(__name__)


def attempt_permute_to_contiguous_view(
    kv_caches: DiscoverableKVCache,
) -> DiscoverableKVCache:
    """Return a contiguous view of *kv_caches*, metadata-only (no copy).

    For a tensor leaf: reorders the dims by stride magnitude so shape
    lines up with a contiguous layout. For a list: recurses into each
    element. Tensor leaves alias the input's storage; list nodes are
    freshly allocated but hold the same tensor objects (or their
    permuted views).

    Recovers the vLLM HND case: tensors allocated physically as
    ``[2, NB, NH, BS, HS]`` but exposed logically as
    ``[2, NB, BS, NH, HS]`` via a dim permute. Sorting dims by stride
    undoes the permute without touching storage.

    For tensors that remain non-contiguous even after dim-permute
    recovery (e.g. vLLM unified KV pool views where dim-0 has an
    inflated periodic stride because every block slot is padded to a
    model-wide maximum), this function returns the tensor unchanged.
    Rationale: :class:`CudaIPCWrapper` transports ``(shape, stride,
    storage_offset)`` verbatim and the receiver rebuilds the view via
    ``torch.Tensor.set_(storage, offset, shape, stride)``, which
    supports arbitrary strided views (including periodic-dim-0 and
    ``as_strided``-produced layouts) and yields a bit-identical view.
    Downstream consumers that rely on ``shape`` alone to infer the
    physical layout must therefore also consult ``stride``.

    We deliberately never fall back to ``.contiguous()`` (which would
    allocate and copy), so the caller's zero-copy invariant is
    preserved.
    """
    if isinstance(kv_caches, torch.Tensor):
        if kv_caches.is_contiguous():
            return kv_caches
        strides = kv_caches.stride()
        perm = sorted(range(kv_caches.ndim), key=lambda i: strides[i], reverse=True)
        result = kv_caches.permute(perm)
        if result.is_contiguous():
            return result
        logger.debug(
            "attempt_permute_to_contiguous_view: accepting strided view; "
            "CudaIPC carries shape/stride/storage_offset. shape=%s, "
            "stride=%s, storage_offset=%d, storage_nbytes=%s, dtype=%s",
            tuple(result.shape),
            tuple(result.stride()),
            int(result.storage_offset()),
            int(result.untyped_storage().nbytes()),
            result.dtype,
        )
        return result
    return [attempt_permute_to_contiguous_view(sub) for sub in kv_caches]
