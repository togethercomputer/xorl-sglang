# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""Host-pinned RDMA staging buffer for packed sparse-delta payloads.

The ZORL fold-aware sparse-delta sender can write its packed delta to shared
storage and make every receiver read the whole file back. With staging, each
TP rank keeps ONE host-pinned uint8 buffer registered with the local Mooncake
TransferEngine; the sender RDMA-writes the packed
bytes straight into it and the apply request then decodes from memory.

Host-pinned (not GPU) staging is deliberate:
  * A GPU staging buffer the size of the delta can exhaust device memory.
  * The decode path already streams through bounded GPU slabs
    (SGLANG_SPARSE_DELTA_DECODE_SLAB_BYTES / _SELECT_SLAB_ELEMS), so the
    apply reads the staging buffer slab-wise H2D — pinned memory makes
    those copies fast, and peak GPU usage is unchanged vs the file path.
  * RDMA into registered pinned host memory is the same NIC DMA path the
    xorl sender already uses for its CPU scratch pools (source side).

Peak host memory: one buffer of round_up(nbytes, 64 MiB) per TP rank,
persistent across syncs (the file path allocated a same-sized transient
host tensor per sync via np.fromfile, so steady-state is a wash).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_ALLOC_GRANULARITY = 64 * 1024 * 1024  # grow in 64 MiB steps


def _max_staging_bytes() -> int:
    return int(
        os.environ.get(
            "SGLANG_SPARSE_DELTA_STAGING_MAX_BYTES", str(8 * 1024 * 1024 * 1024)
        )
    )


def _register_region_bytes() -> int:
    """Max bytes per Mooncake registration sub-range.

    Bounded regions avoid making one registration depend on every local RNIC
    being able to reach the peer.
    """
    return int(
        os.environ.get(
            "SGLANG_SPARSE_DELTA_STAGING_REGION_BYTES", str(1024 * 1024 * 1024)
        )
    )


class SparseDeltaStagingBuffer:
    """One host-pinned uint8 staging region, registered with Mooncake in
    bounded sub-ranges.

    ``ensure(nbytes, engine)`` grows the buffer when needed (deregister old,
    allocate pinned, register new) and returns ``(ptr, capacity)``. The
    buffer is intentionally never shrunk: fold deltas have a stable size
    band per run and re-pinning a GB costs hundreds of ms.
    """

    def __init__(self) -> None:
        self._buf: Optional[torch.Tensor] = None
        self._registered_ptr: Optional[int] = None
        self._registered_ranges: list[int] = []
        self._engine: Optional[Any] = None

    @property
    def capacity(self) -> int:
        return 0 if self._buf is None else int(self._buf.numel())

    @property
    def ptr(self) -> Optional[int]:
        return self._registered_ptr

    def ensure(
        self, nbytes: int, engine: Any, location: Optional[str] = None
    ) -> tuple[int, int]:
        if nbytes <= 0:
            raise ValueError(f"staging nbytes must be positive, got {nbytes}")
        max_bytes = _max_staging_bytes()
        if nbytes > max_bytes:
            raise ValueError(
                f"staging request of {nbytes} bytes exceeds "
                f"SGLANG_SPARSE_DELTA_STAGING_MAX_BYTES={max_bytes}"
            )
        engine_changed = self._engine is not None and self._engine is not engine
        if self._buf is not None and self.capacity >= nbytes and not engine_changed:
            return int(self._registered_ptr), self.capacity

        self.release()

        capacity = ((nbytes + _ALLOC_GRANULARITY - 1) // _ALLOC_GRANULARITY) * _ALLOC_GRANULARITY
        buf = torch.empty(capacity, dtype=torch.uint8, pin_memory=torch.cuda.is_available())
        ptr = int(buf.data_ptr())
        # Registration carries an explicit topology hint and is split into
        # bounded sub-ranges so it does not depend on every local RNIC being
        # able to reach the peer.
        location = location or os.environ.get("SGLANG_SPARSE_DELTA_STAGING_LOCATION")
        region = max(_ALLOC_GRANULARITY, _register_region_bytes())
        registered: list[int] = []
        for off in range(0, capacity, region):
            length = min(region, capacity - off)
            try:
                ret = (
                    engine.register(ptr + off, length, location)
                    if location
                    else engine.register(ptr + off, length)
                )
            except TypeError:
                ret = engine.register(ptr + off, length)
            if ret != 0:
                for prev in registered:
                    try:
                        engine.deregister(prev)
                    except Exception:  # noqa: BLE001
                        pass
                raise RuntimeError(
                    f"Mooncake registration of the sparse-delta staging buffer failed "
                    f"(ret={ret}, sub-range {off}..{off + length} of {capacity / 1e9:.2f} GB "
                    f"pinned at 0x{ptr:x}, location={location or '<auto>'})"
                )
            registered.append(ptr + off)
        self._buf = buf
        self._registered_ptr = ptr
        self._registered_ranges = registered
        self._engine = engine
        logger.info(
            "[SparseDeltaStaging] registered %.2f GB host-pinned staging at 0x%x "
            "(%d sub-range(s), location=%s)",
            capacity / 1e9,
            ptr,
            len(registered),
            location or "<auto>",
        )
        return ptr, capacity

    def view(self, nbytes: int) -> torch.Tensor:
        if self._buf is None:
            raise RuntimeError(
                "No sparse-delta staging buffer; the sender must POST "
                "staging_op='prepare' before staging_op='apply'"
            )
        if nbytes > self.capacity:
            raise ValueError(
                f"staging apply of {nbytes} bytes exceeds the prepared capacity {self.capacity}"
            )
        return self._buf[:nbytes]

    def release(self) -> None:
        if self._engine is not None:
            for ptr in self._registered_ranges:
                try:
                    self._engine.deregister(ptr)
                except Exception:  # noqa: BLE001 - engine may already be torn down
                    logger.warning(
                        "[SparseDeltaStaging] failed to deregister a staging sub-range",
                        exc_info=True,
                    )
        self._buf = None
        self._registered_ptr = None
        self._registered_ranges = []
        self._engine = None
