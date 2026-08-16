"""Asynchronous file side channel for captured router replay tensors."""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_PUBLISHERS: dict[tuple[str, int], RoutedExpertsSideChannelPublisher] = {}
_PUBLISHERS_LOCK = threading.Lock()


class RoutedExpertsSideChannelPublisher:
    """Publish CPU tensors without serializing their bodies through IPC/HTTP.

    The final path is installed with ``os.replace`` only after both fields have
    been written. Consumers can therefore wait for the descriptor path without
    observing a partially written payload.
    """

    def __init__(self, root: str, workers: int) -> None:
        configured_root = Path(root).expanduser()
        if not configured_root.is_absolute():
            raise ValueError("routed experts side-channel directory must be absolute")
        self.root = configured_root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(workers)),
            thread_name_prefix="routed-experts-publish",
        )

    @staticmethod
    def _field(tensor: torch.Tensor, offset: int) -> dict[str, Any]:
        if tensor.device.type != "cpu":
            raise ValueError("routed experts side-channel tensors must be on CPU")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return {
            "offset": offset,
            "nbytes": tensor.numel() * tensor.element_size(),
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }

    @staticmethod
    def _write_tensor(fd: int, tensor: torch.Tensor, offset: int) -> None:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        view = memoryview(tensor.numpy()).cast("B")
        written = 0
        while written < len(view):
            count = os.pwrite(fd, view[written:], offset + written)
            if count <= 0:
                raise OSError("short write while publishing routed experts")
            written += count

    def publish(
        self,
        *,
        request_id: str,
        routed_experts: Optional[torch.Tensor],
        expert_logits: Optional[torch.Tensor],
        start_row: int,
    ) -> dict[str, Any]:
        tensors = {
            "routed_experts": routed_experts,
            "routed_expert_logits": expert_logits,
        }
        tensors = {
            name: tensor for name, tensor in tensors.items() if tensor is not None
        }
        if not tensors:
            raise ValueError("at least one routed-experts tensor is required")

        fields: dict[str, Any] = {}
        offset = 0
        normalized: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            tensor = tensor.detach()
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            normalized[name] = tensor
            fields[name] = self._field(tensor, offset)
            offset += int(fields[name]["nbytes"])

        safe_id = "".join(
            ch if ch.isalnum() or ch in "._-" else "_" for ch in str(request_id)
        )
        stem = f"{safe_id or 'request'}.{uuid.uuid4().hex}"
        final_path = self.root / f"{stem}.bin"
        tmp_path = self.root / f".{stem}.partial"
        error_path = self.root / f".{stem}.error.json"
        descriptor = {
            "schema": "sglang.routed_experts.file.v1",
            "path": str(final_path),
            "error_path": str(error_path),
            "start_row": int(start_row),
            "rows": int(next(iter(normalized.values())).shape[0]),
            "nbytes": offset,
            "fields": fields,
        }

        queued_at = time.monotonic()

        def _publish() -> None:
            started_at = time.monotonic()
            try:
                fd = os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                try:
                    os.ftruncate(fd, offset)
                    for name, tensor in normalized.items():
                        self._write_tensor(fd, tensor, int(fields[name]["offset"]))
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.chmod(tmp_path, 0o444)
                os.replace(tmp_path, final_path)
                logger.debug(
                    "Published routed experts side payload request=%s bytes=%d queue_s=%.6f write_s=%.6f path=%s",
                    request_id,
                    offset,
                    started_at - queued_at,
                    time.monotonic() - started_at,
                    final_path,
                )
            except Exception as exc:
                try:
                    tmp_path.unlink(missing_ok=True)
                    error_path.write_text(
                        json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                        encoding="utf-8",
                    )
                except Exception:
                    logger.exception(
                        "Failed to publish routed experts side-channel error marker"
                    )
                logger.exception(
                    "Failed to publish routed experts side payload for request %s",
                    request_id,
                )

        self._executor.submit(_publish)
        return descriptor

    def close(self) -> None:
        self._executor.shutdown(wait=True)


def get_routed_experts_side_channel_publisher(
    root: Optional[str], workers: int
) -> Optional[RoutedExpertsSideChannelPublisher]:
    if not root:
        return None
    key = (str(Path(root).expanduser().resolve()), max(1, int(workers)))
    with _PUBLISHERS_LOCK:
        publisher = _PUBLISHERS.get(key)
        if publisher is None:
            publisher = RoutedExpertsSideChannelPublisher(*key)
            _PUBLISHERS[key] = publisher
        return publisher


@atexit.register
def _close_publishers() -> None:
    with _PUBLISHERS_LOCK:
        publishers = list(_PUBLISHERS.values())
        _PUBLISHERS.clear()
    for publisher in publishers:
        publisher.close()
