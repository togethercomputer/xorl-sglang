# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""Apply packed sparse weight deltas to live model parameters.

The wire format matches ``delta_encoding.encoding.packed.pack_delta_buffer``:

    [header_len: uint64 little endian][JSON header][delta/value segments...]

The sender may use HF/SGLang logical names. When a logical name does not match
an actual parameter exactly, callers can pass P2P tensor-map locators from the
receiver. Those locators describe which slice of the full logical tensor lives
in this rank's local parameter memory.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import struct
import time
from dataclasses import dataclass
from math import ceil, prod
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FORMAT_VERSION = "delta_packed_v1"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

_STR_TO_DTYPE: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


@dataclass(frozen=True)
class PackedDeltaEntry:
    name: str
    nnz: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    deltas_offset: int
    deltas_nbytes: int
    values_offset: int
    values_nbytes: int


@dataclass
class SparseDeltaApplyStats:
    path: str
    packed_bytes: int
    tensors: int
    uncompressed_packed_bytes: int = 0
    compression: str = "none"
    total_nnz: int = 0
    applied_nnz: int = 0
    direct_tensors: int = 0
    translated_tensors: int = 0
    skipped_empty_tensors: int = 0
    decode_s: float = 0.0
    scatter_s: float = 0.0
    total_s: float = 0.0

    def message(self, verb: str = "Applied") -> str:
        compression = (
            f", compression={self.compression}, unpacked={self.uncompressed_packed_bytes / 1e6:.3f} MB"
            if self.compression != "none"
            else ""
        )
        return (
            f"{verb} sparse delta {self.path}: packed={self.packed_bytes / 1e6:.3f} MB, "
            f"tensors={self.tensors}{compression}, nnz={self.applied_nnz}/{self.total_nnz}, "
            f"direct={self.direct_tensors}, translated={self.translated_tensors}, "
            f"empty={self.skipped_empty_tensors}, decode={self.decode_s:.4f}s, "
            f"scatter={self.scatter_s:.4f}s, total={self.total_s:.4f}s"
        )


def apply_sparse_delta_file(
    model: nn.Module,
    delta_path: str | Path,
    *,
    locators: Iterable[dict[str, Any]] | None = None,
    use_pinned_scatter: bool | None = None,
    validate_only: bool = False,
    expected_sha256: str | None = None,
) -> SparseDeltaApplyStats:
    """Apply one packed sparse delta file to this rank's model parameters."""
    path = Path(delta_path)
    if expected_sha256 is not None:
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Sparse delta file sha256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
            )
    buf, compression = _read_sparse_delta_tensor(path)
    return apply_sparse_delta_tensor(
        model,
        buf,
        source_label=str(path),
        packed_bytes=int(path.stat().st_size),
        compression=compression,
        locators=locators,
        use_pinned_scatter=use_pinned_scatter,
        validate_only=validate_only,
    )


def sha256_of_tensor(buf: torch.Tensor) -> str:
    """sha256 of a CPU uint8 tensor without copying the payload."""
    view = buf.contiguous()
    return hashlib.sha256(memoryview(view.numpy())).hexdigest()


def apply_sparse_delta_tensor(
    model: nn.Module,
    buf: torch.Tensor,
    *,
    source_label: str = "<memory>",
    packed_bytes: int | None = None,
    compression: str = "none",
    locators: Iterable[dict[str, Any]] | None = None,
    use_pinned_scatter: bool | None = None,
    validate_only: bool = False,
    expected_sha256: str | None = None,
) -> SparseDeltaApplyStats:
    """Apply one packed sparse delta held in a CPU uint8 tensor.

    This is the transport-agnostic core of :func:`apply_sparse_delta_file`;
    the RDMA staging path calls it directly on the (host-pinned) staging
    buffer, skipping the shared-filesystem read entirely.
    """
    if buf.dtype != torch.uint8 or buf.device.type != "cpu":
        raise ValueError(
            f"apply_sparse_delta_tensor requires a CPU uint8 tensor, got {buf.dtype} on {buf.device}"
        )
    if expected_sha256 is not None:
        actual_sha256 = sha256_of_tensor(buf)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Sparse delta payload sha256 mismatch for {source_label}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
    entries = _unpack_header(buf)
    params = _named_parameter_dict(model)
    locator_map = _locator_map(locators or [])

    stats = SparseDeltaApplyStats(
        path=source_label,
        packed_bytes=int(packed_bytes if packed_bytes is not None else buf.numel()),
        uncompressed_packed_bytes=int(buf.numel()),
        compression=compression,
        tensors=len(entries),
    )
    if use_pinned_scatter is None:
        use_pinned_scatter = os.environ.get("SGLANG_SPARSE_DELTA_PINNED_SCATTER", "0") == "1"

    # Light validate: the scheduler runs a validate_only pass before the
    # apply pass on every TP rank, doubling decode/translate work per
    # update. With sha256 integrity already checked upstream, a header-level
    # check (every entry resolves to a param or locator with a consistent
    # shape) catches the realistic failure modes; enable via
    # SGLANG_SPARSE_DELTA_LIGHT_VALIDATE=1 for large fold deltas.
    if validate_only and os.environ.get("SGLANG_SPARSE_DELTA_LIGHT_VALIDATE", "0") == "1":
        t_total = time.perf_counter()
        for entry in entries:
            stats.total_nnz += entry.nnz
            if entry.nnz == 0:
                stats.skipped_empty_tensors += 1
                continue
            direct_param = params.get(entry.name)
            if direct_param is not None and tuple(direct_param.data.shape) == entry.shape:
                stats.direct_tensors += 1
                stats.applied_nnz += entry.nnz
                continue
            candidates = locator_map.get(entry.name, [])
            if not candidates:
                raise KeyError(
                    f"Sparse delta tensor {entry.name!r} did not match a local parameter "
                    "or any local tensor-map locator."
                )
            for loc in candidates:
                loc_shape = tuple(int(x) for x in loc["full_shape"])
                if loc_shape != entry.shape:
                    raise ValueError(
                        f"Sparse delta shape mismatch for {entry.name!r}: "
                        f"packed={entry.shape}, locator={loc_shape}"
                    )
                _param_for_locator(params, loc)  # raises if memory is unmapped
            stats.translated_tensors += 1
            stats.applied_nnz += entry.nnz
        stats.total_s = time.perf_counter() - t_total
        return stats

    t_total = time.perf_counter()
    for entry in entries:
        stats.total_nnz += entry.nnz
        if entry.nnz == 0:
            stats.skipped_empty_tensors += 1
            continue

        flat_deltas = buf[entry.deltas_offset : entry.deltas_offset + entry.deltas_nbytes]
        values = buf[entry.values_offset : entry.values_offset + entry.values_nbytes].view(dtype=entry.dtype)

        # Vectorized fused-MoE locators (kind="fused_moe"): the fold-aware
        # sparse-delta sender ships whole-layer fused 3D entries (e.g.
        # ``...experts.gate_up_proj`` [E, 2I, H]). Translating those through
        # the generic per-expert locator candidates would run E_total x 2
        # coordinate selections over the full nnz; the fused locator maps ALL
        # indices to local (expert, row, col) destinations in one shot, and —
        # because the destination param's device hosts the arithmetic — the
        # tens-of-millions-of-indices div/mod runs on GPU instead of CPU.
        fast_locators = [
            loc for loc in locator_map.get(entry.name, []) if loc.get("kind") == "fused_moe"
        ]
        if fast_locators:
            resolved = [(loc, *_param_for_locator(params, loc)) for loc in fast_locators]
            work_device = resolved[0][1].data.device
            t_decode = time.perf_counter()
            flat_indices = _decode_escape_on_device(flat_deltas, entry.nnz, work_device)
            values_dev = (
                values if work_device.type == "cpu" else values.to(device=work_device)
            )
            stats.decode_s += time.perf_counter() - t_decode
            applied_here = 0
            for loc, param, storage_offset in resolved:
                dest, value_mask = _select_fused_moe_indices(flat_indices, entry.shape, loc)
                if dest.numel() == 0:
                    continue
                loc_values = _index_values(values_dev, value_mask)
                if not validate_only:
                    t_scatter = time.perf_counter()
                    _scatter_flat(
                        param.data,
                        dest + storage_offset,
                        loc_values,
                        use_pinned_scatter=use_pinned_scatter,
                    )
                    stats.scatter_s += time.perf_counter() - t_scatter
                applied_here += int(dest.numel())
            stats.applied_nnz += applied_here
            if applied_here:
                stats.translated_tensors += 1
            else:
                stats.skipped_empty_tensors += 1
            continue

        t_decode = time.perf_counter()
        flat_indices = _decode_escape(flat_deltas, entry.nnz)
        stats.decode_s += time.perf_counter() - t_decode

        direct_param = params.get(entry.name)
        if direct_param is not None and tuple(direct_param.data.shape) == entry.shape:
            if not validate_only:
                t_scatter = time.perf_counter()
                _scatter_flat(
                    direct_param.data,
                    flat_indices,
                    values,
                    use_pinned_scatter=use_pinned_scatter,
                )
                stats.scatter_s += time.perf_counter() - t_scatter
            stats.applied_nnz += entry.nnz
            stats.direct_tensors += 1
            continue

        matched = False
        locator_candidates = locator_map.get(entry.name, [])
        for loc in locator_candidates:
            selected = _select_locator_indices(flat_indices, entry.shape, loc)
            if selected is None:
                continue
            local_indices, value_mask = selected
            if local_indices.numel() == 0:
                continue

            param, storage_offset = _param_for_locator(params, loc)
            loc_values = _index_values(values, value_mask)
            if not validate_only:
                t_scatter = time.perf_counter()
                _scatter_flat(
                    param.data,
                    local_indices + storage_offset,
                    loc_values,
                    use_pinned_scatter=use_pinned_scatter,
                )
                stats.scatter_s += time.perf_counter() - t_scatter
            stats.applied_nnz += int(local_indices.numel())
            matched = True

        if matched:
            stats.translated_tensors += 1
            continue
        if locator_candidates:
            stats.skipped_empty_tensors += 1
            continue

        raise KeyError(
            f"Sparse delta tensor {entry.name!r} did not match a local parameter "
            "or any local tensor-map locator with changed entries."
        )

    if torch.cuda.is_available() and not validate_only:
        torch.cuda.synchronize()
    stats.total_s = time.perf_counter() - t_total
    return stats


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_sparse_delta_tensor(path: Path) -> tuple[torch.Tensor, str]:
    """Read a packed delta file into a uint8 tensor with one copy.

    ``read_bytes`` + ``bytearray`` + ``frombuffer`` costs two full copies —
    at multi-GB fold deltas that is seconds of pure memcpy. Uncompressed
    files (the common case) load via ``numpy.fromfile`` instead.
    """
    with path.open("rb") as f:
        magic = f.read(len(ZSTD_MAGIC))
    if magic != ZSTD_MAGIC:
        import numpy as np

        return torch.from_numpy(np.fromfile(path, dtype=np.uint8)), "none"
    payload, compression = _read_sparse_delta_payload(path)
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8), compression


def _read_sparse_delta_payload(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    if not raw.startswith(ZSTD_MAGIC):
        return raw, "none"

    try:
        import zstandard as zstd  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Sparse delta file appears to be zstd-compressed, but the optional `zstandard` package is not installed"
        ) from exc

    with zstd.ZstdDecompressor().stream_reader(io.BytesIO(raw)) as reader:
        return reader.read(), "zstd"


def _named_parameter_dict(model: nn.Module) -> dict[str, nn.Parameter]:
    try:
        return dict(model.named_parameters(remove_duplicate=False))
    except TypeError:
        return dict(model.named_parameters())


def _align(offset: int, alignment: int) -> int:
    return ceil(offset / alignment) * alignment


def _unpack_header(buf: torch.Tensor) -> list[PackedDeltaEntry]:
    header_len = struct.unpack("<Q", buf[:8].numpy().tobytes())[0]
    header_bytes = buf[8 : 8 + header_len].numpy().tobytes()
    header = json.loads(header_bytes)
    if header.get("format") != FORMAT_VERSION:
        raise ValueError(f"Unsupported sparse delta format: {header.get('format')!r}")
    data_start = _align(8 + header_len, 8)
    return [
        PackedDeltaEntry(
            name=t["name"],
            nnz=int(t["nnz"]),
            dtype=_STR_TO_DTYPE[t["dtype"]],
            shape=tuple(int(x) for x in t["shape"]),
            deltas_offset=data_start + int(t["deltas_offset"]),
            deltas_nbytes=int(t["deltas_nbytes"]),
            values_offset=data_start + int(t["values_offset"]),
            values_nbytes=int(t["values_nbytes"]),
        )
        for t in header["tensors"]
    ]


def _decode_escape(flat_deltas: torch.Tensor, nnz: int) -> torch.Tensor:
    if nnz == 0:
        return torch.empty(0, dtype=torch.int32)
    try:
        from delta_encoding.encoding._escape_ext import decode_escape

        return decode_escape(flat_deltas.contiguous(), nnz)
    except (ImportError, OSError):
        running = flat_deltas.to(torch.int32)
        torch.cumsum(running, dim=0, out=running)
        return running[flat_deltas != 255].contiguous()


def _decode_escape_on_device(
    flat_deltas: torch.Tensor, nnz: int, device: torch.device
) -> torch.Tensor:
    """Escape decode with the cumsum/compact running on ``device``.

    Used by the fused-MoE fast path so multi-million-nnz entries decode on
    the destination GPU instead of the CPU fallback.

    The stream is processed in bounded slabs because the monolithic
    ``stream.to(int32)`` transient is four times the stream length and can
    exhaust device memory. Integer cumsum is associative, so slab-wise cumsum
    with a carried base is bitwise identical.
    """
    if device.type == "cpu":
        return _decode_escape(flat_deltas, nnz)
    if nnz == 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    slab = int(os.environ.get("SGLANG_SPARSE_DELTA_DECODE_SLAB_BYTES", str(1 << 26)))
    stream_cpu = flat_deltas.contiguous()
    total = stream_cpu.numel()
    if total <= slab:
        stream = stream_cpu.to(device=device)
        running = stream.to(torch.int32)
        torch.cumsum(running, dim=0, out=running)
        return running[stream != 255].contiguous()
    outs = []
    base = torch.zeros((), dtype=torch.int32, device=device)
    for start in range(0, total, slab):
        chunk = stream_cpu[start : start + slab].to(device=device)
        running = chunk.to(torch.int32)
        torch.cumsum(running, dim=0, out=running)
        running += base
        base = running[-1].clone()
        outs.append(running[chunk != 255])
        del running, chunk
    return torch.cat(outs).contiguous()


def _locator_map(locators: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for loc in locators:
        result.setdefault(str(loc["hf_name"]), []).append(loc)
    return result


def _param_for_locator(
    params: dict[str, nn.Parameter],
    loc: dict[str, Any],
) -> tuple[nn.Parameter, int]:
    ptr = int(loc["ptr"])
    nbytes = int(loc["nbytes"])
    for param in params.values():
        data = param.data
        start = int(data.data_ptr())
        end = start + data.numel() * data.element_size()
        if start <= ptr and ptr + nbytes <= end:
            return param, (ptr - start) // data.element_size()
    raise KeyError(f"No parameter memory covers sparse-delta locator ptr=0x{ptr:x}, nbytes={nbytes}")


def _select_fused_moe_indices(
    flat_indices: torch.Tensor,
    shape: tuple[int, ...],
    loc: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slab wrapper over :func:`_select_fused_moe_indices_block`.

    The block translation materializes several int64 temporaries the size of
    ``flat_indices`` and can exhaust device memory for a large per-tensor
    delta. Chunking preserves output order exactly: per-chunk masked
    destinations concatenate to the monolithic result.
    """
    n = int(flat_indices.numel())
    slab = int(os.environ.get("SGLANG_SPARSE_DELTA_SELECT_SLAB_ELEMS", str(4 << 20)))
    if slab <= 0 or n <= slab:
        return _select_fused_moe_indices_block(flat_indices, shape, loc)
    dests: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for start in range(0, n, slab):
        dest, mask = _select_fused_moe_indices_block(
            flat_indices[start : start + slab], shape, loc
        )
        if dest.numel():
            dests.append(dest)
        masks.append(mask)
    full_mask = torch.cat(masks)
    if not dests:
        return torch.empty(0, dtype=torch.int64), full_mask
    return torch.cat(dests), full_mask


def _select_fused_moe_indices_block(
    flat_indices: torch.Tensor,
    shape: tuple[int, ...],
    loc: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized index translation for a ``kind="fused_moe"`` locator.

    ``shape`` is the sender's full fused 3D shape ``[E_total, d1, d2]``. The
    locator describes this rank's slice: an expert window on dim 0
    (``expert_offset``/``num_local_experts``) plus contiguous bands on
    ``shard_dim`` (1 or 2) mapping full coordinates onto local coordinates
    (``bands`` entries are ``(full_start, local_start, length)`` — two bands
    for w13's gate/up halves, one for w2). Returns (local element indices
    relative to the locator's ``ptr``, value mask).
    """
    loc_shape = tuple(int(x) for x in loc["full_shape"])
    if loc_shape != tuple(shape):
        raise ValueError(
            f"Sparse delta shape mismatch for {loc['hf_name']}: packed={tuple(shape)}, locator={loc_shape}"
        )
    if len(shape) != 3:
        raise ValueError(f"fused_moe locator requires a 3D entry, got shape={tuple(shape)}")

    _e_dim, d1, d2 = (int(x) for x in shape)
    per_expert = d1 * d2
    idx = flat_indices.to(torch.int64)
    e = idx // per_expert
    rem = idx - e * per_expert
    r = rem // d2
    c = rem - r * d2

    e0 = int(loc["expert_offset"])
    e_local = int(loc["num_local_experts"])
    mask = (e >= e0) & (e < e0 + e_local)

    shard_dim = int(loc["shard_dim"])
    if shard_dim not in (1, 2):
        raise ValueError(f"fused_moe locator shard_dim must be 1 or 2, got {shard_dim}")
    coord = r if shard_dim == 1 else c
    local_coord = torch.full_like(coord, -1)
    for band in loc["bands"]:
        full_start, local_start, length = (int(x) for x in band)
        in_band = (coord >= full_start) & (coord < full_start + length)
        local_coord = torch.where(in_band, coord - (full_start - local_start), local_coord)
    mask &= local_coord >= 0
    if not bool(mask.any().item()):
        return torch.empty(0, dtype=torch.int64), mask

    le = (e - e0)[mask]
    if shard_dim == 1:
        lr = local_coord[mask]
        lc = c[mask]
    else:
        lr = r[mask]
        lc = local_coord[mask]
    dest = (
        le * int(loc["expert_stride_elems"])
        + lr * int(loc["row_stride_elems"])
        + lc * int(loc["col_stride_elems"])
    )
    return dest, mask


def _select_locator_indices(
    flat_indices: torch.Tensor,
    shape: tuple[int, ...],
    loc: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    loc_shape = tuple(int(x) for x in loc["full_shape"])
    if loc_shape != shape:
        raise ValueError(f"Sparse delta shape mismatch for {loc['hf_name']}: packed={shape}, locator={loc_shape}")

    slices = [(int(start), int(stop)) for start, stop in loc["slice"]]
    if len(slices) != len(shape):
        raise ValueError(f"Locator rank mismatch for {loc['hf_name']}: shape={shape}, slice={slices}")

    coords = _flat_to_coords(flat_indices, shape)
    mask = torch.ones(flat_indices.numel(), dtype=torch.bool)
    local_coords = []
    local_shape = []
    for coord, (start, stop) in zip(coords, slices):
        mask &= (coord >= start) & (coord < stop)
        local_coords.append(coord - start)
        local_shape.append(stop - start)
    if not bool(mask.any().item()):
        return torch.empty(0, dtype=torch.int32), mask
    selected_coords = [coord[mask] for coord in local_coords]
    return _coords_to_flat(selected_coords, tuple(local_shape)), mask


def _flat_to_coords(flat_indices: torch.Tensor, shape: tuple[int, ...]) -> list[torch.Tensor]:
    remaining = flat_indices.to(torch.int64)
    coords = [torch.empty_like(remaining) for _ in shape]
    for dim in range(len(shape) - 1, -1, -1):
        coords[dim] = remaining % shape[dim]
        remaining = remaining // shape[dim]
    return coords


def _coords_to_flat(coords: list[torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
    flat = torch.zeros_like(coords[0], dtype=torch.int64)
    stride = 1
    for coord, dim in zip(reversed(coords), reversed(shape)):
        flat += coord.to(torch.int64) * stride
        stride *= dim
    if prod(shape) > torch.iinfo(torch.int32).max:
        return flat
    return flat.to(torch.int32)


def _index_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
        selected = values.view(torch.uint8)[mask].contiguous()
        return selected.view(values.dtype)
    return values[mask]


def _scatter_flat(
    dst: torch.Tensor,
    flat_indices: torch.Tensor,
    values: torch.Tensor,
    *,
    use_pinned_scatter: bool,
) -> None:
    flat_dst = dst.view(-1)
    if flat_indices.numel() == 0:
        return

    if flat_dst.is_cuda and use_pinned_scatter:
        try:
            from delta_encoding.cuda._scatter_ext import scatter_pinned

            idx = flat_indices.to(torch.int32).contiguous()
            vals = values.to(dtype=flat_dst.dtype).contiguous()
            if not idx.is_pinned():
                idx = idx.pin_memory()
            if not vals.is_pinned():
                vals = vals.pin_memory()
            scatter_pinned(flat_dst, idx, vals, int(idx.numel()))
            return
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning(
                "Pinned sparse-delta scatter unavailable; falling back to torch indexing: %s",
                exc,
            )

    device = flat_dst.device
    idx = flat_indices.to(device=device, dtype=torch.long, non_blocking=True)
    vals = values.to(device=device, dtype=flat_dst.dtype, non_blocking=True)
    flat_dst[idx] = vals
