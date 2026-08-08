from __future__ import annotations

import bisect
from typing import Any, Dict, List, Optional, Tuple


def p2p_locator_intervals(locators: List[Dict[str, Any]]) -> List[Tuple[int, int, str]]:
    intervals: List[Tuple[int, int, str]] = []
    for loc in locators:
        try:
            start = int(loc["ptr"])
            nbytes = int(loc["nbytes"])
        except (KeyError, TypeError, ValueError):
            continue
        if nbytes <= 0:
            continue
        intervals.append((start, start + nbytes, str(loc.get("hf_name", "?"))))
    return intervals


def p2p_format_locator_interval(interval: Tuple[int, int, str]) -> str:
    start, end, name = interval
    return f"{name} ptr=0x{start:x} nbytes={end - start}"


def p2p_regions_from_memory_snapshot(
    locators: List[Dict[str, Any]],
    snapshot: List[Dict[str, Any]],
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Select active allocator blocks that overlap P2P locators."""

    intervals = p2p_locator_intervals(locators)
    if not intervals:
        return [], []

    ordered = sorted(range(len(intervals)), key=lambda idx: intervals[idx][0])
    covered = [False] * len(intervals)
    active_regions: List[Tuple[int, int]] = []

    for segment in snapshot:
        cursor = int(segment.get("address", -1))
        for block in segment.get("blocks", []) or []:
            try:
                size = int(block.get("size", -1))
                address = int(block.get("address", cursor))
            except (TypeError, ValueError):
                continue
            state = block.get("state", "")
            if cursor >= 0 and size > 0:
                cursor = address + size
            if address < 0 or size <= 0 or state != "active_allocated":
                continue
            active_regions.append((address, address + size))

    active_regions = sorted(set(active_regions))
    selected_regions: List[Tuple[int, int]] = []
    candidate_pos = 0
    for region_start, region_end in active_regions:
        while (
            candidate_pos < len(ordered)
            and intervals[ordered[candidate_pos]][1] <= region_start
        ):
            candidate_pos += 1
        j = candidate_pos
        overlaps = False
        while j < len(ordered) and intervals[ordered[j]][0] < region_end:
            overlaps = True
            idx = ordered[j]
            start, end, _ = intervals[idx]
            if region_start <= start and end <= region_end:
                covered[idx] = True
            j += 1
        if overlaps:
            selected_regions.append((region_start, region_end))

    missing = [
        p2p_format_locator_interval(interval)
        for idx, interval in enumerate(intervals)
        if not covered[idx]
    ]
    return selected_regions, missing


def p2p_segment_regions_from_memory_snapshot(
    locators: List[Dict[str, Any]],
    snapshot: List[Dict[str, Any]],
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Select full CUDA allocator segments that contain P2P locators.

    Mooncake/RDMA registration can reject allocator sub-block addresses with
    ``Bad address``. Registering the original CUDA allocation segment gives
    Mooncake the allocation base while the sender still writes only locator
    sub-ranges via offsets from ``memory_handle``.
    """

    intervals = p2p_locator_intervals(locators)
    if not intervals:
        return [], []

    segment_regions: List[Tuple[int, int]] = []
    for segment in snapshot:
        try:
            address = int(segment.get("address", -1))
            total_size = int(segment.get("total_size", -1))
        except (TypeError, ValueError):
            continue
        if address < 0:
            continue
        if total_size <= 0:
            cursor = address
            for block in segment.get("blocks", []) or []:
                try:
                    block_address = int(block.get("address", cursor))
                    block_size = int(block.get("size", -1))
                except (TypeError, ValueError):
                    continue
                if block_size <= 0:
                    continue
                cursor = block_address + block_size
            total_size = cursor - address
        if total_size <= 0:
            continue
        segment_regions.append((address, address + total_size))

    selected_regions: List[Tuple[int, int]] = []
    for region_start, region_end in sorted(set(segment_regions)):
        if any(
            region_start <= start and end <= region_end for start, end, _ in intervals
        ):
            selected_regions.append((region_start, region_end))

    return selected_regions, p2p_missing_locators_for_regions(
        locators, selected_regions
    )


def p2p_capped_block_registration_regions(
    locators: List[Dict[str, Any]],
    snapshot: List[Dict[str, Any]],
    *,
    max_region_bytes: int,
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Coalesce the *mapped* allocator blocks that cover the weight locators
    into a minimal set of registration regions.

    This is the safe-and-fast receiver registration path. Two failure modes it
    is designed to avoid, both observed live at EP8 (see XORL-252):

    * ``segment``/``allocator`` mode registers a whole allocator segment
      ``[address, address + total_size)``. ``total_size`` is the segment's
      *reserved* extent, which for an expandable / partially-mapped segment is
      larger than the physically mapped range. Mooncake hands that straight to
      ``ibv_reg_mr``, which returns ``EFAULT`` ("Bad address") the moment the
      range steps past mapped memory -> ``ERR_CONTEXT`` (-202) -> the whole
      ``/prepare_weights_update`` fails.
    * ``exact``/``block`` mode registers one region per weight tensor (tens of
      thousands). Each region is mapped and registers fine, but the strict
      serial ``engine.register()`` loop (the only path that actually detects a
      per-region failure -- Mooncake's batch register swallows them) is so slow
      the trainer's ``/prepare`` HTTP call times out.

    This mode registers only ``active_allocated`` (i.e. physically mapped)
    blocks, and merges *physically contiguous* blocks within a single segment
    into runs of at most ``max_region_bytes``. Because a run never spans an
    unmapped gap it never trips ``EFAULT``; because contiguous weights collapse
    into a handful of runs the serial strict registration stays fast. A single
    block larger than ``max_region_bytes`` is emitted whole (an allocation is
    never split mid-block, so no locator is ever straddled).
    """

    intervals = p2p_locator_intervals(locators)
    if not intervals:
        return [], []

    cap = max(1, int(max_region_bytes))

    runs: List[Tuple[int, int]] = []
    for segment in snapshot:
        cursor_default = int(segment.get("address", -1))
        cursor = cursor_default
        run_start: Optional[int] = None
        run_end: Optional[int] = None
        for block in segment.get("blocks", []) or []:
            try:
                size = int(block.get("size", -1))
                address = int(block.get("address", cursor))
            except (TypeError, ValueError):
                continue
            state = block.get("state", "")
            if size > 0:
                cursor = address + size
            if size <= 0 or state != "active_allocated":
                continue
            block_end = address + size
            if (
                run_start is not None
                and address == run_end
                and block_end - run_start <= cap
            ):
                # Physically contiguous with the current run and still within
                # the size cap -> extend it.
                run_end = block_end
            else:
                if run_start is not None:
                    runs.append((run_start, run_end))
                run_start, run_end = address, block_end
        if run_start is not None:
            runs.append((run_start, run_end))

    runs = sorted(set(runs))

    # Keep only the runs that actually cover a weight locator; a run that
    # covers none is non-weight memory we have no reason to register.
    selected: List[Tuple[int, int]] = []
    ordered = sorted(range(len(intervals)), key=lambda idx: intervals[idx][0])
    candidate_pos = 0
    for run_start, run_end in runs:
        while (
            candidate_pos < len(ordered)
            and intervals[ordered[candidate_pos]][1] <= run_start
        ):
            candidate_pos += 1
        j = candidate_pos
        covers = False
        while j < len(ordered) and intervals[ordered[j]][0] < run_end:
            start, end, _ = intervals[ordered[j]]
            if run_start <= start and end <= run_end:
                covers = True
                break
            j += 1
        if covers:
            selected.append((run_start, run_end))

    return selected, p2p_missing_locators_for_regions(locators, selected)


def p2p_missing_locators_for_regions(
    locators: List[Dict[str, Any]],
    regions: List[Tuple[int, int]],
) -> List[str]:
    intervals = p2p_locator_intervals(locators)
    if not intervals:
        return []
    regions = sorted(regions)
    if not regions:
        return [p2p_format_locator_interval(interval) for interval in intervals]

    starts = [start for start, _ in regions]
    missing: List[str] = []
    for interval in intervals:
        start, end, _ = interval
        idx = bisect.bisect_right(starts, start) - 1
        covered = False
        while idx >= 0:
            if regions[idx][0] <= start and end <= regions[idx][1]:
                covered = True
                break
            idx -= 1
        if not covered:
            missing.append(p2p_format_locator_interval(interval))
    return missing


def p2p_locator_registration_regions(
    locators: List[Dict[str, Any]],
) -> List[Tuple[int, int]]:
    """Return non-overlapping registration regions from locator target ranges.

    Exact locator mode deliberately keeps adjacent ranges separate so the
    sender cannot coalesce across receiver memory registrations. Only true
    overlaps are merged because Mooncake rejects overlapping registrations.
    """

    intervals = sorted(
        {(start, end) for start, end, _ in p2p_locator_intervals(locators)}
    )
    merged: List[Tuple[int, int]] = []
    for start, end in intervals:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def annotate_p2p_locators_with_memory_handles(
    locators: List[Dict[str, Any]],
    regions: List[Tuple[int, int]],
) -> List[str]:
    regions = sorted(regions)
    starts = [start for start, _ in regions]
    missing: List[str] = []
    for loc in locators:
        loc.pop("memory_handle", None)
        loc.pop("memory_nbytes", None)
        try:
            ptr = int(loc["ptr"])
            nbytes = int(loc["nbytes"])
        except (KeyError, TypeError, ValueError):
            continue
        end = ptr + nbytes
        idx = bisect.bisect_right(starts, ptr) - 1
        while idx >= 0:
            base, region_end = regions[idx]
            if base <= ptr and end <= region_end:
                loc["memory_handle"] = base
                loc["memory_nbytes"] = region_end - base
                break
            idx -= 1
        else:
            missing.append(
                p2p_format_locator_interval((ptr, end, str(loc.get("hf_name", "?"))))
            )
    return missing


def p2p_qwen35_linear_attn_qkvz_locators(
    *,
    module_name: str,
    output_sizes: List[int],
    input_size: int,
    tp_rank: int,
    tp_size: int,
    base_ptr: int,
    itemsize: int,
    dtype: str,
    dp_rank: int,
    ep_rank: int,
) -> List[Dict[str, Any]]:
    """Emit Qwen3.5 GDN locators for SGLang's fused q/k/v/z receiver weight.

    XORL streams the Qwen3.5 checkpoint layout names:
    ``in_proj_qkv.weight`` and ``in_proj_z.weight``. SGLang stores those four
    logical shards in one ``in_proj_qkvz.weight`` tensor, with local q/k/v/z
    rows packed contiguously on each attention-TP rank. The q/k/v rows are
    therefore three receiver ranges keyed by the same fused HF source name.
    """

    if len(output_sizes) != 4:
        raise ValueError(
            f"expected q/k/v/z output sizes for {module_name!r}, got {output_sizes!r}"
        )
    if tp_size <= 0:
        raise ValueError(f"invalid tp_size={tp_size}")

    prefix = module_name.rsplit(".", 1)[0]
    q_full, k_full, v_full, z_full = [int(size) for size in output_sizes]
    for full_size in (q_full, k_full, v_full, z_full):
        if full_size % tp_size != 0:
            raise ValueError(
                f"{module_name!r} output size {full_size} is not divisible by tp_size={tp_size}"
            )

    q_local = q_full // tp_size
    k_local = k_full // tp_size
    v_local = v_full // tp_size
    z_local = z_full // tp_size
    qkv_full = q_full + k_full + v_full

    locators: List[Dict[str, Any]] = []
    local_row_offset = 0
    for source_offset, _full_size, local_size in (
        (0, q_full, q_local),
        (q_full, k_full, k_local),
        (q_full + k_full, v_full, v_local),
    ):
        slc = [
            [
                source_offset + tp_rank * local_size,
                source_offset + (tp_rank + 1) * local_size,
            ],
            [0, input_size],
        ]
        locators.append(
            {
                "hf_name": f"{prefix}.in_proj_qkv.weight",
                "tp_rank": tp_rank,
                "dp_rank": dp_rank,
                "ep_rank": ep_rank,
                "dtype": dtype,
                "full_shape": [qkv_full, input_size],
                "slice": slc,
                "ptr": base_ptr + local_row_offset * input_size * itemsize,
                "nbytes": local_size * input_size * itemsize,
                "_local_rows": local_size,
            }
        )
        local_row_offset += local_size

    slc = [[tp_rank * z_local, (tp_rank + 1) * z_local], [0, input_size]]
    locators.append(
        {
            "hf_name": f"{prefix}.in_proj_z.weight",
            "tp_rank": tp_rank,
            "dp_rank": dp_rank,
            "ep_rank": ep_rank,
            "dtype": dtype,
            "full_shape": [z_full, input_size],
            "slice": slc,
            "ptr": base_ptr + local_row_offset * input_size * itemsize,
            "nbytes": z_local * input_size * itemsize,
            "_local_rows": z_local,
        }
    )
    return locators


def p2p_qwen35_linear_attn_conv1d_locators(
    *,
    module_name: str,
    output_sizes: List[int],
    input_size: int,
    tp_rank: int,
    tp_size: int,
    base_ptr: int,
    itemsize: int,
    dtype: str,
    dp_rank: int,
    ep_rank: int,
) -> List[Dict[str, Any]]:
    """Emit Qwen3.5 GDN locators for the fused q/k/v conv1d receiver weight.

    SGLang stores each TP rank's local q/k/v conv rows packed contiguously,
    while the HF/XORL source tensor is packed as all-q, all-k, then all-v rows.
    A generic contiguous ColumnParallelLinear slice is therefore wrong for
    TP>1; it would copy the wrong global rows into every rank except rank 0.
    """

    if len(output_sizes) != 3:
        raise ValueError(
            f"expected q/k/v output sizes for {module_name!r}, got {output_sizes!r}"
        )
    if tp_size <= 0:
        raise ValueError(f"invalid tp_size={tp_size}")

    q_full, k_full, v_full = [int(size) for size in output_sizes]
    for full_size in (q_full, k_full, v_full):
        if full_size % tp_size != 0:
            raise ValueError(
                f"{module_name!r} output size {full_size} is not divisible by tp_size={tp_size}"
            )

    q_local = q_full // tp_size
    k_local = k_full // tp_size
    v_local = v_full // tp_size
    full_rows = q_full + k_full + v_full

    locators: List[Dict[str, Any]] = []
    local_row_offset = 0
    for source_offset, local_size in (
        (0, q_local),
        (q_full, k_local),
        (q_full + k_full, v_local),
    ):
        slc = [
            [
                source_offset + tp_rank * local_size,
                source_offset + (tp_rank + 1) * local_size,
            ],
            [0, input_size],
        ]
        locators.append(
            {
                "hf_name": f"{module_name}.weight",
                "tp_rank": tp_rank,
                "dp_rank": dp_rank,
                "ep_rank": ep_rank,
                "dtype": dtype,
                "full_shape": [full_rows, input_size],
                "slice": slc,
                "ptr": base_ptr + local_row_offset * input_size * itemsize,
                "nbytes": local_size * input_size * itemsize,
                "_local_rows": local_size,
            }
        )
        local_row_offset += local_size

    return locators


def p2p_qwen35_full_attention_hf_name(
    name: str,
    layers_block_type: Optional[List[str]] = None,
) -> str:
    """Restore HF ``self_attn`` names for flattened Qwen3.5 attention layers."""

    if layers_block_type is None:
        return name

    prefix = "model.layers."
    if not name.startswith(prefix):
        return name
    layer_id_text, sep, suffix = name[len(prefix) :].partition(".")
    if not sep or not layer_id_text.isdigit() or not suffix:
        return name

    if suffix.startswith(("self_attn.", "linear_attn.", "mlp.")):
        return name

    layer_id = int(layer_id_text)
    if layer_id >= len(layers_block_type) or layers_block_type[layer_id] != "attention":
        return name

    if not (
        suffix
        in {"qkv_proj", "q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"}
        or suffix.startswith(
            (
                "qkv_proj.",
                "q_proj.",
                "k_proj.",
                "v_proj.",
                "o_proj.",
                "q_norm.",
                "k_norm.",
            )
        )
    ):
        return name

    return f"{prefix}{layer_id}.self_attn.{suffix}"


def p2p_register_regions(
    engine: Any,
    regions: List[Tuple[int, int]],
    *,
    chunk_size: int = 4096,
    strict: bool = True,
    location: Optional[str] = None,
) -> Tuple[int, List[int]]:
    ptrs = [start for start, _ in regions]
    nbytes_list = [end - start for start, end in regions]
    registered_ptrs: List[int] = []

    if strict:
        for ptr, nbytes in zip(ptrs, nbytes_list):
            if location is None:
                ret = engine.register(ptr, nbytes)
            else:
                try:
                    ret = engine.register(ptr, nbytes, location=location)
                except TypeError:
                    ret = engine.register(ptr, nbytes)
            if ret is None:
                ret = 0
            if ret != 0:
                if registered_ptrs:
                    try:
                        engine.batch_deregister(registered_ptrs)
                    except Exception:
                        pass
                return int(ret), registered_ptrs
            registered_ptrs.append(ptr)
        return 0, registered_ptrs

    chunk_size = max(1, int(chunk_size))
    for offset in range(0, len(ptrs), chunk_size):
        chunk_ptrs = ptrs[offset : offset + chunk_size]
        chunk_nbytes = nbytes_list[offset : offset + chunk_size]
        if location is None:
            ret = engine.batch_register(chunk_ptrs, chunk_nbytes)
        else:
            try:
                ret = engine.batch_register(chunk_ptrs, chunk_nbytes, location=location)
            except TypeError:
                ret = engine.batch_register(chunk_ptrs, chunk_nbytes)
        if ret != 0:
            if registered_ptrs:
                try:
                    engine.batch_deregister(registered_ptrs)
                except Exception:
                    pass
            return int(ret), registered_ptrs
        registered_ptrs.extend(chunk_ptrs)
    return 0, registered_ptrs
