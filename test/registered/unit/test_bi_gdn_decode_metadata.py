import torch

from sglang.kernels.ops.attention.fla.bi_gdn_decode import BIGDNDecodeCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _cache(num_slots: int = 4) -> BIGDNDecodeCache:
    return BIGDNDecodeCache(
        num_slots=num_slots,
        qkv_dim=24,
        num_v_heads=2,
        head_k_dim=4,
        head_v_dim=4,
        device=torch.device("cpu"),
    )


def test_prepare_step_metadata_builds_one_chunk_per_request() -> None:
    cache = _cache()
    slots = [0, 1, 2, 3]
    slot_indices = torch.tensor(slots, dtype=torch.int32)
    # Scheduler lengths include the token currently consumed: write rows are
    # 0, 7, 63, and 0 respectively.
    seq_lens = torch.tensor([1, 8, 64, 65], dtype=torch.int32)

    metadata = cache.prepare_step_metadata(slots, slot_indices, seq_lens)

    assert metadata.slots == (0, 1, 2, 3)
    assert metadata.fill_before.tolist() == [0, 7, 63, 0]
    assert metadata.fill_after.tolist() == [1, 8, 64, 1]
    assert metadata.cu_seqlens.tolist() == [0, 1, 9, 73, 74]
    assert metadata.output_rows.tolist() == [0, 8, 72, 73]
    assert metadata.completed_mask.tolist() == [False, False, True, False]
    assert metadata.chunk_indices.tolist() == [[0, 0], [1, 0], [2, 0], [3, 0]]
    assert metadata.chunk_offsets.tolist() == [0, 1, 2, 3, 4]
    assert metadata.slot_indices is slot_indices
    assert metadata.slot_indices_long.dtype == torch.long


def test_scheduler_positions_cover_every_fill_and_boundary_crossing() -> None:
    cache = _cache(num_slots=64)
    slots = list(range(64))
    slot_indices = torch.arange(64, dtype=torch.int32)
    seq_lens = torch.arange(1, 65, dtype=torch.int32)

    metadata = cache.prepare_step_metadata(slots, slot_indices, seq_lens)

    assert metadata.fill_before.tolist() == list(range(64))
    assert metadata.fill_after.tolist() == list(range(1, 65))
    assert metadata.completed_mask.tolist() == [False] * 63 + [True]
    assert metadata.cu_seqlens[-1].item() == sum(range(1, 65))

    # Device-generated gather indices must reproduce a legacy per-slot concat.
    rows = torch.arange(64 * 64).view(64, 64)
    packed = rows.flatten().index_select(0, metadata.packed_row_indices)
    legacy = torch.cat([rows[slot, : slot + 1] for slot in slots])
    torch.testing.assert_close(packed, legacy, rtol=0, atol=0)


def test_bs1_static_metadata_uses_resident_slot_view() -> None:
    cache = _cache()
    slot_indices = torch.tensor([2], dtype=torch.int32)

    metadata = cache.prepare_step_metadata(
        [2], slot_indices, torch.tensor([37], dtype=torch.int32), static_bs1=True
    )

    assert metadata.static_bs1
    assert metadata.fill_after.tolist() == [37]
    assert metadata.cu_seqlens.tolist() == [0, 37]
    assert metadata.output_rows.tolist() == [36]
    assert metadata.packed_row_indices.numel() == 0


def test_static_bs1_request_falls_back_for_larger_batches() -> None:
    cache = _cache()
    slot_indices = torch.tensor([0, 2], dtype=torch.int32)

    metadata = cache.prepare_step_metadata(
        [0, 2],
        slot_indices,
        torch.tensor([1, 64], dtype=torch.int32),
        static_bs1=True,
    )

    assert not metadata.static_bs1
    assert metadata.packed_row_indices.numel() == 65


def test_graph_metadata_refresh_preserves_addresses_and_updates_fill() -> None:
    cache = _cache()
    slot_indices = torch.tensor([0], dtype=torch.int32)
    metadata = cache.prepare_step_metadata(
        [0], slot_indices, torch.tensor([1], dtype=torch.int32), static_bs1=True
    )
    addresses = (
        metadata.fill_before.data_ptr(),
        metadata.fill_after.data_ptr(),
        metadata.cu_seqlens.data_ptr(),
        metadata.output_rows.data_ptr(),
        metadata.completed_mask.data_ptr(),
    )

    cache.refresh_graph_metadata(metadata, torch.tensor([64], dtype=torch.int32))

    assert addresses == (
        metadata.fill_before.data_ptr(),
        metadata.fill_after.data_ptr(),
        metadata.cu_seqlens.data_ptr(),
        metadata.output_rows.data_ptr(),
        metadata.completed_mask.data_ptr(),
    )
    assert metadata.fill_before.tolist() == [63]
    assert metadata.fill_after.tolist() == [64]
    assert metadata.cu_seqlens.tolist() == [0, 64]
    assert metadata.output_rows.tolist() == [63]
    assert metadata.completed_mask.tolist() == [True]


def test_graph_metadata_refresh_supports_fixed_multi_row_bucket() -> None:
    cache = _cache(num_slots=8)
    slot_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    metadata = cache.prepare_step_metadata(
        [0, 1, 2],
        slot_indices,
        torch.tensor([2, 65, 66], dtype=torch.int32),
        static_bs1=True,
    )
    assert not metadata.static_bs1
    addresses = (
        metadata.fill_before.data_ptr(),
        metadata.fill_after.data_ptr(),
        metadata.cu_seqlens.data_ptr(),
        metadata.output_rows.data_ptr(),
        metadata.completed_mask.data_ptr(),
    )

    cache.refresh_graph_metadata(metadata, torch.tensor([65, 66, 3], dtype=torch.int32))

    assert addresses == (
        metadata.fill_before.data_ptr(),
        metadata.fill_after.data_ptr(),
        metadata.cu_seqlens.data_ptr(),
        metadata.output_rows.data_ptr(),
        metadata.completed_mask.data_ptr(),
    )
    assert metadata.fill_before.tolist() == [0, 1, 2]
    assert metadata.fill_after.tolist() == [1, 2, 3]
    assert metadata.cu_seqlens.tolist() == [0, 1, 3, 6]
    assert metadata.output_rows.tolist() == [0, 2, 5]
    assert metadata.completed_mask.tolist() == [False, False, False]


def test_multi_row_graph_capture_reserves_fixed_partial_chunk_rows() -> None:
    cache = _cache(num_slots=8)
    slot_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    metadata = cache.prepare_step_metadata(
        [0, 1, 2],
        slot_indices,
        torch.tensor([1, 1, 1], dtype=torch.int32),
        graph_capture=True,
    )

    assert metadata.fill_after.tolist() == [64, 64, 64]
    assert metadata.packed_row_indices.numel() == 3 * 64
    assert metadata.packed_row_indices[:8].tolist() == list(range(8))

    cache.refresh_graph_metadata(metadata, torch.tensor([2, 66, 3], dtype=torch.int32))

    assert metadata.fill_after.tolist() == [2, 2, 3]
    assert metadata.cu_seqlens.tolist() == [0, 2, 4, 7]
    assert metadata.output_rows.tolist() == [1, 3, 6]
    assert metadata.packed_row_indices.numel() == 3 * 64
    assert metadata.packed_row_indices[:7].tolist() == [
        0,
        1,
        64,
        65,
        128,
        129,
        130,
    ]
    assert sorted(metadata.packed_row_indices.tolist()) == list(range(3 * 64))


def test_graph_workspace_round_trip_preserves_live_slot_zero() -> None:
    cache = _cache()
    cache.boundary[0].fill_(11)
    cache.rows_qkv[0].fill_(12)
    cache.rows_g[0].fill_(13)
    cache.rows_beta[0].fill_(14)
    cache.boundary[2].fill_(3)
    cache.rows_qkv[2].fill_(4)
    cache.rows_g[2].fill_(5)
    cache.rows_beta[2].fill_(6)
    slot_zero = (
        cache.boundary[0].clone(),
        cache.rows_qkv[0].clone(),
        cache.rows_g[0].clone(),
        cache.rows_beta[0].clone(),
    )
    index = torch.tensor([2], dtype=torch.int32)

    cache.copy_to_graph_workspace(index)
    cache.boundary[0].add_(7)
    cache.rows_qkv[0].add_(7)
    cache.rows_g[0].add_(7)
    cache.rows_beta[0].add_(7)
    graph_output = (
        cache.boundary[0].clone(),
        cache.rows_qkv[0].clone(),
        cache.rows_g[0].clone(),
        cache.rows_beta[0].clone(),
    )
    cache.copy_from_graph_workspace(index)

    torch.testing.assert_close(cache.boundary[2], graph_output[0])
    torch.testing.assert_close(cache.rows_qkv[2], graph_output[1])
    torch.testing.assert_close(cache.rows_g[2], graph_output[2])
    torch.testing.assert_close(cache.rows_beta[2], graph_output[3])
    torch.testing.assert_close(cache.boundary[0], slot_zero[0])
    torch.testing.assert_close(cache.rows_qkv[0], slot_zero[1])
    torch.testing.assert_close(cache.rows_g[0], slot_zero[2])
    torch.testing.assert_close(cache.rows_beta[0], slot_zero[3])


def test_multi_row_graph_workspace_round_trip_preserves_idle_and_live_rows() -> None:
    cache = _cache(num_slots=8)
    cache.configure_graph_workspace(3)
    for slot, value in enumerate((3, 4, 5, 6, 7, 8, 9, 10)):
        cache.boundary[slot].fill_(value)
        cache.rows_qkv[slot].fill_(value + 1)
        cache.rows_g[slot].fill_(value + 2)
        cache.rows_beta[slot].fill_(value + 3)
    staging = (
        cache.boundary[:3].clone(),
        cache.rows_qkv[:3].clone(),
        cache.rows_g[:3].clone(),
        cache.rows_beta[:3].clone(),
    )
    state_indices = torch.tensor([5, 7, -1], dtype=torch.int32)

    cache.copy_to_graph_workspace(state_indices)
    cache.boundary[:3].add_(10)
    cache.rows_qkv[:3].add_(10)
    cache.rows_g[:3].add_(10)
    cache.rows_beta[:3].add_(10)
    graph_output = (
        cache.boundary[:3].clone(),
        cache.rows_qkv[:3].clone(),
        cache.rows_g[:3].clone(),
        cache.rows_beta[:3].clone(),
    )
    cache.copy_from_graph_workspace(state_indices)

    torch.testing.assert_close(cache.boundary[5], graph_output[0][0])
    torch.testing.assert_close(cache.boundary[7], graph_output[0][1])
    torch.testing.assert_close(cache.rows_qkv[5], graph_output[1][0])
    torch.testing.assert_close(cache.rows_qkv[7], graph_output[1][1])
    torch.testing.assert_close(cache.rows_g[5], graph_output[2][0])
    torch.testing.assert_close(cache.rows_g[7], graph_output[2][1])
    torch.testing.assert_close(cache.rows_beta[5], graph_output[3][0])
    torch.testing.assert_close(cache.rows_beta[7], graph_output[3][1])
    torch.testing.assert_close(cache.boundary[:3], staging[0])
    torch.testing.assert_close(cache.rows_qkv[:3], staging[1])
    torch.testing.assert_close(cache.rows_g[:3], staging[2])
    torch.testing.assert_close(cache.rows_beta[:3], staging[3])


def test_multi_row_graph_workspace_does_not_clobber_live_slot_zero_with_padding() -> (
    None
):
    cache = _cache(num_slots=8)
    cache.configure_graph_workspace(3)
    for slot, value in enumerate((3, 4, 5, 6, 7, 8, 9, 10)):
        cache.boundary[slot].fill_(value)
        cache.rows_qkv[slot].fill_(value + 1)
        cache.rows_g[slot].fill_(value + 2)
        cache.rows_beta[slot].fill_(value + 3)

    staging = (
        cache.boundary[1].clone(),
        cache.rows_qkv[1].clone(),
        cache.rows_g[1].clone(),
        cache.rows_beta[1].clone(),
    )
    state_indices = torch.tensor([0, -1, 5], dtype=torch.int32)
    cache.copy_to_graph_workspace(state_indices)
    cache.boundary[:3].add_(10)
    cache.rows_qkv[:3].add_(10)
    cache.rows_g[:3].add_(10)
    cache.rows_beta[:3].add_(10)
    graph_output = (
        cache.boundary[:3].clone(),
        cache.rows_qkv[:3].clone(),
        cache.rows_g[:3].clone(),
        cache.rows_beta[:3].clone(),
    )
    cache.copy_from_graph_workspace(state_indices)

    torch.testing.assert_close(cache.boundary[0], graph_output[0][0])
    torch.testing.assert_close(cache.rows_qkv[0], graph_output[1][0])
    torch.testing.assert_close(cache.rows_g[0], graph_output[2][0])
    torch.testing.assert_close(cache.rows_beta[0], graph_output[3][0])
    torch.testing.assert_close(cache.boundary[5], graph_output[0][2])
    torch.testing.assert_close(cache.rows_qkv[5], graph_output[1][2])
    torch.testing.assert_close(cache.rows_g[5], graph_output[2][2])
    torch.testing.assert_close(cache.rows_beta[5], graph_output[3][2])
    torch.testing.assert_close(cache.boundary[1], staging[0])
    torch.testing.assert_close(cache.rows_qkv[1], staging[1])
    torch.testing.assert_close(cache.rows_g[1], staging[2])
    torch.testing.assert_close(cache.rows_beta[1], staging[3])


def test_graph_workspace_round_trip_updates_live_slot_zero() -> None:
    cache = _cache()
    cache.boundary[0].fill_(3)
    cache.rows_qkv[0].fill_(4)
    cache.rows_g[0].fill_(5)
    cache.rows_beta[0].fill_(6)
    index = torch.tensor([0], dtype=torch.int32)

    cache.copy_to_graph_workspace(index)
    cache.boundary[0].add_(7)
    cache.rows_qkv[0].add_(7)
    cache.rows_g[0].add_(7)
    cache.rows_beta[0].add_(7)
    expected = (
        cache.boundary[0].clone(),
        cache.rows_qkv[0].clone(),
        cache.rows_g[0].clone(),
        cache.rows_beta[0].clone(),
    )
    cache.copy_from_graph_workspace(index)

    torch.testing.assert_close(cache.boundary[0], expected[0])
    torch.testing.assert_close(cache.rows_qkv[0], expected[1])
    torch.testing.assert_close(cache.rows_g[0], expected[2])
    torch.testing.assert_close(cache.rows_beta[0], expected[3])


def test_graph_workspace_idle_dp_row_preserves_live_slot_zero() -> None:
    cache = _cache(num_slots=1)
    cache.boundary[0].fill_(3)
    cache.rows_qkv[0].fill_(4)
    cache.rows_g[0].fill_(5)
    cache.rows_beta[0].fill_(6)
    expected = (
        cache.boundary[0].clone(),
        cache.rows_qkv[0].clone(),
        cache.rows_g[0].clone(),
        cache.rows_beta[0].clone(),
    )
    pad_index = torch.tensor([-1], dtype=torch.int32)

    cache.copy_to_graph_workspace(pad_index)
    cache.boundary[0].add_(7)
    cache.rows_qkv[0].add_(7)
    cache.rows_g[0].add_(7)
    cache.rows_beta[0].add_(7)
    cache.copy_from_graph_workspace(pad_index)

    torch.testing.assert_close(cache.boundary[0], expected[0])
    torch.testing.assert_close(cache.rows_qkv[0], expected[1])
    torch.testing.assert_close(cache.rows_g[0], expected[2])
    torch.testing.assert_close(cache.rows_beta[0], expected[3])
