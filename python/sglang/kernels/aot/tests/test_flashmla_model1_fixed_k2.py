import math
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
from sgl_kernel.flash_mla import FlashMLASchedMeta, flash_mla_with_kvcache

HEAD_DIM = 512
HEADS = 64
MAIN_TOPK = 128
EXTRA_TOPK = 1024
BYTES_PER_TOKEN = 584


def is_sm90_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


pytestmark = pytest.mark.skipif(
    not is_sm90_supported(), reason="SM90 required for MODEL1 FlashMLA decode"
)


@dataclass
class SparseCase:
    q: torch.Tensor
    main_cache: torch.Tensor
    main_indices: torch.Tensor
    main_lengths: torch.Tensor
    attn_sink: torch.Tensor
    extra_cache: Optional[torch.Tensor] = None
    extra_indices: Optional[torch.Tensor] = None
    extra_lengths: Optional[torch.Tensor] = None


def _make_model1_cache(num_tokens: int, page_size: int, seed: int) -> torch.Tensor:
    """Create the MODEL1 page layout: FP8 nope + BF16 rope, then UE8M0 scales."""
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    num_pages = math.ceil(num_tokens / page_size)

    data = torch.empty((num_pages, page_size, 576), dtype=torch.uint8, device="cuda")
    nope = (
        torch.randn(
            (num_pages, page_size, 448),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        .clamp_(-4, 4)
        .to(torch.float8_e4m3fn)
    )
    rope = torch.randn(
        (num_pages, page_size, 64),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    data[..., :448].copy_(nope.view(torch.uint8))
    data[..., 448:].copy_(rope.view(torch.uint8))

    packed = torch.empty(
        (num_pages, page_size * BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device="cuda",
    )
    packed[:, : page_size * 576].copy_(data.view(num_pages, -1))
    # 127 encodes a unit UE8M0 scale. The eighth byte is MODEL1's scale pad.
    packed[:, page_size * 576 :].view(num_pages, page_size, 8).fill_(127)
    return packed.view(num_pages, page_size, 1, BYTES_PER_TOKEN)


def _unpack_fixture_keys(cache: torch.Tensor) -> torch.Tensor:
    """Recover the BF16 keys represented by `_make_model1_cache`."""
    num_pages, page_size = cache.shape[:2]
    page_bytes = cache.view(num_pages, -1)
    payload = page_bytes[:, : page_size * 576].view(num_pages, page_size, 576)
    nope = payload[..., :448].contiguous().view(torch.float8_e4m3fn)
    rope = payload[..., 448:].contiguous().view(torch.bfloat16)
    # Fixture UE8M0 scales are all 127, i.e. exactly one.
    return torch.cat((nope.to(torch.bfloat16), rope), dim=-1).view(-1, HEAD_DIM)


def _make_case(
    *,
    main_lengths: list[int],
    extra_lengths: Optional[list[int]],
    extra_page_size: Optional[int],
    seed: int = 1234,
) -> SparseCase:
    batch = len(main_lengths)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn(
        (batch, 1, HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    main_cache = _make_model1_cache(MAIN_TOPK, page_size=128, seed=seed + 1)
    main_indices = (
        torch.arange(MAIN_TOPK, dtype=torch.int32, device="cuda")
        .view(1, 1, MAIN_TOPK)
        .expand(batch, -1, -1)
        .contiguous()
    )
    attn_sink = torch.linspace(-0.5, 0.5, HEADS, dtype=torch.float32, device="cuda")

    if extra_lengths is None:
        assert extra_page_size is None
        return SparseCase(
            q=q,
            main_cache=main_cache,
            main_indices=main_indices,
            main_lengths=torch.tensor(main_lengths, dtype=torch.int32, device="cuda"),
            attn_sink=attn_sink,
        )

    assert extra_page_size is not None
    assert len(extra_lengths) == batch
    extra_cache = _make_model1_cache(
        EXTRA_TOPK, page_size=extra_page_size, seed=seed + 2
    )
    extra_indices = (
        torch.arange(EXTRA_TOPK, dtype=torch.int32, device="cuda")
        .view(1, 1, EXTRA_TOPK)
        .expand(batch, -1, -1)
        .contiguous()
    )
    return SparseCase(
        q=q,
        main_cache=main_cache,
        main_indices=main_indices,
        main_lengths=torch.tensor(main_lengths, dtype=torch.int32, device="cuda"),
        attn_sink=attn_sink,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lengths=torch.tensor(extra_lengths, dtype=torch.int32, device="cuda"),
    )


def _run(
    case: SparseCase, metadata: Optional[FlashMLASchedMeta] = None
) -> tuple[torch.Tensor, torch.Tensor, FlashMLASchedMeta]:
    metadata = metadata or FlashMLASchedMeta()
    out, lse = flash_mla_with_kvcache(
        q=case.q,
        k_cache=case.main_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=HEAD_DIM,
        tile_scheduler_metadata=metadata,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
        is_fp8_kvcache=True,
        indices=case.main_indices,
        topk_length=case.main_lengths,
        attn_sink=case.attn_sink,
        extra_k_cache=case.extra_cache,
        extra_indices_in_kvcache=case.extra_indices,
        extra_topk_length=case.extra_lengths,
    )
    return out, lse, metadata


def _assert_fixed_k2_metadata(
    metadata: FlashMLASchedMeta, num_blocks: list[int]
) -> None:
    batch = len(num_blocks)
    assert metadata.tile_scheduler_metadata is not None
    assert metadata.num_splits is not None
    scheduler = metadata.tile_scheduler_metadata.cpu().view(batch, 2, 8)
    expected_splits = [0]

    for req_idx, blocks in enumerate(num_blocks):
        first = scheduler[req_idx, 0].tolist()
        second = scheduler[req_idx, 1].tolist()
        if blocks == 1:
            assert first == [req_idx, req_idx, 0, 1, 0, 0, 0, 0]
            # Decode checks begin_req_idx before consuming the remaining
            # scheduler fields, so they are outside the sentinel ABI.
            assert second[0] >= batch
            expected_splits.append(expected_splits[-1] + 1)
        else:
            midpoint = (blocks + 1) // 2
            assert first == [req_idx, req_idx, 0, midpoint, 0, 1, 1, 0]
            assert second == [
                req_idx,
                req_idx,
                midpoint,
                blocks,
                1,
                1,
                1,
                0,
            ]
            expected_splits.append(expected_splits[-1] + 2)

    assert metadata.num_splits.cpu().tolist() == expected_splits


def _reference(case: SparseCase) -> tuple[torch.Tensor, torch.Tensor]:
    main_keys = _unpack_fixture_keys(case.main_cache).float()
    extra_keys = (
        _unpack_fixture_keys(case.extra_cache).float()
        if case.extra_cache is not None
        else None
    )
    outputs = []
    lses = []
    for row in range(case.q.shape[0]):
        keys = main_keys[: int(case.main_lengths[row].item())]
        if extra_keys is not None:
            assert case.extra_lengths is not None
            keys = torch.cat(
                (keys, extra_keys[: int(case.extra_lengths[row].item())]), dim=0
            )
        scores = torch.einsum("hd,kd->hk", case.q[row, 0].float(), keys)
        scores *= HEAD_DIM**-0.5
        lses.append(torch.logsumexp(scores, dim=-1))
        scores_with_sink = torch.cat(
            (scores, case.attn_sink[:, None].to(scores.dtype)), dim=-1
        )
        probabilities = torch.softmax(scores_with_sink, dim=-1)[..., :-1]
        outputs.append(probabilities @ keys)
    return torch.stack(outputs).unsqueeze(1), torch.stack(lses).unsqueeze(-1)


@pytest.mark.parametrize(
    ("main_lengths", "extra_lengths", "extra_page_size", "num_blocks"),
    [
        ([1, 64, 65, 128], None, None, [1, 1, 2, 2]),
        (
            [64, 128, 128, 128, 128, 128],
            [0, 0, 64, 128, 512, 1024],
            64,
            [1, 2, 3, 4, 10, 18],
        ),
        (
            [64, 128, 128, 128, 128, 128],
            [0, 0, 64, 128, 512, 1024],
            2,
            [1, 2, 3, 4, 10, 18],
        ),
    ],
    ids=["c1", "c4", "c128"],
)
@torch.inference_mode()
def test_model1_fixed_k2_metadata(
    main_lengths: list[int],
    extra_lengths: Optional[list[int]],
    extra_page_size: Optional[int],
    num_blocks: list[int],
) -> None:
    case = _make_case(
        main_lengths=main_lengths,
        extra_lengths=extra_lengths,
        extra_page_size=extra_page_size,
    )
    out, lse, metadata = _run(case)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()
    _assert_fixed_k2_metadata(metadata, num_blocks)


@pytest.mark.parametrize("extra_page_size", [64, 2], ids=["c4", "c128"])
@torch.inference_mode()
def test_model1_fixed_k2_matches_independent_reference(
    extra_page_size: int,
) -> None:
    case = _make_case(
        main_lengths=[64, 128, 128, 128, 128, 128],
        extra_lengths=[0, 0, 64, 128, 512, 1024],
        extra_page_size=extra_page_size,
    )
    out, lse, _ = _run(case)
    expected_out, expected_lse = _reference(case)
    torch.testing.assert_close(out.float(), expected_out, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(lse.float(), expected_lse, atol=2e-2, rtol=2e-2)


def _lengths_for_blocks(num_blocks: int) -> tuple[int, int]:
    if num_blocks == 1:
        return 64, 0
    if num_blocks == 2:
        return 128, 0
    return 128, (num_blocks - 2) * 64


@pytest.mark.parametrize("target_blocks", [1, 2, 3, 4, 10, 18])
@torch.inference_mode()
def test_model1_batch_row_and_neighbor_invariance(target_blocks: int) -> None:
    target_main, target_extra = _lengths_for_blocks(target_blocks)
    single = _make_case(
        main_lengths=[target_main],
        extra_lengths=[target_extra],
        extra_page_size=64,
    )
    single_out, single_lse, _ = _run(single)

    block_pattern = [1, 2, 3, 4, 10, 18]
    batch_blocks = [block_pattern[i % len(block_pattern)] for i in range(68)]
    target_idx = 64
    batch_blocks[target_idx] = target_blocks
    lengths = [_lengths_for_blocks(blocks) for blocks in batch_blocks]
    batch = _make_case(
        main_lengths=[item[0] for item in lengths],
        extra_lengths=[item[1] for item in lengths],
        extra_page_size=64,
    )
    batch.q[target_idx].copy_(single.q[0])
    batch_out, batch_lse, _ = _run(batch)

    assert torch.equal(single_out[0], batch_out[target_idx])
    assert torch.equal(single_lse[0], batch_lse[target_idx])

    # Changing every neighboring request must not alter the target program.
    changed = _make_case(
        main_lengths=[64] * 68,
        extra_lengths=[0 if i % 2 else 1024 for i in range(68)],
        extra_page_size=64,
    )
    changed.q.copy_(batch.q)
    changed.main_lengths[target_idx] = target_main
    assert changed.extra_lengths is not None
    changed.extra_lengths[target_idx] = target_extra
    changed_out, changed_lse, _ = _run(changed)
    assert torch.equal(batch_out[target_idx], changed_out[target_idx])
    assert torch.equal(batch_lse[target_idx], changed_lse[target_idx])

    # A row permutation may move scheduler slots, but not row-local arithmetic.
    permutation = torch.randperm(68, device="cuda")
    permuted = SparseCase(
        q=batch.q[permutation].contiguous(),
        main_cache=batch.main_cache,
        main_indices=batch.main_indices[permutation].contiguous(),
        main_lengths=batch.main_lengths[permutation].contiguous(),
        attn_sink=batch.attn_sink,
        extra_cache=batch.extra_cache,
        extra_indices=batch.extra_indices[permutation].contiguous(),
        extra_lengths=batch.extra_lengths[permutation].contiguous(),
    )
    permuted_out, permuted_lse, _ = _run(permuted)
    inverse = torch.argsort(permutation)
    assert torch.equal(batch_out, permuted_out[inverse])
    assert torch.equal(batch_lse, permuted_lse[inverse])


@torch.inference_mode()
def test_model1_cuda_graph_replay_rebuilds_fixed_k2_metadata() -> None:
    case = _make_case(
        main_lengths=[64] * 68,
        extra_lengths=[0] * 68,
        extra_page_size=2,
        seed=9012,
    )

    # Compile and initialize CUDA-side state before capture, but capture a fresh
    # metadata object so its scheduler kernel is part of the graph.
    _run(case)
    torch.cuda.synchronize()
    metadata = FlashMLASchedMeta()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_lse, metadata = _run(case, metadata)

    replay_blocks = [1, 2, 3, 4, 10, 18]
    replay_lengths = [
        _lengths_for_blocks(replay_blocks[i % len(replay_blocks)]) for i in range(68)
    ]
    case.main_lengths.copy_(
        torch.tensor([item[0] for item in replay_lengths], device="cuda")
    )
    assert case.extra_lengths is not None
    case.extra_lengths.copy_(
        torch.tensor([item[1] for item in replay_lengths], device="cuda")
    )
    graph.replay()
    torch.cuda.synchronize()
    replay_out = graph_out.clone()
    replay_lse = graph_lse.clone()

    eager_out, eager_lse, _ = _run(case)
    assert torch.equal(replay_out, eager_out)
    assert torch.equal(replay_lse, eager_lse)
    _assert_fixed_k2_metadata(
        metadata, [replay_blocks[i % len(replay_blocks)] for i in range(68)]
    )
