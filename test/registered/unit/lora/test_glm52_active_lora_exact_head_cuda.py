"""Real-CUDA gates for the exact GLM-5.2 active-LoRA sampler head."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd
from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd
from sglang.srt.batch_invariant_ops.bi_families_v2 import (
    head_v2_full_logits_with_lse,
)
from sglang.srt.layers import communicator
from sglang.srt.layers import vocab_parallel_embedding as vocab_embedding_module
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.layers.xorl_batch_invariant import xorl_bi_lm_head
from sglang.srt.lora import mem_pool as mem_pool_module
from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.lora.layers import ParallelLMHeadWithLoRA
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


_CUDA_BF16_AVAILABLE = bool(
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] >= 8
)

pytestmark = pytest.mark.skipif(
    not _CUDA_BF16_AVAILABLE,
    reason="the exact active-LoRA head requires a CUDA GPU with BF16 tensor cores",
)


def _rand_bf16(shape, generator: torch.Generator) -> torch.Tensor:
    return (
        torch.randn(
            shape,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        .to(torch.bfloat16)
        .contiguous()
    )


def _make_real_lora_head(
    weight: torch.Tensor,
    backend: TritonLoRABackend,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> ParallelLMHeadWithLoRA:
    vocab_size, hidden_size = weight.shape
    with torch.device(weight.device):
        base_head = ParallelLMHead(
            vocab_size,
            hidden_size,
            params_dtype=torch.bfloat16,
            padding_size=1,
            enable_tp=False,
        )
    base_head.weight.data.copy_(weight)
    head = ParallelLMHeadWithLoRA(base_head, backend)
    head.set_lora_info(lora_a, lora_b)
    return head


def _literal_v2_plus_lora(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    batch_info,
) -> torch.Tensor:
    base_logits, _ = head_v2_full_logits_with_lse(hidden_states, weight)
    rank_output = sgemm_lora_a_fwd(hidden_states, lora_a, batch_info)
    return sgemm_lora_b_fwd(
        rank_output,
        lora_b,
        batch_info,
        base_output=base_logits,
    )


def _assert_same_bytes(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_exact_active_lora_head_matches_literal_kernels_for_pruned_prefill_rows():
    """The real wrapper consumes lm_head-pruned metadata, not full prefill rows."""

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(1701)
    hidden_size = 64
    vocab_size = 512
    pruned_rows = 5

    weight = _rand_bf16((vocab_size, hidden_size), generator)
    hidden_states = _rand_bf16((pruned_rows, hidden_size), generator)
    lora_a = _rand_bf16((2, 1, hidden_size), generator)
    lora_b = _rand_bf16((2, vocab_size, 1), generator)

    backend = TritonLoRABackend(max_loras_per_batch=2, device=device)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        extend_seq_lens_cpu=[8],
        extend_seq_lens=torch.tensor([8], device=device, dtype=torch.int32),
        return_logprob=True,
        extend_logprob_start_lens_cpu=[3],
    )
    backend.prepare_lora_batch(
        forward_batch,
        weight_indices=[0],
        lora_ranks=[1, 0],
        scalings=[1.0, 0.0],
        use_cuda_graph=False,
    )
    backend._glm52_exact_batch_certified = True
    head = _make_real_lora_head(weight, backend, lora_a, lora_b)

    assert backend.batch_info.seg_lens.tolist() == [8]
    lm_head_info = backend.lm_head_batch_info
    assert lm_head_info is not None
    assert lm_head_info.use_cuda_graph is False
    assert lm_head_info.expected_tokens == pruned_rows
    assert lm_head_info.seg_lens.tolist() == [pruned_rows]

    actual = xorl_bi_lm_head(
        hidden_states,
        head,
        use_fp32_lm_head=False,
        family="v2",
    )
    expected = _literal_v2_plus_lora(
        hidden_states,
        head.weight,
        lora_a,
        lora_b,
        lm_head_info,
    )
    base_only, _ = head_v2_full_logits_with_lse(hidden_states, head.weight)
    torch.cuda.synchronize()

    _assert_same_bytes(actual, expected)
    assert torch.count_nonzero(actual != base_only).item() > 0


def test_decode_graph_captured_with_base_metadata_replays_active_rank_one_exactly():
    """A/B launches captured as rank-zero no-ops must remain live graph nodes.

    The graph owns 16 physical rows, while replay installs one logical active
    request. Changing A and then B after capture must change only that row and
    still match the direct v2 + literal Triton A/B program byte for byte.
    """

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(1702)
    hidden_size = 64
    vocab_size = 512
    graph_width = 16

    weight = _rand_bf16((vocab_size, hidden_size), generator)
    hidden_states = _rand_bf16((graph_width, hidden_size), generator)
    lora_a = torch.zeros((1, 1, hidden_size), device=device, dtype=torch.bfloat16)
    lora_b = torch.zeros((1, vocab_size, 1), device=device, dtype=torch.bfloat16)
    active_a = _rand_bf16(lora_a.shape, generator)
    active_b = _rand_bf16(lora_b.shape, generator)

    backend = TritonLoRABackend(max_loras_per_batch=1, device=device)
    backend.init_cuda_graph_batch_info(
        max_bs_in_cuda_graph=graph_width,
        num_tokens_per_req=1,
    )
    capture_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        batch_size=graph_width,
    )
    backend.prepare_lora_batch(
        capture_batch,
        weight_indices=[0] * graph_width,
        lora_ranks=[0],
        scalings=[0.0],
        use_cuda_graph=True,
    )
    backend._glm52_exact_batch_certified = True
    head = _make_real_lora_head(weight, backend, lora_a, lora_b)

    # Compile both real LoRA kernels while their device-side rank is zero.
    for _ in range(2):
        xorl_bi_lm_head(
            hidden_states,
            head,
            use_fp32_lm_head=False,
            family="v2",
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = xorl_bi_lm_head(
            hidden_states,
            head,
            use_fp32_lm_head=False,
            family="v2",
        )
    graph.replay()
    torch.cuda.synchronize()
    capture_output = captured_output.clone()
    capture_base, _ = head_v2_full_logits_with_lse(hidden_states, head.weight)
    torch.cuda.synchronize()
    _assert_same_bytes(capture_output, capture_base)

    # This is the production replay shape: one logical request in a width-16
    # graph bucket, with the static metadata buffers refreshed in place.
    lora_a.copy_(active_a)
    lora_b.copy_(active_b)
    replay_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
    )
    backend.prepare_lora_batch(
        replay_batch,
        weight_indices=[0],
        lora_ranks=[1],
        scalings=[1.0],
        use_cuda_graph=True,
    )
    backend._glm52_exact_batch_certified = True
    assert backend.batch_info.bs == 1
    assert backend.sgemm_batch_info.seg_lens.tolist() == [1]

    graph.replay()
    torch.cuda.synchronize()
    expected = _literal_v2_plus_lora(
        hidden_states,
        head.weight,
        lora_a,
        lora_b,
        backend.sgemm_batch_info,
    )
    torch.cuda.synchronize()
    _assert_same_bytes(captured_output, expected)
    assert not torch.equal(
        captured_output.view(torch.uint8), capture_output.view(torch.uint8)
    )
    _assert_same_bytes(captured_output[1:], capture_base[1:])

    # Replay-time A mutation proves the captured shrink node reads the live A
    # buffer. A byte-exact literal result rules out an accidental base-only path.
    previous = captured_output.clone()
    lora_a.mul_(-0.5).add_(0.125)
    graph.replay()
    torch.cuda.synchronize()
    expected = _literal_v2_plus_lora(
        hidden_states,
        head.weight,
        lora_a,
        lora_b,
        backend.sgemm_batch_info,
    )
    torch.cuda.synchronize()
    _assert_same_bytes(captured_output, expected)
    assert not torch.equal(
        captured_output.view(torch.uint8), previous.view(torch.uint8)
    )

    # Replay-time B mutation provides the corresponding proof for the expand
    # node and its in-place FP32 base-logit update.
    previous = captured_output.clone()
    lora_b.mul_(0.25).add_(0.0625)
    graph.replay()
    torch.cuda.synchronize()
    expected = _literal_v2_plus_lora(
        hidden_states,
        head.weight,
        lora_a,
        lora_b,
        backend.sgemm_batch_info,
    )
    torch.cuda.synchronize()
    _assert_same_bytes(captured_output, expected)
    assert not torch.equal(
        captured_output.view(torch.uint8), previous.view(torch.uint8)
    )


@pytest.mark.parametrize(
    ("tp_rank", "expected_vocab_range"),
    ((0, (0, 32)), (15, (480, 512))),
)
def test_lm_head_loader_replicates_a_and_loads_exact_b_shard_bytes(
    monkeypatch,
    tp_rank: int,
    expected_vocab_range: tuple[int, int],
):
    """The real CUDA pool keeps A whole and loads this TP rank's B bytes."""

    device = torch.device("cuda")
    hidden_size = 64
    vocab_size = 512
    tp_size = 16
    parallel = SimpleNamespace(
        tp_rank=tp_rank,
        tp_size=tp_size,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_tp_size=1,
        moe_tp_rank=0,
    )
    monkeypatch.setattr(vocab_embedding_module, "get_parallel", lambda: parallel)
    monkeypatch.setattr(mem_pool_module, "get_parallel", lambda: parallel)
    monkeypatch.setattr(
        communicator,
        "get_attn_tp_context",
        lambda: SimpleNamespace(allow_input_scattered=False),
    )

    base_config = SimpleNamespace(
        architectures=["TinyForCausalLM"],
        hidden_size=hidden_size,
        intermediate_size=128,
        num_attention_heads=1,
        num_hidden_layers=1,
        vocab_size=vocab_size,
    )
    lora_config = LoRAConfig.from_dict(
        {
            "lora_alpha": 1,
            "peft_type": "LORA",
            "r": 1,
            "target_modules": ["lm_head"],
        }
    )
    generator = torch.Generator(device=device).manual_seed(1703)
    exported_a = _rand_bf16((1, hidden_size), generator)
    exported_b = _rand_bf16((vocab_size, 1), generator)
    adapter = LoRAAdapter(
        "active-rank-one",
        lora_config,
        base_config,
        load_config=None,
        lora_backend=SimpleNamespace(),
    )
    adapter.initialize_weights_from_tensors(
        {
            "base_model.model.lm_head.lora_embedding_A": exported_a,
            "base_model.model.lm_head.lora_embedding_B": exported_b,
        }
    )
    assert adapter.scaling == 1.0

    with torch.device(device):
        base_head = ParallelLMHead(
            vocab_size,
            hidden_size,
            params_dtype=torch.bfloat16,
            padding_size=1,
            enable_tp=True,
        )
    backend = TritonLoRABackend(max_loras_per_batch=1, device=device)
    lora_head = ParallelLMHeadWithLoRA(base_head, backend)
    base_model = torch.nn.Module()
    base_model.config = base_config
    base_model.lm_head = base_head

    pool = LoRAMemoryPool(
        base_hf_config=base_config,
        max_loras_per_batch=1,
        dtype=torch.bfloat16,
        tp_size=tp_size,
        tp_rank=tp_rank,
        attn_tp_size=tp_size,
        max_lora_rank=1,
        target_modules={"lm_head"},
        base_model=base_model,
        eviction_policy="lru",
        lora_added_tokens_size=0,
    )
    pool.load_lora_weight_to_buffer(
        "active-rank-one",
        0,
        adapter,
        lora_modules=[{}],
        lora_embed_tokens_module=None,
        lora_lm_head_module=lora_head,
    )
    loaded_a = pool.lm_head_A_buffer["lm_head"][0]
    loaded_b = pool.lm_head_B_buffer["lm_head"][0]
    start, end = expected_vocab_range
    assert (
        base_head.shard_indices.org_vocab_start_index,
        base_head.shard_indices.org_vocab_end_index,
    ) == expected_vocab_range
    torch.cuda.synchronize()

    assert loaded_a.is_cuda and loaded_b.is_cuda
    assert loaded_a.is_contiguous() and loaded_b.is_contiguous()
    _assert_same_bytes(loaded_a, exported_a)
    _assert_same_bytes(loaded_b, exported_b[start:end])
