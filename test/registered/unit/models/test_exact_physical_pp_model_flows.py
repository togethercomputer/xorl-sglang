"""Model-body evidence for exact physical pipeline composition."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from sglang.srt.distributed.canonical_moe import SamplerParallelPlan
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.deepseek_v2 import DeepseekV2Model
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForCausalLM,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _PPGroup:
    def __init__(self, rank: int, world_size: int):
        self.rank_in_group = rank
        self.world_size = world_size
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == world_size - 1


class _ForwardMode:
    def is_idle(self):
        return False

    def is_extend(self, **_kwargs):
        return False


def _forward_batch():
    return SimpleNamespace(
        can_run_tbo=False,
        reuse_dsa_topk_indices=False,
        forward_mode=_ForwardMode(),
    )


class _ResidualNorm(nn.Module):
    def forward(self, hidden, residual=None):
        if residual is None:
            return hidden
        return hidden + residual, residual


class _QwenBodyLayer(nn.Module):
    def __init__(self, layer_id: int, *, moe: bool):
        super().__init__()
        self.layer_id = layer_id
        self.bias = (layer_id + 1) * (0.125 if moe else 0.0625)

    def forward(
        self,
        *,
        positions,
        hidden_states,
        residual,
        forward_batch,
        captured_last_layer_outputs,
    ):
        del positions, forward_batch, captured_last_layer_outputs
        residual = hidden_states if residual is None else residual + hidden_states
        hidden_states = hidden_states * (1 + (self.layer_id + 1) / 16) + self.bias
        return hidden_states, residual


def _make_qwen_stage(model_cls, rank, world_size, layer_range):
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.pp_group = _PPGroup(rank, world_size)
    model.hidden_size = 5
    model._start_layer, model._end_layer = layer_range
    is_moe = model_cls is Qwen3_5MoeForCausalLM
    model.layers = nn.ModuleList(
        [_QwenBodyLayer(i, moe=is_moe) for i in range(4)]
    )
    model.embed_tokens = nn.Identity()
    model.norm = _ResidualNorm()
    model.layers_to_capture = []
    return model


@pytest.mark.parametrize(
    "model_cls", [Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM]
)
def test_qwen_exact_dense_and_moe_body_match_uncut_across_physical_pp(model_cls):
    """Exercise the real Qwen body/proxy/terminal-norm control flow."""

    hidden = torch.arange(15, dtype=torch.float32).view(3, 5) / 8
    ids = torch.tensor([1, 2, 3])
    positions = torch.tensor([7, 8, 9])
    recorder = SimpleNamespace(with_current_layer=lambda _layer: nullcontext())
    with patch(
        "sglang.srt.models.qwen3_5.get_global_expert_distribution_recorder",
        return_value=recorder,
    ):
        uncut = _make_qwen_stage(model_cls, 0, 1, (0, 4))
        expected = uncut(
            ids, positions, _forward_batch(), input_embeds=hidden.clone()
        )

        stages = [
            _make_qwen_stage(model_cls, 0, 3, (0, 1)),
            _make_qwen_stage(model_cls, 1, 3, (1, 3)),
            _make_qwen_stage(model_cls, 2, 3, (3, 4)),
        ]
        proxy = stages[0](
            ids, positions, _forward_batch(), input_embeds=hidden.clone()
        )
        assert isinstance(proxy, PPProxyTensors)
        proxy = stages[1](ids, positions, _forward_batch(), pp_proxy_tensors=proxy)
        actual = stages[2](
            ids, positions, _forward_batch(), pp_proxy_tensors=proxy
        )

    assert torch.equal(actual, expected)


class _GlmBodyLayer(nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.prev_topk = None

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        residual,
        zero_allocator,
        gemm_output_zero_allocator,
        llama_4_scaling,
        *,
        prev_topk_indices,
        captured_last_layer_outputs,
        next_full_attention_layer_id,
    ):
        del (
            positions,
            forward_batch,
            zero_allocator,
            gemm_output_zero_allocator,
            llama_4_scaling,
            captured_last_layer_outputs,
            next_full_attention_layer_id,
        )
        self.prev_topk = (
            None if prev_topk_indices is None else prev_topk_indices.clone()
        )
        residual = hidden_states if residual is None else residual + hidden_states
        hidden_states = hidden_states * (1 + (self.layer_id + 1) / 32) + (
            self.layer_id + 1
        ) / 64
        topk = torch.full(
            (hidden_states.shape[0], 2),
            self.layer_id,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        return hidden_states, residual, topk


def _make_glm_stage(rank, world_size, layer_range):
    model = DeepseekV2Model.__new__(DeepseekV2Model)
    nn.Module.__init__(model)
    model.pp_group = _PPGroup(rank, world_size)
    model.config = SimpleNamespace(num_hidden_layers=3)
    model.use_dsa = True
    model.dsa_enable_prefill_cp = False
    model.mla_enable_prefill_cp = False
    model.cp_size = None
    model.first_k_dense_replace = 0
    model.start_layer, model.end_layer = layer_range
    model.layers = nn.ModuleList([_GlmBodyLayer(i) for i in range(3)])
    model.embed_tokens = nn.Identity()
    model.norm = _ResidualNorm()
    model.gemm_output_zero_allocator_size = 0
    model.layers_to_capture = []
    model.next_full_attention_layer_id = {0: 1, 1: 2}
    model.glm52_deferred_status_book = None
    model.llama_4_scaling_config = None
    model._dsa_forward_uses_topk = lambda: True
    plan = SamplerParallelPlan.glm52(
        contributors=4,
        pp_size=world_size,
        pp_rank=rank,
        physical_ranks=tuple(range(rank * 4, (rank + 1) * 4)),
        attention_dp_size=2,
        ep_size=4,
    )
    model.glm52_parallel_plan = replace(plan, stage_layer_range=layer_range)
    return model


def test_glm_exact_mixed_dp_cp_stage_plan_and_topk_proxy_match_uncut_body():
    """Bind a DP2/CP2 exact plan to each real GLM model-body PP stage."""

    hidden = torch.arange(20, dtype=torch.float32).view(4, 5) / 16
    ids = torch.tensor([3, 4, 5, 6])
    positions = torch.tensor([20, 21, 22, 23])
    patches = (
        patch(
            "sglang.srt.models.deepseek_v2.check_cuda_graph_backend",
            return_value=True,
        ),
        patch("sglang.srt.models.deepseek_v2.is_cp_v2_active", return_value=False),
        patch("sglang.srt.models.deepseek_v2.dsa_use_prefill_cp", return_value=False),
        patch("sglang.srt.models.deepseek_v2.mla_use_prefill_cp", return_value=False),
        patch("sglang.srt.models.deepseek_v2.dsa_layer_skips_topk", return_value=True),
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        uncut = _make_glm_stage(0, 1, (0, 3))
        expected = uncut(
            ids, positions, _forward_batch(), input_embeds=hidden.clone()
        )

        stages = [
            _make_glm_stage(0, 3, (0, 1)),
            _make_glm_stage(1, 3, (1, 2)),
            _make_glm_stage(2, 3, (2, 3)),
        ]
        proxy = stages[0](
            ids, positions, _forward_batch(), input_embeds=hidden.clone()
        )
        assert torch.equal(
            proxy["topk_indices"], torch.zeros((4, 2), dtype=torch.int32)
        )
        proxy = stages[1](ids, positions, _forward_batch(), pp_proxy_tensors=proxy)
        assert torch.equal(
            stages[1].layers[1].prev_topk,
            torch.zeros((4, 2), dtype=torch.int32),
        )
        actual = stages[2](
            ids, positions, _forward_batch(), pp_proxy_tensors=proxy
        )

    assert torch.equal(actual, expected)
    assert torch.equal(
        stages[2].layers[2].prev_topk,
        torch.ones((4, 2), dtype=torch.int32),
    )
    for rank, stage in enumerate(stages):
        plan = stage.glm52_parallel_plan
        assert plan.attention_dp_size == 2
        assert plan.attention_cp_size == 2
        assert plan.ep_size == plan.contributor_count == 4
        assert plan.pp_rank == rank
        assert plan.stage_layer_range == (rank, rank + 1)
        assert plan.physical_ranks == tuple(range(rank * 4, (rank + 1) * 4))


@pytest.mark.parametrize("with_lora", [False, True])
@pytest.mark.parametrize("stage", [0, 1, 2])
def test_eager_exact_hooks_are_safe_and_stage_local_with_lora_off_or_on(
    with_lora, stage
):
    """Every PP worker executes the eager hook, including LoRA-disabled ones."""

    runner = SimpleNamespace(
        server_args=SimpleNamespace(),
        hisparse_coordinator=None,
        pp_rank=stage,
    )
    if with_lora:
        runner.lora_manager = MagicMock()
    batch = SimpleNamespace(
        global_num_tokens_cpu=None,
        global_num_tokens_gpu=None,
        num_token_non_padded=None,
        prepare_attn_tp_scatter_input=MagicMock(),
    )

    ModelRunner._prepare_eager_forward_batch(runner, batch)

    batch.prepare_attn_tp_scatter_input.assert_called_once_with(runner)
    if with_lora:
        runner.lora_manager.prepare_dsv4_flash_exact_dp_lora_batch.assert_called_once_with(
            batch
        )
        runner.lora_manager.prepare_glm52_exact_dp_lora_batch.assert_called_once_with(
            batch
        )
    else:
        assert not hasattr(runner, "lora_manager")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
