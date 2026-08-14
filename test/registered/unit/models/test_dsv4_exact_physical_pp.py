"""Physical-pipeline contracts for the exact DeepSeek-V4 model."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner_utils.buffers import (
    add_dsv4_exact_pp_proxy_buffers,
)
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4Model,
    pack_dsv4_exact_pp_proxy,
    unpack_dsv4_exact_pp_proxy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_fused_mhc_proxy_round_trips_every_live_operand_bitwise():
    rows, hc_mult, hidden_size = 3, 4, 5
    hidden = torch.arange(rows * hidden_size, dtype=torch.bfloat16).view(
        rows, hidden_size
    )
    residual = torch.arange(rows * hc_mult * hidden_size, dtype=torch.bfloat16).view(
        rows, hc_mult, hidden_size
    )
    post = torch.arange(rows * hc_mult, dtype=torch.float32).view(rows, hc_mult)
    comb = torch.arange(rows * hc_mult * hc_mult, dtype=torch.float32).view(
        rows, hc_mult, hc_mult
    )
    input_ids = torch.tensor([11, 12, 13])
    positions = torch.tensor([101, 102, 103])

    proxy = pack_dsv4_exact_pp_proxy(
        hidden,
        input_ids=input_ids,
        positions=positions,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        deferred_mhc=True,
        residual=residual,
        post=post,
        comb=comb,
    )
    assert proxy["hidden_states"].shape == (rows, hc_mult * hidden_size)
    assert torch.equal(proxy["hidden_states"][:, :hidden_size], hidden)
    assert torch.count_nonzero(proxy["hidden_states"][:, hidden_size:]) == 0

    actual = unpack_dsv4_exact_pp_proxy(
        proxy,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        deferred_mhc=True,
    )
    for got, expected in zip(
        actual, (hidden, residual, post, comb, input_ids, positions)
    ):
        assert torch.equal(got, expected)


def test_unfused_mhc_proxy_round_trips_completed_hyperconnection_image():
    rows, hc_mult, hidden_size = 2, 3, 7
    completed = torch.arange(rows * hc_mult * hidden_size, dtype=torch.bfloat16).view(
        rows, hc_mult, hidden_size
    )
    proxy = pack_dsv4_exact_pp_proxy(
        completed,
        input_ids=torch.tensor([4, 5]),
        positions=torch.tensor([9, 10]),
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        deferred_mhc=False,
    )
    hidden, residual, post, comb, input_ids, positions = unpack_dsv4_exact_pp_proxy(
        proxy,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        deferred_mhc=False,
    )
    assert torch.equal(hidden, completed)
    assert residual is post is comb is None
    assert torch.equal(input_ids, torch.tensor([4, 5]))
    assert torch.equal(positions, torch.tensor([9, 10]))


def test_exact_proxy_static_graph_buffers_cover_state_and_metadata():
    buffers = {"hidden_states": torch.zeros((8, 24), dtype=torch.bfloat16)}
    add_dsv4_exact_pp_proxy_buffers(
        buffers,
        max_num_token=8,
        hidden_size=6,
        hc_hidden_size=24,
        dtype=torch.bfloat16,
    )
    assert buffers["dsv4_mhc_residual"].shape == (8, 4, 6)
    assert buffers["dsv4_mhc_residual"].dtype is torch.bfloat16
    assert buffers["dsv4_mhc_post"].shape == (8, 4)
    assert buffers["dsv4_mhc_post"].dtype is torch.float32
    assert buffers["dsv4_mhc_comb"].shape == (8, 4, 4)
    assert buffers["dsv4_mhc_comb"].dtype is torch.float32
    assert buffers["dsv4_exact_input_ids"].dtype is torch.int64
    assert buffers["dsv4_exact_positions"].dtype is torch.int64


def test_exact_proxy_fails_closed_when_metadata_is_missing():
    proxy = PPProxyTensors(
        {"hidden_states": torch.zeros((2, 12), dtype=torch.bfloat16)}
    )
    with pytest.raises(KeyError, match="missing"):
        unpack_dsv4_exact_pp_proxy(
            proxy,
            hc_mult=3,
            hidden_size=4,
            deferred_mhc=True,
        )


class _PPGroup:
    def __init__(self, rank: int, world_size: int):
        self.rank_in_group = rank
        self.world_size = world_size
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == world_size - 1


class _FusedMHCLayer(nn.Module):
    """Small differentiable stand-in for DSV4's deferred fused mHC layer."""

    def __init__(self, layer_id: int, hc_mult: int):
        super().__init__()
        self.layer_id = layer_id
        self.hc_mult = hc_mult
        self.scale = nn.Parameter(torch.tensor(1.0 + layer_id / 8))
        self.hc_post_calls = 0
        self.last_input_ids = None
        self.last_positions = None

    def hc_post(self, hidden, residual, post, comb):
        self.hc_post_calls += 1
        return (
            post.unsqueeze(-1) * hidden.unsqueeze(1)
            + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
        ).type_as(hidden)

    def forward(
        self,
        *,
        positions,
        hidden_states,
        forward_batch,
        input_ids,
        input_ids_global,
        prev_residual,
        prev_post,
        prev_comb,
    ):
        del forward_batch, input_ids_global
        self.last_input_ids = input_ids.detach().clone()
        self.last_positions = positions.detach().clone()
        completed = (
            hidden_states
            if prev_residual is None
            else self.hc_post(hidden_states, prev_residual, prev_post, prev_comb)
        )
        hidden = completed.mean(dim=1) * self.scale + (self.layer_id + 1) / 16
        rows = hidden.shape[0]
        post = torch.full(
            (rows, self.hc_mult),
            0.25 + self.layer_id / 32,
            dtype=torch.float32,
            device=hidden.device,
        )
        comb = torch.eye(
            self.hc_mult, dtype=torch.float32, device=hidden.device
        ).expand(rows, -1, -1)
        return hidden, completed, post, comb


class _IdentityHeadNorm(nn.Module):
    def forward(self, hidden):
        return hidden


def test_model_constructor_owns_config_used_by_exact_pp_and_graph_forward():
    config = SimpleNamespace(
        _dsv4_flash_exact_mode=True,
        hidden_size=4,
        rms_norm_eps=1e-6,
        num_hidden_layers=3,
        hc_eps=1e-6,
        hc_mult=3,
    )
    middle_stage = _PPGroup(rank=1, world_size=3)
    with (
        patch("sglang.srt.models.deepseek_v4._is_cuda", False),
        patch("sglang.srt.models.deepseek_v4._is_hip", False),
        patch("sglang.srt.models.deepseek_v4._is_npu", False),
        patch("sglang.srt.models.deepseek_v4.get_pp_group", return_value=middle_stage),
        patch(
            "sglang.srt.models.deepseek_v4.make_layers",
            return_value=(nn.ModuleList(), 1, 2),
        ),
        patch(
            "sglang.srt.models.deepseek_v4.is_dsa_enable_prefill_cp",
            return_value=False,
        ),
        patch(
            "sglang.srt.models.deepseek_v4._is_fused_mhc_post_pre_enabled",
            return_value=False,
        ),
    ):
        model = DeepseekV4Model(config)

    assert model.config is config


def _make_exact_dsv4_stage(rank, world_size, layer_range, *, rows=3, hidden=4):
    model = DeepseekV4Model.__new__(DeepseekV4Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(_dsv4_flash_exact_mode=True)
    model.pp_group = _PPGroup(rank, world_size)
    model.hidden_size = hidden
    model.hc_mult = 3
    model.use_fused_mhc_post_pre = True
    model.dsa_enable_prefill_cp = False
    model.dspark_layers_to_capture = None
    model.start_layer, model.end_layer = layer_range
    model.layers = nn.ModuleList(
        [_FusedMHCLayer(layer_id, model.hc_mult) for layer_id in range(3)]
    )
    model.embed_tokens = nn.Embedding(32, hidden)
    with torch.no_grad():
        values = torch.arange(32 * hidden, dtype=torch.float32).view(32, hidden)
        model.embed_tokens.weight.copy_(values / 64)
    model.hc_head = lambda image, *_: image.sum(dim=1)
    model.hc_head_fn = torch.empty(0)
    model.hc_head_scale = torch.empty(0)
    model.hc_head_base = torch.empty(0)
    model.norm = _IdentityHeadNorm()
    model._can_run_tbo = lambda _forward_batch: False
    return model


def _forward_batch():
    return SimpleNamespace(can_run_tbo=False)


def test_fused_mhc_first_middle_last_flow_matches_uncut_model_and_backward():
    """A PP cut must transport, consume, and finally complete one mHC program."""

    input_ids = torch.tensor([2, 5, 7], dtype=torch.int64)
    positions = torch.tensor([11, 12, 13], dtype=torch.int64)
    owner = SimpleNamespace(dp_rank=2)
    with (
        patch("sglang.srt.models.deepseek_v4.dsa_use_prefill_cp", return_value=False),
        patch(
            "sglang.srt.models.deepseek_v4.check_cuda_graph_backend",
            return_value=True,
        ),
        patch(
            "sglang.srt.models.deepseek_v4.resolve_dsv4_owner_plane",
            return_value=owner,
        ),
    ):
        uncut = _make_exact_dsv4_stage(0, 1, (0, 3))
        uncut_out, uncut_pre_head = uncut(
            input_ids, positions, _forward_batch(), input_embeds=None
        )

        stages = [
            _make_exact_dsv4_stage(0, 3, (0, 1)),
            _make_exact_dsv4_stage(1, 3, (1, 2)),
            _make_exact_dsv4_stage(2, 3, (2, 3)),
        ]
        proxy = stages[0](input_ids, positions, _forward_batch(), None)
        assert isinstance(proxy, PPProxyTensors)
        # Later schedulers may hold stage-local placeholders. The proxy must
        # restore the logical owner's ids and positions before the layer runs.
        proxy = stages[1](
            torch.tensor([30, 30, 30]),
            torch.tensor([90, 90, 90]),
            _forward_batch(),
            None,
            proxy,
        )
        staged_out, staged_pre_head = stages[2](
            torch.tensor([31, 31, 31]),
            torch.tensor([91, 91, 91]),
            _forward_batch(),
            None,
            proxy,
        )

    assert torch.equal(staged_out, uncut_out)
    assert torch.equal(staged_pre_head, uncut_pre_head)
    assert staged_out.shape == (3, 4)
    assert staged_pre_head.shape == (3, 12)
    assert torch.equal(stages[1].layers[1].last_input_ids, input_ids)
    assert torch.equal(stages[1].layers[1].last_positions, positions)
    assert torch.equal(stages[2].layers[2].last_input_ids, input_ids)
    assert torch.equal(stages[2].layers[2].last_positions, positions)
    # Middle/last stages consume the preceding deferred post; the last stage
    # alone performs a second hc_post to complete its own deferred output.
    assert stages[0].layers[0].hc_post_calls == 0
    assert stages[1].layers[1].hc_post_calls == 1
    assert stages[2].layers[2].hc_post_calls == 2

    (uncut_out.sum() + uncut_pre_head.sum()).backward()
    (staged_out.sum() + staged_pre_head.sum()).backward()
    for layer_id, stage in enumerate(stages):
        assert torch.equal(
            stage.layers[layer_id].scale.grad,
            uncut.layers[layer_id].scale.grad,
        )
    assert torch.equal(
        stages[0].embed_tokens.weight.grad,
        uncut.embed_tokens.weight.grad,
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
