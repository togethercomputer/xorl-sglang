"""Physical-pipeline contracts for the exact DeepSeek-V4 model."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    DSV4AttnMetadata,
    DSV4Metadata,
)
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
)
from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
from sglang.srt.layers.cp.utils import (
    cp_shard_model_inputs,
    cp_split_before_forward,
    prepare_cp_forward,
)
from sglang.srt.layers.dsv4_ownership import reconstruct_dsv4_dp_rows
from sglang.srt.layers.logical_row_ownership import LogicalRowOwnership
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.model_executor.runner_utils.buffers import (
    add_dsv4_exact_pp_proxy_buffers,
)
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4Model,
    MQALayer,
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


class _TupleIdentity(nn.Module):
    def forward(self, hidden):
        return hidden, None


class _FusedQContractLayer(_FusedMHCLayer):
    """Reach DSV4's real fused-Q call from a small model-stage fixture."""

    def __init__(self, layer_id: int, hc_mult: int, hidden_size: int):
        super().__init__(layer_id, hc_mult)
        self.attn = MQALayer.__new__(MQALayer)
        nn.Module.__init__(self.attn)
        self.attn.wq_b = _TupleIdentity()
        self.attn.n_local_heads = 1
        self.attn.head_dim = hidden_size
        self.attn.eps = 1e-6
        self.attn.freqs_cis = torch.empty(0)

    def forward(self, **kwargs):
        hidden_states = kwargs["hidden_states"]
        positions = kwargs["positions"]
        self.attn._compute_q_b(hidden_states.mean(dim=1), positions)
        return super().forward(**kwargs)


class _OwnerReconstructLayer(_FusedMHCLayer):
    """Exercise the exact owner-row seam from a CP-local model body."""

    def __init__(self, layer_id: int, hc_mult: int, strategy):
        super().__init__(layer_id, hc_mult)
        self.strategy = strategy
        self.reconstructed = None

    def forward(self, **kwargs):
        self.reconstructed = reconstruct_dsv4_dp_rows(
            kwargs["hidden_states"],
            kwargs["forward_batch"],
            LogicalRowOwnership(2, 4, 0, 0, 8),
            [6, 6],
            context_sharded=True,
            strategy=self.strategy,
        )
        return super().forward(**kwargs)


class _UnexpectedEmbedding(nn.Module):
    def forward(self, _input_ids):
        raise AssertionError("CP-local input_embeds must bypass full-id embedding")


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


def _forward_batch(positions):
    return SimpleNamespace(can_run_tbo=False, positions=positions)


def test_cp_v2_ragged_prefill_aligns_fused_q_rows_and_keeps_full_pp_metadata():
    """CP-local Q rows must coexist with full logical exact-PP metadata."""

    input_ids = torch.arange(10, dtype=torch.int64)
    positions = torch.arange(10, dtype=torch.int64)
    complete_embeds = torch.arange(40, dtype=torch.float32).view(10, 4)
    forward_batch = SimpleNamespace(
        can_run_tbo=False,
        forward_mode=ForwardMode.EXTEND,
        input_ids=input_ids,
        positions=positions,
        seq_lens_cpu=torch.tensor([10]),
        extend_seq_lens_cpu=[10],
        attn_cp_metadata=None,
        global_num_tokens_cpu=None,
        out_cache_loc=None,
    )
    strategy = InterleaveCPStrategy(cp_size=4)
    cp_parallel = SimpleNamespace(attn_cp_rank=2, attn_cp_size=4)

    with (
        patch("sglang.srt.layers.cp.utils.is_cp_v2_active", return_value=True),
        patch("sglang.srt.layers.cp.utils.get_cp_strategy", return_value=strategy),
        patch("sglang.srt.layers.cp.base.get_parallel", return_value=cp_parallel),
        patch(
            "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
            return_value=4,
        ),
    ):
        prepare_cp_forward(forward_batch)
        local_embeds, local_positions = cp_split_before_forward(
            complete_embeds, positions, forward_batch
        )

    metadata = forward_batch.attn_cp_metadata
    assert metadata.per_rank_logical_token == [3, 3, 2, 2]
    assert metadata.per_rank_actual_token == [4, 4, 4, 4]
    assert torch.equal(local_positions, torch.tensor([2, 6, 0, 0]))
    assert local_embeds.shape == (4, 4)

    model = _make_exact_dsv4_stage(0, 2, (0, 1))
    model.layers[0] = _FusedQContractLayer(0, model.hc_mult, model.hidden_size)
    model.embed_tokens = _UnexpectedEmbedding()
    fused_contract = {}

    def _fused_q_norm_rope(q, q_out, _eps, _freqs_cis, q_positions):
        fused_contract["q_rows"] = q.shape[0]
        fused_contract["q"] = q[:, 0, :].detach().clone()
        fused_contract["positions"] = q_positions.detach().clone()
        assert q.shape[0] == q_positions.shape[0]
        q_out.copy_(q)

    with (
        patch(
            "sglang.srt.models.deepseek_v4.fused_q_norm_rope",
            side_effect=_fused_q_norm_rope,
        ),
        patch("sglang.srt.models.deepseek_v4.dsa_use_prefill_cp", return_value=False),
        patch(
            "sglang.srt.models.deepseek_v4.check_cuda_graph_backend",
            return_value=True,
        ),
    ):
        proxy = model(
            input_ids,
            local_positions,
            forward_batch,
            input_embeds=local_embeds,
        )

    assert fused_contract["q_rows"] == 4
    assert torch.equal(fused_contract["q"], local_embeds)
    assert torch.equal(fused_contract["positions"], local_positions)
    proxy_hidden, _, _, _, proxy_ids, proxy_positions = unpack_dsv4_exact_pp_proxy(
        proxy,
        hc_mult=model.hc_mult,
        hidden_size=model.hidden_size,
        deferred_mhc=True,
    )
    assert proxy_hidden.shape[0] == 4
    assert torch.equal(proxy_ids, input_ids)
    assert torch.equal(proxy_positions, positions)


def test_cp_v2_keeps_dp_max_padding_in_exact_owner_block():
    """A four-row DP batch padded to six must reconstruct all six owner rows."""

    input_ids = torch.arange(6, dtype=torch.int64)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=input_ids,
        positions=torch.arange(6, dtype=torch.int64),
        seq_lens_cpu=torch.tensor([4]),
        extend_seq_lens_cpu=[4],
        attn_cp_metadata=None,
        global_num_tokens_cpu=[6, 6],
        out_cache_loc=torch.arange(6, dtype=torch.int64),
    )
    strategy = InterleaveCPStrategy(cp_size=4)
    parallel = SimpleNamespace(
        attn_cp_rank=0,
        attn_cp_size=4,
        attn_cp_group=object(),
    )
    hf_config = SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        index_topk=2048,
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(hf_config=hf_config)
    )

    def all_gather(output, _local):
        output.copy_(
            torch.tensor(
                [
                    0,
                    4,
                    0,
                    0,
                    1,
                    5,
                    0,
                    0,
                    2,
                    0,
                    0,
                    0,
                    3,
                    0,
                    0,
                    0,
                ],
                dtype=input_ids.dtype,
            )
        )

    with (
        patch("sglang.srt.layers.cp.utils.is_cp_v2_active", return_value=True),
        patch("sglang.srt.layers.cp.utils.get_cp_strategy", return_value=strategy),
        patch("sglang.srt.layers.cp.base.get_parallel", return_value=parallel),
        patch("sglang.srt.layers.cp.interleave.get_parallel", return_value=parallel),
        patch(
            "sglang.srt.layers.attention.dsa.utils.envs.SGLANG_ENABLE_CP_V2.get",
            return_value=True,
        ),
        patch(
            "sglang.srt.layers.attention.dsa.utils.get_parallel",
            return_value=parallel,
        ),
        patch(
            "sglang.srt.layers.attention.dsa.utils.get_server_args",
            return_value=server_args,
        ),
        patch(
            "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
            return_value=4,
        ),
        patch("sglang.srt.layers.dp_attention.set_local_dp_buffer_len"),
        patch(
            "sglang.srt.layers.cp.interleave.attn_cp_all_gather_into_tensor",
            side_effect=all_gather,
        ),
        patch(
            "sglang.srt.layers.cp.interleave.is_allocation_symmetric",
            return_value=False,
        ),
        patch(
            "sglang.srt.layers.cp.interleave.use_symmetric_memory",
            return_value=torch.no_grad(),
        ),
    ):
        prepare_cp_forward(forward_batch)
        local_rows = strategy.shard_hidden_states(input_ids, forward_batch)
        context_sharded = dsa_use_prefill_cp(
            forward_batch,
            dsa_enable_prefill_cp=is_dsa_enable_prefill_cp(),
        )
        reconstructed = reconstruct_dsv4_dp_rows(
            local_rows,
            forward_batch,
            LogicalRowOwnership(2, 4, 0, 0, 8),
            [6, 6],
            context_sharded=context_sharded,
            strategy=strategy,
        )

    assert forward_batch.attn_cp_metadata.total_seq_lens == 6
    assert forward_batch.attn_cp_metadata.per_rank_logical_token == [2, 2, 1, 1]
    assert local_rows.shape == (4,)
    assert context_sharded
    assert forward_batch.out_cache_loc.shape[0] == 6
    assert torch.equal(reconstructed, input_ids)


@pytest.mark.parametrize(
    ("cp_rank", "expected_positions", "logical_rows"),
    ((0, [0, 4, 0, 0], 2), (2, [2, 0, 0, 0], 1)),
)
def test_cp_v2_ragged_metadata_reindex_keeps_global_cache_rows(
    cp_rank, expected_positions, logical_rows
):
    input_ids = torch.arange(6, dtype=torch.int64)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=input_ids,
        positions=torch.arange(6, dtype=torch.int64),
        seq_lens_cpu=torch.tensor([4]),
        extend_seq_lens_cpu=[4],
        attn_cp_metadata=None,
        global_num_tokens_cpu=None,
        out_cache_loc=None,
    )
    strategy = InterleaveCPStrategy(cp_size=4)
    parallel = SimpleNamespace(attn_cp_rank=cp_rank, attn_cp_size=4)
    core = object.__new__(DSV4AttnMetadata)
    length_fields = {
        "seq_lens_casual",
        "swa_topk_lengths",
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c128_topk_lengths_clamp1",
    }
    for field_name in core._CP_REINDEX_FIELDS:
        values = torch.arange(1, 7) if field_name in length_fields else torch.arange(6)
        setattr(core, field_name, values)
    global_fields = {}
    for offset, field_name in enumerate(core._CP_GLOBAL_FIELDS):
        value = torch.arange(6) + 100 * (offset + 1)
        setattr(core, field_name, value)
        global_fields[field_name] = value

    with (
        patch("sglang.srt.layers.cp.utils.is_cp_v2_active", return_value=True),
        patch("sglang.srt.layers.cp.utils.get_cp_strategy", return_value=strategy),
        patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=strategy),
        patch("sglang.srt.layers.cp.base.get_parallel", return_value=parallel),
        patch(
            "sglang.srt.layers.attention.deepseek_v4_backend.get_parallel",
            return_value=parallel,
        ),
        patch(
            "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
            return_value=4,
        ),
    ):
        prepare_cp_forward(forward_batch)
        core.apply_cp_v2_reindex(forward_batch)

    assert core.positions_casual.tolist() == expected_positions
    for field_name in (
        "positions_casual",
        "page_table",
        "swa_page_indices",
        "c128_page_indices",
    ):
        assert getattr(core, field_name)[logical_rows:].count_nonzero().item() == 0
    assert core.seq_lens_casual.shape == (4,)
    assert core.seq_lens_casual[logical_rows:].tolist() == [1] * (4 - logical_rows)
    assert core.swa_topk_lengths[logical_rows:].tolist() == [1] * (4 - logical_rows)
    assert core.c4_topk_lengths_raw[logical_rows:].tolist() == [0] * (4 - logical_rows)
    assert core.c4_topk_lengths_clamp1[logical_rows:].tolist() == [1] * (
        4 - logical_rows
    )
    assert core.c128_topk_lengths_clamp1[logical_rows:].tolist() == [1] * (
        4 - logical_rows
    )
    for field_name, value in global_fields.items():
        assert getattr(core, field_name) is value
        assert value.shape == (6,)


def test_cp_v2_metadata_finalization_rebuilds_flash_and_indexer_plans():
    calls = []
    forward_batch = SimpleNamespace()
    core = SimpleNamespace(
        apply_cp_v2_reindex=lambda batch: calls.append(("reindex", batch)),
        init_flashmla_related=lambda **kwargs: calls.append(("flash", kwargs)),
    )
    original_indexer = object()
    rebuilt_indexer = object()
    metadata = DSV4Metadata(core, original_indexer)
    backend = object.__new__(DeepseekV4AttnBackend)

    def rebuild(received_core, *, use_prefill_cuda_graph):
        calls.append(("indexer", received_core, use_prefill_cuda_graph))
        return rebuilt_indexer

    backend.init_forward_metadata_indexer = rebuild
    backend._finalize_cp_v2_prefill_metadata(
        metadata,
        forward_batch,
        use_prefill_cuda_graph=True,
    )

    assert calls == [
        ("reindex", forward_batch),
        ("flash", {"is_prefill": True}),
        ("indexer", core, True),
    ]
    assert metadata.indexer_metadata is rebuilt_indexer


def test_cp_v2_eager_extend_owns_shard_gather_and_exact_owner_handoff():
    """CP-v2 must not re-enter DSV4's legacy model-side CP-v1 boundary."""

    input_ids = torch.arange(6, dtype=torch.int64)
    positions = torch.arange(6, dtype=torch.int64)
    complete_embeds = torch.arange(24, dtype=torch.float32).view(6, 4)
    out_cache_loc = torch.arange(1, 7, dtype=torch.int64)
    forward_batch = SimpleNamespace(
        can_run_tbo=False,
        forward_mode=ForwardMode.EXTEND,
        input_ids=input_ids,
        positions=positions,
        seq_lens_cpu=torch.tensor([4]),
        extend_seq_lens_cpu=[4],
        attn_cp_metadata=None,
        global_num_tokens_cpu=[6, 6],
        out_cache_loc=out_cache_loc,
        spec_info=None,
    )
    strategy = InterleaveCPStrategy(cp_size=4)
    cp_parallel = SimpleNamespace(
        attn_cp_rank=0,
        attn_cp_size=4,
        attn_cp_group=object(),
    )

    body = _make_exact_dsv4_stage(0, 1, (0, 1))
    owner_layer = _OwnerReconstructLayer(0, body.hc_mult, strategy)
    body.layers[0] = owner_layer
    body.embed_tokens = _UnexpectedEmbedding()
    body.dsa_enable_prefill_cp = True

    logits_call = {}

    def logits_processor(
        logits_input_ids,
        hidden_states,
        _lm_head,
        logits_forward_batch,
        *,
        aux_hidden_states,
        hidden_states_before_norm,
    ):
        logits_call.update(
            input_ids=logits_input_ids.detach().clone(),
            hidden_states=hidden_states.detach().clone(),
            forward_batch=logits_forward_batch,
            aux_hidden_states=aux_hidden_states,
            hidden_states_before_norm=hidden_states_before_norm.detach().clone(),
        )
        return hidden_states

    wrapper = SimpleNamespace(
        model=body,
        pp_group=body.pp_group,
        capture_aux_hidden_states=False,
        logits_processor=logits_processor,
        lm_head=object(),
    )
    eager = EagerRunner.__new__(EagerRunner)
    eager.model_runner = SimpleNamespace(
        model=wrapper,
        server_args=SimpleNamespace(enable_lora=False),
    )

    gather_calls = []

    def gather_hidden_states(local_rows, _forward_batch, _stream=None):
        gather_calls.append(tuple(local_rows.shape))
        rows_shape = (6, *local_rows.shape[1:])
        return torch.arange(
            torch.tensor(rows_shape).prod().item(), dtype=local_rows.dtype
        ).view(rows_shape)

    owner = SimpleNamespace(dp_rank=0)
    forbidden_cp_v1 = {
        "cp_split_and_rebuild_data": patch(
            "sglang.srt.models.deepseek_v4.cp_split_and_rebuild_data",
            side_effect=AssertionError("CP-v1 data split ran under CP-v2"),
        ),
        "cp_split_and_rebuild_position": patch(
            "sglang.srt.models.deepseek_v4.cp_split_and_rebuild_position",
            side_effect=AssertionError("CP-v1 position split ran under CP-v2"),
        ),
        "cp_round_robin_input_ids": patch(
            "sglang.srt.models.deepseek_v4.cp_round_robin_input_ids",
            side_effect=AssertionError("CP-v1 input-id split ran under CP-v2"),
        ),
        "cp_all_gather_rerange_output": patch(
            "sglang.srt.models.deepseek_v4.cp_all_gather_rerange_output",
            side_effect=AssertionError("CP-v1 terminal gather ran under CP-v2"),
        ),
    }

    with (
        patch("sglang.srt.layers.cp.utils.is_cp_v2_active", return_value=True),
        patch("sglang.srt.layers.cp.utils.get_cp_strategy", return_value=strategy),
        patch("sglang.srt.layers.cp.base.get_parallel", return_value=cp_parallel),
        patch(
            "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
            return_value=4,
        ),
        patch("sglang.srt.layers.dp_attention.set_local_dp_buffer_len"),
        patch.object(
            strategy, "gather_hidden_states", side_effect=gather_hidden_states
        ),
        patch("sglang.srt.models.deepseek_v4.is_cp_v2_active", return_value=True),
        patch("sglang.srt.models.deepseek_v4.dsa_use_prefill_cp", return_value=True),
        patch(
            "sglang.srt.models.deepseek_v4.check_cuda_graph_backend",
            return_value=True,
        ),
        patch(
            "sglang.srt.models.deepseek_v4.resolve_dsv4_owner_plane",
            return_value=owner,
        ),
        forbidden_cp_v1["cp_split_and_rebuild_data"],
        forbidden_cp_v1["cp_split_and_rebuild_position"],
        forbidden_cp_v1["cp_round_robin_input_ids"],
        forbidden_cp_v1["cp_all_gather_rerange_output"],
        patch(
            "sglang.srt.model_executor.runner.eager_runner.torch.cuda.current_stream",
            return_value=None,
        ),
    ):
        prepare_cp_forward(forward_batch)
        result = eager._execute_extend_cp_v2(
            forward_batch, {"input_embeds": complete_embeds}
        )

    metadata = forward_batch.attn_cp_metadata
    assert metadata.per_rank_logical_token == [2, 2, 1, 1]
    assert metadata.per_rank_actual_token == [4, 4, 4, 4]
    assert gather_calls == [(4, 3, 4), (4, 4), (4, 12)]
    assert owner_layer.reconstructed.shape == (6, 3, 4)
    assert result.shape == (6, 4)
    assert logits_call["hidden_states"].shape == (6, 4)
    assert logits_call["hidden_states_before_norm"].shape == (6, 12)
    assert torch.equal(logits_call["input_ids"], input_ids)
    assert logits_call["aux_hidden_states"] is None
    assert logits_call["forward_batch"] is forward_batch
    assert forward_batch.dsv4_exact_logits_rows_reconstructed
    assert forward_batch.dsv4_exact_logits_owner_rows == 6
    assert forward_batch.dsv4_exact_logits_dp_rank == 0
    assert torch.equal(forward_batch.out_cache_loc, out_cache_loc)
    assert not hasattr(forward_batch, "_cp_v2_out_cache_loc_is_local")


def test_cp_v2_exact_logits_reject_local_rows_and_accept_gathered_owner_rows():
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens=torch.tensor([6], dtype=torch.int64),
        extend_seq_lens_cpu=[6],
        extend_return_logprob=False,
        global_num_tokens_for_logprob_cpu=[1, 1],
        dsv4_exact_logits_rows_reconstructed=True,
        dsv4_exact_logits_owner_rows=6,
        dsv4_exact_logits_dp_rank=0,
    )

    with pytest.raises(RuntimeError, match="complete reconstructed DP-owner rows"):
        LogitsProcessor._get_pruned_states(
            None, torch.zeros((4, 4)), None, None, metadata
        )

    gathered = torch.arange(24, dtype=torch.float32).view(6, 4)
    pruned = LogitsProcessor._get_pruned_states(None, gathered, None, None, metadata)[0]
    assert torch.equal(pruned, gathered[-1:])


def test_cp_v2_eager_preserves_tensor_body_hidden_none_convention():
    class _HiddenNoneBody(nn.Module):
        def forward(
            self,
            _input_ids,
            _positions,
            _forward_batch,
            *,
            input_embeds,
        ):
            return input_embeds, None

    input_ids = torch.arange(6, dtype=torch.int64)
    positions = torch.arange(6, dtype=torch.int64)
    complete_embeds = torch.arange(24, dtype=torch.float32).view(6, 4)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=input_ids,
        positions=positions,
        seq_lens_cpu=torch.tensor([4]),
        extend_seq_lens_cpu=[4],
        attn_cp_metadata=None,
        global_num_tokens_cpu=None,
        out_cache_loc=None,
        spec_info=None,
    )
    strategy = InterleaveCPStrategy(cp_size=4)
    parallel = SimpleNamespace(attn_cp_rank=0, attn_cp_size=4)
    logits_call = {}

    def logits_processor(
        _input_ids,
        hidden_states,
        _lm_head,
        _forward_batch,
        *,
        aux_hidden_states,
        hidden_states_before_norm,
    ):
        logits_call["aux"] = aux_hidden_states
        logits_call["before_norm"] = hidden_states_before_norm
        return hidden_states

    wrapper = SimpleNamespace(
        model=_HiddenNoneBody(),
        pp_group=_PPGroup(0, 1),
        capture_aux_hidden_states=False,
        logits_processor=logits_processor,
        lm_head=object(),
    )
    eager = EagerRunner.__new__(EagerRunner)
    eager.model_runner = SimpleNamespace(
        model=wrapper,
        server_args=SimpleNamespace(enable_lora=False),
    )

    with (
        patch("sglang.srt.layers.cp.utils.is_cp_v2_active", return_value=True),
        patch("sglang.srt.layers.cp.utils.get_cp_strategy", return_value=strategy),
        patch("sglang.srt.layers.cp.base.get_parallel", return_value=parallel),
        patch(
            "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
            return_value=4,
        ),
        patch.object(
            strategy,
            "gather_hidden_states",
            return_value=complete_embeds,
        ) as gather,
        patch(
            "sglang.srt.model_executor.runner.eager_runner.torch.cuda.current_stream",
            return_value=None,
        ),
    ):
        prepare_cp_forward(forward_batch)
        result = eager._execute_extend_cp_v2(
            forward_batch, {"input_embeds": complete_embeds}
        )

    assert torch.equal(result, complete_embeds)
    assert gather.call_count == 1
    assert logits_call == {"aux": None, "before_norm": None}


def test_cp_v2_pp2_keeps_local_positions_and_global_owner_metadata():
    input_ids = torch.arange(6, dtype=torch.int64)
    positions = torch.arange(6, dtype=torch.int64)
    complete_embeds = torch.arange(24, dtype=torch.float32).view(6, 4)
    forward_batch = SimpleNamespace(
        can_run_tbo=False,
        forward_mode=ForwardMode.EXTEND,
        input_ids=input_ids,
        positions=positions,
        seq_lens_cpu=torch.tensor([4]),
        extend_seq_lens_cpu=[4],
        attn_cp_metadata=None,
        global_num_tokens_cpu=None,
        out_cache_loc=None,
        spec_info=None,
    )
    strategy = InterleaveCPStrategy(cp_size=4)
    parallel = SimpleNamespace(attn_cp_rank=0, attn_cp_size=4)
    first = _make_exact_dsv4_stage(0, 2, (0, 1))
    last = _make_exact_dsv4_stage(1, 2, (1, 2))
    first.embed_tokens = _UnexpectedEmbedding()
    first.dsa_enable_prefill_cp = True
    last.dsa_enable_prefill_cp = True

    final_call = {}

    def logits_processor(
        _input_ids,
        hidden_states,
        _lm_head,
        logits_forward_batch,
        *,
        aux_hidden_states,
        hidden_states_before_norm,
    ):
        final_call.update(
            hidden=hidden_states,
            before=hidden_states_before_norm,
            forward_batch=logits_forward_batch,
            aux=aux_hidden_states,
        )
        return hidden_states

    first_wrapper = SimpleNamespace(
        model=first,
        pp_group=first.pp_group,
        capture_aux_hidden_states=False,
    )
    last_wrapper = SimpleNamespace(
        model=last,
        pp_group=last.pp_group,
        capture_aux_hidden_states=False,
        logits_processor=logits_processor,
        lm_head=object(),
    )
    eager = EagerRunner.__new__(EagerRunner)
    server_args = SimpleNamespace(enable_lora=False)
    gather_calls = []

    def gather(local_rows, _forward_batch, _stream=None):
        gather_calls.append(tuple(local_rows.shape))
        return local_rows.new_zeros((6, *local_rows.shape[1:]))

    with (
        patch("sglang.srt.layers.cp.utils.is_cp_v2_active", return_value=True),
        patch("sglang.srt.layers.cp.utils.get_cp_strategy", return_value=strategy),
        patch("sglang.srt.layers.cp.base.get_parallel", return_value=parallel),
        patch(
            "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
            return_value=4,
        ),
        patch.object(strategy, "gather_hidden_states", side_effect=gather),
        patch("sglang.srt.models.deepseek_v4.is_cp_v2_active", return_value=True),
        patch("sglang.srt.models.deepseek_v4.dsa_use_prefill_cp", return_value=True),
        patch(
            "sglang.srt.models.deepseek_v4.check_cuda_graph_backend",
            return_value=True,
        ),
        patch(
            "sglang.srt.models.deepseek_v4.resolve_dsv4_owner_plane",
            return_value=SimpleNamespace(dp_rank=0),
        ),
        patch(
            "sglang.srt.models.deepseek_v4.cp_split_and_rebuild_data",
            side_effect=AssertionError("CP-v1 data split ran under CP-v2"),
        ),
        patch(
            "sglang.srt.models.deepseek_v4.cp_split_and_rebuild_position",
            side_effect=AssertionError("CP-v1 position split ran under CP-v2"),
        ),
        patch(
            "sglang.srt.models.deepseek_v4.cp_round_robin_input_ids",
            side_effect=AssertionError("CP-v1 input-id split ran under CP-v2"),
        ),
        patch(
            "sglang.srt.model_executor.runner.eager_runner.torch.cuda.current_stream",
            return_value=None,
        ),
    ):
        prepare_cp_forward(forward_batch)
        eager.model_runner = SimpleNamespace(
            model=first_wrapper, server_args=server_args
        )
        proxy = eager._execute_extend_cp_v2(
            forward_batch, {"input_embeds": complete_embeds}
        )
        eager.model_runner = SimpleNamespace(
            model=last_wrapper, server_args=server_args
        )
        result = eager._execute_extend_cp_v2(
            forward_batch,
            {"input_embeds": complete_embeds, "pp_proxy_tensors": proxy},
        )

    assert isinstance(proxy, PPProxyTensors)
    proxy_hidden, _, _, _, proxy_ids, proxy_positions = unpack_dsv4_exact_pp_proxy(
        proxy,
        hc_mult=first.hc_mult,
        hidden_size=first.hidden_size,
        deferred_mhc=True,
    )
    assert proxy_hidden.shape[0] == 4
    assert torch.equal(proxy_ids, input_ids)
    assert torch.equal(proxy_positions, positions)
    assert last.layers[1].last_positions.tolist() == [0, 4, 0, 0]
    assert last.layers[1].last_positions.shape[0] == proxy_hidden.shape[0]
    assert gather_calls == [(4, 4), (4, 12)]
    assert result.shape == (6, 4)
    assert final_call["before"].shape == (6, 12)
    assert final_call["forward_batch"] is forward_batch
    assert final_call["aux"] is None
    assert torch.equal(forward_batch._dsv4_exact_dp_positions, positions)


def test_cp_v1_model_boundary_still_splits_and_gathers_once():
    input_ids = torch.arange(4, dtype=torch.int64)
    positions = torch.arange(4, dtype=torch.int64)
    forward_batch = _forward_batch(positions)
    forward_batch.attn_cp_metadata = SimpleNamespace(total_seq_lens=4)
    model = _make_exact_dsv4_stage(0, 1, (0, 1))
    strategy = SimpleNamespace(
        gather_hidden_states=lambda rows, _forward_batch, _stream: torch.cat(
            (rows, rows), dim=0
        )
    )

    with (
        patch("sglang.srt.models.deepseek_v4.is_cp_v2_active", return_value=False),
        patch("sglang.srt.models.deepseek_v4.dsa_use_prefill_cp", return_value=True),
        patch(
            "sglang.srt.models.deepseek_v4.cp_split_and_rebuild_data",
            side_effect=lambda _forward_batch, rows: rows[:2],
        ) as split_data,
        patch(
            "sglang.srt.models.deepseek_v4.cp_split_and_rebuild_position",
            return_value=torch.tensor([0, 2]),
        ) as split_positions,
        patch(
            "sglang.srt.models.deepseek_v4.cp_round_robin_input_ids",
            return_value=torch.tensor([0, 2]),
        ) as split_ids,
        patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=strategy),
        patch(
            "sglang.srt.models.deepseek_v4.check_cuda_graph_backend",
            return_value=True,
        ),
        patch(
            "sglang.srt.models.deepseek_v4.resolve_dsv4_owner_plane",
            return_value=SimpleNamespace(dp_rank=0),
        ),
        patch(
            "sglang.srt.models.deepseek_v4.torch.cuda.current_stream", return_value=None
        ),
    ):
        hidden, pre_head = model(input_ids, positions, forward_batch, input_embeds=None)

    split_data.assert_called_once()
    split_positions.assert_called_once()
    split_ids.assert_called_once()
    assert model.layers[0].last_positions.tolist() == [0, 2]
    assert model.layers[0].last_input_ids.tolist() == [0, 2]
    assert hidden.shape == (4, 4)
    assert pre_head.shape == (4, 12)
    assert forward_batch.dsv4_exact_logits_rows_reconstructed
    assert forward_batch.dsv4_exact_logits_owner_rows == 4


def test_cp_v2_nvidia_kv_gathers_logical_rows_before_global_cache_store():
    input_ids = torch.arange(6, dtype=torch.int64)
    positions = torch.arange(6, dtype=torch.int64)
    complete_embeds = torch.arange(24, dtype=torch.float32).view(6, 4)
    global_cache_loc = torch.arange(1, 7, dtype=torch.int64)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=input_ids,
        positions=positions,
        seq_lens_cpu=torch.tensor([4]),
        extend_seq_lens_cpu=[4],
        attn_cp_metadata=None,
        global_num_tokens_cpu=None,
        out_cache_loc=global_cache_loc,
        spec_info=None,
    )
    strategy = InterleaveCPStrategy(cp_size=4)
    cp_parallel = SimpleNamespace(attn_cp_rank=0, attn_cp_size=4)

    attn = MQALayer.__new__(MQALayer)
    nn.Module.__init__(attn)
    attn.fuse_wqa_wkv = False
    attn.wq_a = _TupleIdentity()
    attn.q_norm = nn.Identity()
    attn._compute_q_b = lambda q, _positions, _q_out: q
    attn._compute_kv_bf16 = lambda x, _positions, qkv_a=None: x
    attn.dsa_enable_prefill_cp = True
    attn.cp_size = 4
    attn.use_fused_qk_norm_rope = False
    attn.indexer = None
    attn.compressor = None
    attn.layer_id = 0

    stored = {}
    backend = SimpleNamespace(
        store_cache=lambda **kwargs: stored.update(kwargs),
    )
    gather_calls = []

    def gather(local_rows, _forward_batch, _stream=None):
        gather_calls.append(tuple(local_rows.shape))
        assert local_rows.shape == (4, 4)
        return complete_embeds.clone()

    with (
        patch("sglang.srt.layers.cp.utils.is_cp_v2_active", return_value=True),
        patch("sglang.srt.layers.cp.utils.get_cp_strategy", return_value=strategy),
        patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=strategy),
        patch("sglang.srt.layers.cp.base.get_parallel", return_value=cp_parallel),
        patch(
            "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
            return_value=4,
        ),
        patch.object(strategy, "gather_hidden_states", side_effect=gather),
        patch("sglang.srt.models.deepseek_v4.dsa_use_prefill_cp", return_value=True),
        patch("sglang.srt.models.deepseek_v4._is_npu", False),
        patch(
            "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate."
            "is_unified_kv_triton",
            return_value=False,
        ),
        patch(
            "sglang.srt.layers.attention.dsa.utils.torch.cuda.current_stream",
            return_value=None,
        ),
    ):
        prepare_cp_forward(forward_batch)
        local_embeds, local_positions = cp_split_before_forward(
            complete_embeds, positions, forward_batch
        )
        q, kv = attn._forward_prepare(
            local_embeds, local_positions, forward_batch, backend
        )

    assert q.shape[0] == 4
    assert kv.shape == (6, 4)
    assert gather_calls == [(4, 4)]
    assert stored["swa_k"].shape == (6, 4)
    assert torch.equal(stored["forward_batch"].out_cache_loc, global_cache_loc)
    assert not hasattr(forward_batch, "_cp_v2_out_cache_loc_is_local")


def test_cp_v2_compressor_materializes_logical_rows_on_cuda_and_npu_paths():
    from sglang.srt.layers.attention.dsv4.compressor import Compressor

    input_ids = torch.arange(6, dtype=torch.int64)
    positions = torch.arange(6, dtype=torch.int64)
    complete_rows = torch.arange(24, dtype=torch.float32).view(6, 4)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=input_ids,
        positions=positions,
        seq_lens_cpu=torch.tensor([4]),
        extend_seq_lens_cpu=[4],
        attn_cp_metadata=None,
        global_num_tokens_cpu=None,
        out_cache_loc=None,
    )
    strategy = InterleaveCPStrategy(cp_size=4)
    parallel = SimpleNamespace(attn_cp_rank=0, attn_cp_size=4)
    compressor = Compressor.__new__(Compressor)
    nn.Module.__init__(compressor)
    compressor.wkv_gate = SimpleNamespace(weight=torch.empty(0))
    backend = SimpleNamespace(forward_compress=lambda _compressor, x, _batch: x)
    gather_calls = []

    def gather(local_rows, _forward_batch, _stream=None):
        gather_calls.append(tuple(local_rows.shape))
        return complete_rows.clone()

    with (
        patch("sglang.srt.layers.cp.utils.is_cp_v2_active", return_value=True),
        patch("sglang.srt.layers.cp.utils.get_cp_strategy", return_value=strategy),
        patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=strategy),
        patch("sglang.srt.layers.cp.base.get_parallel", return_value=parallel),
        patch(
            "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
            return_value=4,
        ),
        patch.object(strategy, "gather_hidden_states", side_effect=gather),
        patch(
            "sglang.srt.layers.attention.dsv4.compressor.dsa_use_prefill_cp",
            return_value=True,
        ),
        patch(
            "sglang.srt.layers.attention.dsv4.compressor.linear_bf16_fp32",
            side_effect=lambda x, _weight: x,
        ),
        patch(
            "sglang.srt.layers.attention.dsv4.compressor.get_attn_backend",
            return_value=backend,
        ),
        patch(
            "sglang.srt.layers.attention.dsa.utils.torch.cuda.current_stream",
            return_value=None,
        ),
    ):
        prepare_cp_forward(forward_batch)
        local_rows, _ = cp_split_before_forward(complete_rows, positions, forward_batch)
        cuda_rows = compressor.compute_kv_score(local_rows, forward_batch)
        npu_rows = compressor.forward_npu(local_rows, forward_batch)

    assert gather_calls == [(4, 4), (4, 4)]
    assert torch.equal(cuda_rows, complete_rows)
    assert torch.equal(npu_rows, complete_rows)


def test_cp_v2_dsv4_keeps_global_cache_locations_at_eager_boundary():
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

    input_ids = torch.arange(6, dtype=torch.int64)
    positions = torch.arange(6, dtype=torch.int64)
    complete_embeds = torch.arange(24, dtype=torch.float32).view(6, 4)
    global_cache_loc = torch.arange(1, 7, dtype=torch.int64)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=input_ids,
        positions=positions,
        seq_lens_cpu=torch.tensor([4]),
        extend_seq_lens_cpu=[4],
        attn_cp_metadata=None,
        global_num_tokens_cpu=None,
        out_cache_loc=global_cache_loc,
        spec_info=None,
    )
    strategy = InterleaveCPStrategy(cp_size=4)
    cp_parallel = SimpleNamespace(attn_cp_rank=0, attn_cp_size=4)
    cached_swa_loc = torch.arange(10, 16, dtype=torch.int32)
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.forward_metadata = SimpleNamespace(
        core_attn_metadata=SimpleNamespace(swa_out_cache_loc=cached_swa_loc)
    )
    backend.token_to_kv_pool = SimpleNamespace(
        translate_loc_from_full_to_swa=lambda loc: loc.to(torch.int32)
    )

    assert not hasattr(DeepseekV4ForCausalLM, "cp_v2_local_kv_write_locations")
    with (
        patch("sglang.srt.layers.cp.utils.is_cp_v2_active", return_value=True),
        patch("sglang.srt.layers.cp.utils.get_cp_strategy", return_value=strategy),
        patch("sglang.srt.layers.cp.base.get_parallel", return_value=cp_parallel),
        patch(
            "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
            return_value=4,
        ),
    ):
        prepare_cp_forward(forward_batch)
        with cp_shard_model_inputs(
            complete_embeds,
            positions,
            forward_batch,
            shard_out_cache_loc=getattr(
                DeepseekV4ForCausalLM,
                "cp_v2_local_kv_write_locations",
                False,
            ),
        ):
            assert torch.equal(forward_batch.out_cache_loc, global_cache_loc)
            assert not hasattr(forward_batch, "_cp_v2_out_cache_loc_is_local")
            assert backend.get_swa_out_cache_loc(forward_batch) is cached_swa_loc

    assert torch.equal(forward_batch.out_cache_loc, global_cache_loc)


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
            input_ids, positions, _forward_batch(positions), input_embeds=None
        )

        stages = [
            _make_exact_dsv4_stage(0, 3, (0, 1)),
            _make_exact_dsv4_stage(1, 3, (1, 2)),
            _make_exact_dsv4_stage(2, 3, (2, 3)),
        ]
        proxy = stages[0](input_ids, positions, _forward_batch(positions), None)
        assert isinstance(proxy, PPProxyTensors)
        # Later schedulers may hold stage-local placeholders. The proxy must
        # restore the logical owner's ids and positions before the layer runs.
        proxy = stages[1](
            torch.tensor([30, 30, 30]),
            torch.tensor([90, 90, 90]),
            _forward_batch(torch.tensor([90, 90, 90])),
            None,
            proxy,
        )
        staged_out, staged_pre_head = stages[2](
            torch.tensor([31, 31, 31]),
            torch.tensor([91, 91, 91]),
            _forward_batch(torch.tensor([91, 91, 91])),
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
