"""Override twin of ``sglang.srt.models.qwen3_5`` -- xorl exact serving (zero-srt port of PR #41).

Verbatim copies of the retired in-tree edits. Copies live at module top level
(collision-proof ``_Cls__name`` def names for methods) so cross-references stay
module-global, and every attach goes through ``rebind`` so the copy resolves
names via the PATCHED srt module's live dict -- identical to in-tree, including
monkeypatching and ``global`` writes. Replaced/removed upstream symbols are
pinned in ``sglang.overrides._twin_pins``; when the pin test fires after an
upstream sync, re-derive the copies and re-pin.
"""

# ruff: noqa: F821 -- the verbatim copies below resolve upstream names at call
# time via rebind() over the live srt module dict; they are undefined in this
# file's namespace by design.

from __future__ import annotations

from sglang.overrides._twin_bind import rebind


def _qwen35_exact_mode_enabled() -> bool:
    return is_qwen35_gdn_exact_mode(get_global_server_args())


def _qwen35_rmsnorm_family(config) -> str:
    configured_family = getattr(config, "_qwen35_rmsnorm_family", None)
    try:
        runtime_args = get_server_args()
    except ValueError:
        runtime_args = None
    runtime_exact = runtime_args is not None and is_qwen35_gdn_exact_mode(runtime_args)
    if runtime_exact:
        # Deliberate deviation from the in-tree original (which read the field
        # off get_server_args() and tripped the global-config-read ratchet —
        # never run against the original PR): the field is NS("exec.deterministic"),
        # so the resolved value is read from the namespace bag.
        runtime_family = get_exec().deterministic.qwen35_rmsnorm_family
        if configured_family is not None and configured_family != runtime_family:
            raise RuntimeError(
                "Qwen RMSNorm family drifted between the model config and runtime: "
                f"config={configured_family!r}, runtime={runtime_family!r}."
            )
        family = runtime_family
    else:
        family = configured_family or "v1"
    if family not in ("v1", "v2"):
        raise ValueError(f"Unsupported Qwen3.5/3.6 RMSNorm family: {family!r}")
    if family == "v2" and not (
        runtime_exact or bool(getattr(config, "_qwen35_gdn_exact_mode", False))
    ):
        raise RuntimeError(
            "Qwen families-v2 RMSNorm is admitted only in the exact XORL serving lane."
        )
    return family


def _qwen35_rope_class_b_enabled() -> bool:
    return is_qwen35_rope_class_b(get_global_server_args())


def _Qwen3_5AttentionDecoderLayer___apply_rotary(
    self,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if _is_cuda and _qwen35_exact_mode_enabled():
        seq_len = q.shape[0]
        if not _qwen35_rope_class_b_enabled():
            raise RuntimeError("Exact Qwen3.5-family serving requires Class-B RoPE")
        if positions.ndim not in (1, 2):
            raise RuntimeError(
                "Qwen3.5-family Class-B RoPE requires scalar "
                f"text positions; got rank {positions.ndim}"
            )
        if positions.ndim == 2:
            torch._assert_async(
                (positions == positions[:1]).all(),
                "Qwen3.5-family Class-B RoPE does not support multimodal "
                "positions with distinct temporal/height/width axes",
            )
        exact_positions = positions.reshape(-1)[:seq_len]
        if exact_positions.numel() != seq_len:
            raise RuntimeError(
                "Exact Qwen3.5-family RoPE requires one scalar position per token; "
                f"got {positions.numel()} positions for {seq_len} tokens"
            )
        if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16:
            raise RuntimeError(
                "Qwen3.5-family Class-B RoPE requires BF16 q/k; "
                f"got q={q.dtype}, k={k.dtype}"
            )
        if not self.rotary_emb.is_neox_style:
            raise RuntimeError(
                "Qwen3.5-family Class-B RoPE supports only the qualified "
                "Neox half-split feature layout"
            )
        return self.rotary_emb(exact_positions, q, k)
    return self.rotary_emb(positions, q, k)


def _Qwen3_5AttentionDecoderLayer____init__(
    self,
    config: Qwen3_5TextConfig,
    layer_id: int,
    quant_config: Optional[QuantizationConfig] = None,
    prefix: str = "",
    alt_stream: Optional[torch.cuda.Stream] = None,
    is_nextn: bool = False,
) -> None:
    super(Qwen3_5AttentionDecoderLayer, self).__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.attn_tp_rank = get_parallel().attn_tp_rank
    self.attn_tp_size = get_parallel().attn_tp_size
    self.total_num_heads = config.num_attention_heads
    assert self.total_num_heads % self.attn_tp_size == 0
    self.num_heads = self.total_num_heads // self.attn_tp_size
    self.total_num_kv_heads = config.num_key_value_heads
    if self.total_num_kv_heads >= self.attn_tp_size:
        assert self.total_num_kv_heads % self.attn_tp_size == 0
    else:
        assert self.attn_tp_size % self.total_num_kv_heads == 0
    self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
    self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
    self.q_size = self.num_heads * self.head_dim
    self.kv_size = self.num_kv_heads * self.head_dim
    self.scaling = self.head_dim**-0.5
    self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

    self.rope_theta, rope_scaling = get_rope_config(config)
    self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    self.layer_id = layer_id

    # If rope_scaling doesn't specify a scaling type, treat as no scaling
    if rope_scaling and not ("rope_type" in rope_scaling or "type" in rope_scaling):
        rope_scaling = None

    self.attn_output_gate = getattr(config, "attn_output_gate", True)
    if self.attn_output_gate:
        logger.warning_once("using attn output gate!")
    rmsnorm_family = _qwen35_rmsnorm_family(config)
    self.use_fused_qk_norm_rope = bool(
        _is_cuda
        and self.attn_output_gate
        and get_exec().kernel.enable_fused_qk_norm_rope
        and rmsnorm_family != "v2"
    )

    self.rotary_emb = get_rope(
        head_size=self.head_dim,
        rotary_dim=self.head_dim,
        max_position=self.max_position_embeddings,
        rope_scaling=rope_scaling,
        base=self.rope_theta,
        partial_rotary_factor=self.partial_rotary_factor,
        is_neox_style=True,
        dtype=torch.get_default_dtype(),
    )

    self.qkv_proj = QKVParallelLinear(
        config.hidden_size,
        self.head_dim,
        self.total_num_heads * (1 + self.attn_output_gate),
        self.total_num_kv_heads,
        bias=False,
        quant_config=quant_config,
        tp_rank=self.attn_tp_rank,
        tp_size=self.attn_tp_size,
        prefix=add_prefix("qkv_proj", prefix),
    )

    self.o_proj = RowParallelLinear(
        self.total_num_heads * self.head_dim,
        config.hidden_size,
        bias=False,
        quant_config=quant_config,
        reduce_results=False,
        tp_rank=self.attn_tp_rank,
        tp_size=self.attn_tp_size,
        prefix=add_prefix("o_proj", prefix),
    )

    self.attn = RadixAttention(
        self.num_heads,
        self.head_dim,
        self.scaling,
        num_kv_heads=self.num_kv_heads,
        layer_id=layer_id,
        prefix=f"{prefix}.attn",
        quant_config=quant_config,
    )

    # Dense MLP for non-MoE variant
    if config.model_type == "qwen3_5_text":
        self.mlp = Qwen2MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
        )
        is_layer_sparse = False
        is_previous_layer_sparse = False
        is_next_layer_sparse = False
    elif config.model_type == "qwen3_5_moe_text":
        self.mlp = Qwen2MoeSparseMoeBlock(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            alt_stream=(
                alt_stream if (_is_cuda or _disable_shared_experts_fusion()) else None
            ),
            prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
            is_nextn=is_nextn,
            support_shared_expert_fusion=not _disable_shared_experts_fusion(),
        )
        is_layer_sparse = True
        is_previous_layer_sparse = True
        is_next_layer_sparse = True
    else:
        raise ValueError(f"Invalid model type: {config.model_type}")

    self.layer_scatter_modes = LayerScatterModes.init_new(
        layer_id=layer_id,
        num_layers=config.num_hidden_layers,
        is_layer_sparse=is_layer_sparse,
        is_previous_layer_sparse=is_previous_layer_sparse,
        is_next_layer_sparse=is_next_layer_sparse,
    )

    self.input_layernorm = GemmaRMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        xorl_batch_invariant_version=rmsnorm_family,
    )
    self.post_attention_layernorm = GemmaRMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        xorl_batch_invariant_version=rmsnorm_family,
    )

    self.q_norm = GemmaRMSNorm(
        self.head_dim,
        eps=config.rms_norm_eps,
        xorl_batch_invariant_version=rmsnorm_family,
    )
    self.k_norm = GemmaRMSNorm(
        self.head_dim,
        eps=config.rms_norm_eps,
        xorl_batch_invariant_version=rmsnorm_family,
    )

    # Standard attention layers benefit from a fused quant epilogue only
    # when qkv_proj can consume the returned quantized tuple.
    enable_fused_ar_quant = (
        _enable_qwen35_fused_ar_quant() and _linear_accepts_fp8_tuple(self.qkv_proj)
    )
    self.layer_communicator = LayerCommunicator(
        layer_scatter_modes=self.layer_scatter_modes,
        input_layernorm=self.input_layernorm,
        post_attention_layernorm=self.post_attention_layernorm,
        allow_reduce_scatter=True,
        is_last_layer=(layer_id == config.num_hidden_layers - 1),
        enable_fused_ar_quant=enable_fused_ar_quant,
        fused_ar_quant_keep_bf16=False,
    )

    self.alt_stream = alt_stream


def _Qwen3_5AttentionDecoderLayer__forward_prepare_fused_gate(
    self, positions, hidden_states
):
    if _use_aiter and isinstance(hidden_states, tuple):
        hidden_states = _select_fused_ar_input_for_linear(hidden_states, self.qkv_proj)
    qkv, _ = self.qkv_proj(hidden_states)
    if self.attn_output_gate:
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        seq_len = q_gate.shape[0]
        q_flat, k_flat, gate_flat = fused_qk_gemma_rmsnorm_with_gate(
            q_gate,
            k,
            self.q_norm.weight.data,
            self.k_norm.weight.data,
            self.q_norm.variance_epsilon,
            self.head_dim,
            self.num_heads,
        )
        q = q_flat.view(seq_len, -1)
        k = k_flat.view(seq_len, -1)
        gate = gate_flat.view(seq_len, -1)
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        gate = None
        q, k = self._apply_qk_norm(q, k)

    q, k = self._apply_rotary(positions, q, k)
    return q, k, v, gate


def _Qwen3_5AttentionDecoderLayer__forward_prepare_native(
    self, positions, hidden_states
):
    if _use_aiter and isinstance(hidden_states, tuple):
        hidden_states = _select_fused_ar_input_for_linear(hidden_states, self.qkv_proj)
    qkv, _ = self.qkv_proj(hidden_states)
    if self.attn_output_gate:
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        orig_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        q = q.reshape(*orig_shape, -1)
        # gate stays as 3D strided view; fused_sigmoid_mul handles it directly
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        gate = None

    q, k = self._apply_qk_norm(q, k)
    q, k = self._apply_rotary(positions, q, k)
    return q, k, v, gate


def _Qwen3_5AttentionDecoderLayer__self_attention(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """Full attention forward pass."""
    if _is_cuda and self.attn_output_gate and self.use_fused_qk_norm_rope:
        q, k, v, gate = self.forward_prepare_cuda_fused(
            positions=positions,
            hidden_states=hidden_states,
        )
    elif (_is_hip or _is_xpu or _is_cpu) and self.attn_output_gate:
        q, k, v, gate = self.forward_prepare_fused_gate(
            positions=positions,
            hidden_states=hidden_states,
        )
    elif (
        not _is_npu
        or forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
        or not self.attn_output_gate
    ):
        q, k, v, gate = self.forward_prepare_native(
            positions=positions,
            hidden_states=hidden_states,
        )
    else:
        q, k, v, gate = self.forward_prepare_npu(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

    attn_output = self.attn(q, k, v, forward_batch)

    if self.attn_output_gate:
        if _is_cuda and _qwen35_exact_mode_enabled():
            # Match the full-depth trainer's BF16 execution order exactly:
            # materialize the strided gate, evaluate sigmoid, then perform
            # a separate out-of-place multiply. The fused kernel rounds
            # differently even when both input operands are byte-identical.
            attn_output = attn_output.reshape(attn_output.shape[0], -1).contiguous()
            gate_value = gate.reshape(gate.shape[0], -1)
            attn_output = attn_output * torch.sigmoid(gate_value)
        elif not _is_npu:
            attn_output = fused_sigmoid_mul(attn_output, gate, inplace=True)
        else:
            gate_val = gate.reshape(gate.shape[0], -1) if gate.ndim == 3 else gate
            attn_output.mul_(torch.sigmoid(gate_val))

    output, _ = self.o_proj(attn_output)
    return output


def _Qwen3_5ForCausalLM____init__(
    self,
    config: Qwen3_5TextConfig,
    quant_config: Optional[QuantizationConfig] = None,
    prefix: str = "",
    is_nextn: bool = False,
) -> None:
    super(Qwen3_5ForCausalLM, self).__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.pp_group = get_pp_group()

    if _is_hip:
        self._maybe_autodisable_shared_experts_fusion(config, quant_config)

    alt_stream = get_stream("alt") if _is_cuda or _hip_use_alt_stream else None

    # Embedding layer
    if self.pp_group.is_first_rank:
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            enable_tp=not is_dp_attention_enabled(),
        )
    else:
        self.embed_tokens = PPMissingLayer()

    # Decoder layers
    def get_layer(idx: int, prefix: str):
        layer_type = config.layers_block_type[idx]
        layer_class = ALL_DECODER_LAYER_TYPES[layer_type]
        if layer_type == "attention":
            prefix = add_prefix("self_attn", prefix)
        else:
            prefix = add_prefix("linear_attn", prefix)
        return layer_class(
            config=config,
            layer_id=idx,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
            is_nextn=is_nextn,
        )

    self.layers, self._start_layer, self._end_layer = make_layers(
        config.num_hidden_layers,
        get_layer,
        pp_rank=self.pp_group.rank_in_group,
        pp_size=self.pp_group.world_size,
        prefix=f"{prefix}.layers",
    )

    # Final normalization
    if self.pp_group.is_last_rank:
        self.norm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            xorl_batch_invariant_version=_qwen35_rmsnorm_family(config),
        )
    else:
        self.norm = PPMissingLayer()

    self.layers_to_capture = []


def _Qwen3_5LinearDecoderLayer____init__(
    self,
    config: Qwen3_5TextConfig,
    layer_id: int,
    quant_config: Optional[QuantizationConfig] = None,
    prefix: str = "",
    alt_stream: Optional[torch.cuda.Stream] = None,
    is_nextn: bool = False,
) -> None:
    super(Qwen3_5LinearDecoderLayer, self).__init__()
    self.config = config
    self.layer_id = layer_id

    self.linear_attn = Qwen3_5GatedDeltaNet(
        config, layer_id, quant_config, alt_stream, prefix
    )

    # NOTE: Determine the MLP type based on the model type
    # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
    if config.model_type == "qwen3_5_moe_text":
        self.mlp = Qwen2MoeSparseMoeBlock(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            alt_stream=(
                alt_stream if (_is_cuda or _disable_shared_experts_fusion()) else None
            ),
            prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
            is_nextn=is_nextn,
            support_shared_expert_fusion=not _disable_shared_experts_fusion(),
        )
        is_layer_sparse = True
        is_previous_layer_sparse = True
        is_next_layer_sparse = True
    elif config.model_type == "qwen3_5_text":
        self.mlp = Qwen2MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
        )
        is_layer_sparse = False
        is_previous_layer_sparse = False
        is_next_layer_sparse = False
    else:
        raise ValueError(f"Invalid model type: {config.model_type}")

    self.layer_scatter_modes = LayerScatterModes.init_new(
        layer_id=layer_id,
        num_layers=config.num_hidden_layers,
        is_layer_sparse=is_layer_sparse,
        is_previous_layer_sparse=is_previous_layer_sparse,
        is_next_layer_sparse=is_next_layer_sparse,
    )

    rmsnorm_family = _qwen35_rmsnorm_family(config)
    self.input_layernorm = GemmaRMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        xorl_batch_invariant_version=rmsnorm_family,
    )
    self.post_attention_layernorm = GemmaRMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        xorl_batch_invariant_version=rmsnorm_family,
    )
    # GDN layers need both bf16 (for the small in_proj_ba gating
    # projection) and a quantized tuple only when in_proj_qkvz can consume
    # it. Otherwise, stay on the plain AR+RMSNorm path.
    enable_fused_ar_quant = (
        _enable_qwen35_fused_ar_quant()
        and _linear_accepts_fp8_tuple(self.linear_attn.in_proj_qkvz)
    )
    self.layer_communicator = LayerCommunicator(
        layer_scatter_modes=self.layer_scatter_modes,
        input_layernorm=self.input_layernorm,
        post_attention_layernorm=self.post_attention_layernorm,
        allow_reduce_scatter=True,
        is_last_layer=(layer_id == config.num_hidden_layers - 1),
        enable_fused_ar_quant=enable_fused_ar_quant,
        fused_ar_quant_keep_bf16=enable_fused_ar_quant,
    )


def __apply_patch__(mod):
    # Deferred: the finder imports twins under bypass(), so sglang imports at
    # twin top level would cache modules UNPATCHED. Import here (bypass off)
    # and publish onto mod -- in-tree these were the file's module globals.
    from sglang.srt.runtime_context import (
        get_exec,
        get_forward,
        get_parallel,
        get_server_args,
        get_stream,
    )
    from sglang.srt.server_args import (
        get_global_server_args,
        is_qwen35_gdn_exact_mode,
        is_qwen35_rope_class_b,
    )

    # Publish the deferred imports onto mod: in-tree they were the srt
    # file's own module globals, and rebound copies resolve via mod.
    mod.get_exec = get_exec
    mod.get_forward = get_forward
    mod.get_parallel = get_parallel
    mod.get_server_args = get_server_args
    mod.get_stream = get_stream
    mod.get_global_server_args = get_global_server_args
    mod.is_qwen35_gdn_exact_mode = is_qwen35_gdn_exact_mode
    mod.is_qwen35_rope_class_b = is_qwen35_rope_class_b
    mod._qwen35_exact_mode_enabled = rebind(_qwen35_exact_mode_enabled, mod)
    mod._qwen35_rmsnorm_family = rebind(_qwen35_rmsnorm_family, mod)
    mod._qwen35_rope_class_b_enabled = rebind(_qwen35_rope_class_b_enabled, mod)
    mod.Qwen3_5AttentionDecoderLayer._apply_rotary = rebind(
        _Qwen3_5AttentionDecoderLayer___apply_rotary, mod, name="_apply_rotary"
    )
    mod.Qwen3_5AttentionDecoderLayer.__init__ = rebind(
        _Qwen3_5AttentionDecoderLayer____init__, mod, name="__init__"
    )
    mod.Qwen3_5AttentionDecoderLayer.forward_prepare_fused_gate = rebind(
        _Qwen3_5AttentionDecoderLayer__forward_prepare_fused_gate,
        mod,
        name="forward_prepare_fused_gate",
    )
    mod.Qwen3_5AttentionDecoderLayer.forward_prepare_native = rebind(
        _Qwen3_5AttentionDecoderLayer__forward_prepare_native,
        mod,
        name="forward_prepare_native",
    )
    mod.Qwen3_5AttentionDecoderLayer.self_attention = rebind(
        _Qwen3_5AttentionDecoderLayer__self_attention, mod, name="self_attention"
    )
    mod.Qwen3_5ForCausalLM.__init__ = rebind(
        _Qwen3_5ForCausalLM____init__, mod, name="__init__"
    )
    mod.Qwen3_5LinearDecoderLayer.__init__ = rebind(
        _Qwen3_5LinearDecoderLayer____init__, mod, name="__init__"
    )
