"""Override twin of ``sglang.srt.models.qwen3`` -- xorl exact serving (zero-srt port of PR #41).

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


def _Qwen3Attention____init__(
    self,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    layer_id: int = 0,
    start_layer: int = 0,
    rope_theta: float = 1000000,
    rope_scaling: Optional[Dict[str, Any]] = None,
    head_dim: Optional[int] = None,
    max_position_embeddings: int = 32768,
    quant_config: Optional[QuantizationConfig] = None,
    rms_norm_eps: float = None,
    attention_bias: bool = False,
    prefix: str = "",
    alt_stream: Optional[torch.cuda.Stream] = None,
) -> None:
    super(Qwen3Attention, self).__init__()
    self.hidden_size = hidden_size
    self.start_layer = start_layer
    self.tp_size = get_parallel().tp_size
    self.total_num_heads = num_heads
    attn_tp_rank = get_parallel().attn_tp_rank
    attn_tp_size = get_parallel().attn_tp_size

    assert self.total_num_heads % attn_tp_size == 0
    self.num_heads = self.total_num_heads // attn_tp_size
    self.total_num_kv_heads = num_kv_heads
    if self.total_num_kv_heads >= attn_tp_size:
        # Number of KV heads is greater than TP size, so we partition
        # the KV heads across multiple tensor parallel GPUs.
        assert self.total_num_kv_heads % attn_tp_size == 0
    else:
        # Number of KV heads is less than TP size, so we replicate
        # the KV heads across multiple tensor parallel GPUs.
        assert attn_tp_size % self.total_num_kv_heads == 0
    self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
    self.head_dim = head_dim or hidden_size // self.total_num_heads
    self.q_size = self.num_heads * self.head_dim
    self.kv_size = self.num_kv_heads * self.head_dim
    self.scaling = self.head_dim**-0.5
    self.rope_theta = rope_theta
    self.max_position_embeddings = max_position_embeddings
    self.tp_rank = get_parallel().tp_rank

    qwen3_exact = getattr(get_exec().deterministic, "qwen3_dense_exact_mode", False)
    norm_kwargs = (
        dict(batch_invariant_family=RMS_NORM_FAMILY_NO_RESIDUAL)
        if qwen3_exact
        else (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_exec().deterministic.rl_on_policy_target is not None
            else {}
        )
    )
    self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
    self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)

    self.qkv_proj = QKVParallelLinear(
        hidden_size,
        self.head_dim,
        self.total_num_heads,
        self.total_num_kv_heads,
        bias=attention_bias,
        quant_config=quant_config,
        tp_rank=attn_tp_rank,
        tp_size=attn_tp_size,
        prefix=add_prefix("qkv_proj", prefix),
    )
    self.o_proj = RowParallelLinear(
        self.total_num_heads * self.head_dim,
        hidden_size,
        bias=attention_bias,
        quant_config=quant_config,
        tp_rank=attn_tp_rank,
        tp_size=attn_tp_size,
        reduce_results=False,
        prefix=add_prefix("o_proj", prefix),
    )

    self.rotary_emb = get_rope(
        self.head_dim,
        rotary_dim=self.head_dim,
        max_position=max_position_embeddings,
        base=rope_theta,
        rope_scaling=rope_scaling,
    )
    self.attn = RadixAttention(
        self.num_heads,
        self.head_dim,
        self.scaling,
        num_kv_heads=self.num_kv_heads,
        layer_id=layer_id,
        prefix=add_prefix("attn", prefix),
    )
    self.alt_stream = alt_stream

    self.use_fused_qk_norm_mrope = (
        _has_fused_qk_norm_mrope
        and isinstance(self.rotary_emb, MRotaryEmbedding)
        and getattr(self.rotary_emb, "mrope_section", None) is not None
    )
    if self.use_fused_qk_norm_mrope:
        # Scale tensors MUST stay on CPU: the C++ kernel uses .item<float>()
        # which triggers hipMemcpy D2H + sync on CUDA tensors, breaking graph capture.
        # Explicit device='cpu' is required because SGLang constructs models inside
        # a `with torch.device('cuda'):` context that changes the default device.
        self._fused_k_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")
        self._fused_v_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")


def _Qwen3DecoderLayer____init__(
    self,
    config: Qwen3Config,
    layer_id: int = 0,
    start_layer: int = 0,
    quant_config: Optional[QuantizationConfig] = None,
    prefix: str = "",
    alt_stream: Optional[torch.cuda.Stream] = None,
) -> None:
    super(Qwen3DecoderLayer, self).__init__()
    self.hidden_size = config.hidden_size
    if (
        hasattr(config, "rope_parameters")
        and config.rope_parameters
        and "rope_theta" in config.rope_parameters
    ):
        rope_theta = config.rope_parameters["rope_theta"]
        rope_scaling = config.rope_parameters
    else:
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
    max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
    head_dim = getattr(config, "head_dim", None)
    self.self_attn = Qwen3Attention(
        hidden_size=self.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        layer_id=layer_id,
        start_layer=start_layer,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        quant_config=quant_config,
        rms_norm_eps=config.rms_norm_eps,
        attention_bias=config.attention_bias,
        prefix=add_prefix("self_attn", prefix),
        alt_stream=alt_stream,
    )
    self.mlp = Qwen3MLP(
        hidden_size=self.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        quant_config=quant_config,
        prefix=add_prefix("mlp", prefix),
    )

    qwen3_exact = getattr(get_exec().deterministic, "qwen3_dense_exact_mode", False)
    norm_kwargs = (
        None
        if qwen3_exact
        else (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
                override_orig_dtype=torch.float32,
                fp32_residual=True,
            )
            if get_exec().deterministic.rl_on_policy_target is not None
            else {}
        )
    )
    self.input_layernorm = RMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        **(
            {
                "batch_invariant_family": (
                    RMS_NORM_FAMILY_NO_RESIDUAL
                    if layer_id == 0
                    else RMS_NORM_FAMILY_RESIDUAL_TREE
                )
            }
            if qwen3_exact
            else norm_kwargs
        ),
    )
    self.post_attention_layernorm = RMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        **(
            {"batch_invariant_family": RMS_NORM_FAMILY_RESIDUAL_TREE}
            if qwen3_exact
            else norm_kwargs
        ),
    )

    self.layer_scatter_modes = LayerScatterModes.init_new(
        layer_id=layer_id,
        num_layers=config.num_hidden_layers,
        is_layer_sparse=False,
        is_previous_layer_sparse=False,
        is_next_layer_sparse=False,
    )
    self.layer_communicator = LayerCommunicator(
        layer_scatter_modes=self.layer_scatter_modes,
        input_layernorm=self.input_layernorm,
        post_attention_layernorm=self.post_attention_layernorm,
    )


def __apply_patch__(mod):
    # Deferred: the finder imports twins under bypass(), so sglang imports at
    # twin top level would cache modules UNPATCHED. Import here (bypass off)
    # and publish onto mod -- in-tree these were the file's module globals.
    from sglang.xorl.bi import (
        RMS_NORM_FAMILY_NO_RESIDUAL,
        RMS_NORM_FAMILY_RESIDUAL_TREE,
    )

    # Publish the deferred imports onto mod: in-tree they were the srt
    # file's own module globals, and rebound copies resolve via mod.
    mod.RMS_NORM_FAMILY_NO_RESIDUAL = RMS_NORM_FAMILY_NO_RESIDUAL
    mod.RMS_NORM_FAMILY_RESIDUAL_TREE = RMS_NORM_FAMILY_RESIDUAL_TREE
    mod.Qwen3Attention.__init__ = rebind(
        _Qwen3Attention____init__, mod, name="__init__"
    )
    mod.Qwen3DecoderLayer.__init__ = rebind(
        _Qwen3DecoderLayer____init__, mod, name="__init__"
    )
