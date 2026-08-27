"""Override twin of ``sglang.srt.models.qwen2`` -- xorl exact serving (zero-srt port of PR #41).

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

# Self-import is safe at twin-import time: the finder fully executes the
# upstream module before importing its twin, so sys.modules already holds it
# (the bypass-caching hazard only applies to OTHER not-yet-loaded srt modules).
# Needed at def time: Qwen2Model.__init__'s default arg references it.
from sglang.srt.models.qwen2 import Qwen2DecoderLayer


def _is_qwen3_dense_exact_runtime() -> bool:
    return bool(
        getattr(
            get_exec().deterministic,
            "qwen3_dense_exact_mode",
            False,
        )
    )


def _Qwen2MLP____init__(
    self,
    hidden_size: int,
    intermediate_size: int,
    hidden_act: str,
    quant_config: Optional[QuantizationConfig] = None,
    prefix: str = "",
) -> None:
    super(Qwen2MLP, self).__init__()
    self.gate_up_proj = MergedColumnParallelLinear(
        hidden_size,
        [intermediate_size] * 2,
        bias=False,
        quant_config=quant_config,
        prefix=add_prefix("gate_up_proj", prefix),
    )
    self.down_proj = RowParallelLinear(
        intermediate_size,
        hidden_size,
        bias=False,
        quant_config=quant_config,
        prefix=add_prefix("down_proj", prefix),
    )
    if hidden_act != "silu":
        raise ValueError(
            f"Unsupported activation: {hidden_act}. Only silu is supported for now."
        )
    self.act_fn = SiluAndMul()


def _Qwen2Model____init__(
    self,
    config: Qwen2Config,
    quant_config: Optional[QuantizationConfig] = None,
    prefix: str = "",
    decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer,
    alt_stream: Optional[torch.cuda.Stream] = None,
) -> None:
    super(Qwen2Model, self).__init__()
    self.config = config
    self.padding_idx = getattr(config, "pad_token_id", None)
    self.vocab_size = config.vocab_size
    self.pp_group = get_pp_group()
    qwen3_exact = _is_qwen3_dense_exact_runtime()

    if self.pp_group.is_first_rank:
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            use_attn_tp_group=is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
            params_dtype=(
                torch.float32
                if (
                    get_exec().deterministic.rl_on_policy_target is not None
                    and not qwen3_exact
                )
                else None
            ),
        )
    else:
        self.embed_tokens = PPMissingLayer()

    # Use the provided decoder layer type or default to Qwen2DecoderLayer
    decoder_layer_type = decoder_layer_type or Qwen2DecoderLayer
    pp_start_layer, _ = get_pp_indices(
        config.num_hidden_layers,
        self.pp_group.rank_in_group,
        self.pp_group.world_size,
    )
    self.layers, self.start_layer, self.end_layer = make_layers(
        config.num_hidden_layers,
        lambda idx, prefix: decoder_layer_type(
            layer_id=idx,
            start_layer=pp_start_layer,
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
        ),
        pp_rank=self.pp_group.rank_in_group,
        pp_size=self.pp_group.world_size,
        prefix=add_prefix("layers", prefix),
    )
    if self.pp_group.is_last_rank:
        norm_kwargs = (
            {"batch_invariant_family": RMS_NORM_FAMILY_RESIDUAL_TREE}
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
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs)
    else:
        self.norm = PPMissingLayer(return_tuple=True)

    # For EAGLE3 support
    self.layers_to_capture = []


def _Qwen2Model__forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor = None,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
) -> Union[torch.Tensor, PPProxyTensors]:
    if self.pp_group.is_first_rank:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
    else:
        assert pp_proxy_tensors is not None
        hidden_states = pp_proxy_tensors["hidden_states"]
        residual = pp_proxy_tensors["residual"]

    aux_hidden_states = []
    for i in range(self.start_layer, self.end_layer):
        if i in self.layers_to_capture:
            aux_hidden_states.append(
                hidden_states + residual if residual is not None else hidden_states
            )
        layer = self.layers[i]
        hidden_states, residual = layer(
            positions,
            hidden_states,
            forward_batch,
            residual,
        )
    if not self.pp_group.is_last_rank:
        return PPProxyTensors(
            {
                "hidden_states": hidden_states,
                "residual": residual,
            }
        )
    else:
        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

    if len(aux_hidden_states) == 0:
        return hidden_states

    return hidden_states, aux_hidden_states


def _Qwen2Model__load_kv_cache_scales(self, quantization_param_path: str) -> None:
    tp_size = get_parallel().tp_size
    tp_rank = get_parallel().tp_rank
    for layer_idx, scaling_factor in kv_cache_scales_loader(
        quantization_param_path,
        tp_rank,
        tp_size,
        self.config.num_hidden_layers,
        self.config.__class__.model_type,
    ):
        if not isinstance(self.layers[layer_idx], nn.Identity):
            layer_self_attn = self.layers[layer_idx].self_attn
        if hasattr(layer_self_attn.attn, "k_scale"):
            layer_self_attn.attn.k_scale = scaling_factor
            layer_self_attn.attn.v_scale = scaling_factor
        else:
            raise RuntimeError(
                "Self attention has no KV cache scaling factor attribute!"
            )


def __apply_patch__(mod):
    # Publish the twin's top-level imports onto mod: in-tree they were the
    # srt file's own module globals, and rebound copies resolve via mod.
    mod.Qwen2DecoderLayer = Qwen2DecoderLayer
    # Deferred: the finder imports twins under bypass(), so sglang imports at
    # twin top level would cache modules UNPATCHED. Import here (bypass off)
    # and publish onto mod -- in-tree these were the file's module globals.
    from sglang.xorl.bi import RMS_NORM_FAMILY_RESIDUAL_TREE

    # Publish the deferred imports onto mod: in-tree they were the srt
    # file's own module globals, and rebound copies resolve via mod.
    mod.RMS_NORM_FAMILY_RESIDUAL_TREE = RMS_NORM_FAMILY_RESIDUAL_TREE
    mod._is_qwen3_dense_exact_runtime = rebind(_is_qwen3_dense_exact_runtime, mod)
    mod.Qwen2MLP.__init__ = rebind(_Qwen2MLP____init__, mod, name="__init__")
    mod.Qwen2Model.__init__ = rebind(_Qwen2Model____init__, mod, name="__init__")
    mod.Qwen2Model.forward = rebind(_Qwen2Model__forward, mod, name="forward")
    mod.Qwen2Model.load_kv_cache_scales = rebind(
        _Qwen2Model__load_kv_cache_scales, mod, name="load_kv_cache_scales"
    )
