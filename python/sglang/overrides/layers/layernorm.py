"""Override twin of ``sglang.srt.layers.layernorm`` -- xorl exact serving (zero-srt port of PR #41).

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

from typing import Literal, Optional, Tuple, Union

from sglang.overrides._twin_bind import rebind


def _validate_qwen_v2_norm_tensor(
    tensor: torch.Tensor, *, name: str, shape: torch.Size | None = None
) -> None:
    if tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
        raise RuntimeError(
            f"The Qwen families-v2 RMSNorm program requires CUDA BF16 {name}; "
            f"got device={tensor.device}, dtype={tensor.dtype}."
        )
    if shape is not None and tensor.shape != shape:
        raise RuntimeError(
            f"The Qwen families-v2 RMSNorm program requires {name} shape {tuple(shape)}, "
            f"got {tuple(tensor.shape)}."
        )


def _RMSNorm___forward_xorl_batch_invariant(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    post_residual_addition: Optional[torch.Tensor],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    family = self.batch_invariant_family
    if family is None:
        raise RuntimeError(
            "The XORL batch-invariant target reached an RMSNorm site without "
            "an explicit batch_invariant_family."
        )
    if family == RMS_NORM_FAMILY_NO_RESIDUAL and residual is not None:
        raise RuntimeError(
            "An XORL no-residual RMSNorm site received a residual stream."
        )
    if post_residual_addition is not None:
        raise RuntimeError(
            "The XORL batch-invariant RMSNorm contract does not support "
            "post_residual_addition."
        )
    if self.override_orig_dtype is not None or self.fp32_residual:
        raise RuntimeError(
            "The XORL batch-invariant RMSNorm contract does not support "
            "override_orig_dtype or fp32_residual."
        )
    if not self.has_weight:
        raise RuntimeError(
            "The XORL batch-invariant RMSNorm contract requires a learned weight."
        )
    if x.dtype != torch.bfloat16 or self.weight.dtype != torch.bfloat16:
        raise RuntimeError(
            "The XORL batch-invariant RMSNorm contract requires BF16 input "
            f"and weight, got {x.dtype} and {self.weight.dtype}."
        )
    if residual is not None and residual.dtype != torch.bfloat16:
        raise RuntimeError(
            "The XORL batch-invariant RMSNorm contract requires a BF16 residual, "
            f"got {residual.dtype}."
        )

    version = resolve_or_validate_xorl_bi_family(None)
    if version == "v2":
        return rms_norm_v2(
            x,
            self.weight.data,
            self.variance_epsilon,
            residual=residual,
        )
    if residual is not None:
        return bi_fused_add_rms_norm(
            x,
            residual,
            self.weight.data,
            self.variance_epsilon,
            family=family,
        )
    return bi_rms_norm(
        x,
        self.weight.data,
        self.variance_epsilon,
        family=family,
    )


def _GemmaRMSNorm____init__(
    self,
    hidden_size: int,
    eps: float = 1e-6,
    xorl_batch_invariant_version: Optional[Literal["v1", "v2"]] = None,
) -> None:
    super(GemmaRMSNorm, self).__init__()
    if xorl_batch_invariant_version not in (None, "v1", "v2"):
        raise ValueError(
            "GemmaRMSNorm xorl_batch_invariant_version must be None, 'v1', or 'v2'; "
            f"got {xorl_batch_invariant_version!r}."
        )
    self.weight = nn.Parameter(torch.zeros(hidden_size))
    self.variance_epsilon = eps
    self.xorl_batch_invariant_version = xorl_batch_invariant_version
    self.register_buffer("gemma_weight", torch.ones_like(self.weight), persistent=False)
    # (Chen-0210) Gemma weight = standard_weight + 1. Precompute once.
    # If TRTLLM allreduce fusion ever provides gemma-style norm
    # natively, this can be removed.
    self.weight.weight_loader = self._weight_loader


def _GemmaRMSNorm__forward_cuda(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if self.xorl_batch_invariant_version == "v2":
        if not is_batch_invariant_mode_enabled() or not is_batch_invariant_op_enabled(
            "rms_norm"
        ):
            raise RuntimeError(
                "The Qwen families-v2 RMSNorm program requires the exact batch-invariant "
                "rms_norm contract to be engaged."
            )
        _validate_qwen_v2_norm_tensor(x, name="input")
        if post_residual_addition is not None:
            raise RuntimeError(
                "The Qwen families-v2 RMSNorm program does not admit post_residual_addition."
            )
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])
        if x_2d.stride(-1) != 1:
            x_2d = x_2d.contiguous()
        if residual is None:
            out = rms_norm_v2(
                x_2d,
                self.weight.data,
                self.variance_epsilon,
                zero_centered=True,
            )
            return out.reshape(original_shape)
        _validate_qwen_v2_norm_tensor(residual, name="residual", shape=x.shape)
        residual_2d = residual.reshape(-1, original_shape[-1])
        if residual_2d.stride(-1) != 1:
            residual_2d = residual_2d.contiguous()
        out, residual_out = rms_norm_v2(
            x_2d,
            self.weight.data,
            self.variance_epsilon,
            residual=residual_2d,
            zero_centered=True,
        )
        return out.reshape(original_shape), residual_out.reshape(original_shape)

    if is_batch_invariant_mode_enabled() and is_batch_invariant_op_enabled("rms_norm"):
        # Exact Qwen splits zero-centered RMSNorm into two numerical
        # families. No-residual sites use the family-1 BI kernel. Residual
        # sites keep the bf16 add in eager torch and reach the interposed
        # BI mean through forward_native, matching the trainer's
        # fast_zero_centered_batch_invariant_* implementations.
        if residual is None:
            orig_dtype = x.dtype
            out = rms_norm_batch_invariant(
                x.float(),
                1.0 + self.weight.data.float(),
                self.variance_epsilon,
            )
            return out.to(orig_dtype)
        return self.forward_native(x, residual, post_residual_addition)
    return self._forward_impl(x, residual, post_residual_addition)


def _GemmaRMSNorm__forward_with_allreduce_fusion(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
    use_attn_tp_group: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Forward with allreduce fusion; uses 1 + weight for fused kernels."""
    if self.xorl_batch_invariant_version == "v2":
        raise RuntimeError(
            "The Qwen families-v2 RMSNorm program cannot use an allreduce-fused v1 norm path."
        )
    return _forward_with_allreduce_fusion(
        self,
        x,
        residual,
        post_residual_addition,
        self.gemma_weight,
        use_attn_tp_group=True,
    )


def _GemmaRMSNorm__forward_with_allreduce_fusion_quant_per_group(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    group_size: int = 128,
    use_attn_tp_group: bool = True,
    keep_bf16: bool = False,
):
    """Fused AR + RMSNorm + per-group FP8 quant (Gemma-style: weight + 1)."""
    if self.xorl_batch_invariant_version == "v2":
        raise RuntimeError(
            "The Qwen families-v2 RMSNorm program cannot use an allreduce/quant-fused v1 norm path."
        )
    return _forward_with_allreduce_fusion_quant_per_group(
        self,
        x,
        residual,
        self.gemma_weight,
        group_size,
        use_attn_tp_group,
        keep_bf16,
    )


def _RMSNorm____init__(
    self,
    hidden_size: int,
    eps: float = 1e-6,
    var_hidden_size: Optional[int] = None,
    cast_x_before_out_mul: bool = False,
    fp32_residual: bool = False,
    has_weight: bool = True,
    weight_dtype: Optional = None,
    override_orig_dtype: Optional = None,
    x_pad_to_multiple: int = 0,
    batch_invariant_family: Optional[RMSNormFamily] = None,
) -> None:
    super(RMSNorm, self).__init__()
    self.has_weight = has_weight
    self.cast_x_before_out_mul = cast_x_before_out_mul
    self.fp32_residual = fp32_residual
    self.override_orig_dtype = override_orig_dtype
    if (
        batch_invariant_family is not None
        and batch_invariant_family not in RMS_NORM_FAMILIES
    ):
        raise ValueError(
            f"Unknown RMSNorm family {batch_invariant_family!r}; "
            f"expected one of {RMS_NORM_FAMILIES}"
        )
    self.batch_invariant_family = batch_invariant_family
    if self.has_weight:
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))
    else:
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
    self.variance_epsilon = eps
    self.hidden_size = hidden_size
    self.variance_size_override = (
        None if var_hidden_size == hidden_size else var_hidden_size
    )
    # When > 0, fuse a zero-pad of the last dim out to a multiple of
    # this value into the rmsnorm kernel via aiter's
    # `fused_add_rmsnorm_pad` Triton kernel. The padded output has
    # shape (M, ceil(N/x_pad_to_multiple)*x_pad_to_multiple); the
    # residual_out stays at the original (M, N) shape.

    if _use_aiter:
        self.x_pad_to_multiple = x_pad_to_multiple
        self._fused_pad_kernel = None

        if x_pad_to_multiple > 0:
            try:
                from aiter.ops.triton.fused_add_rmsnorm_pad import (
                    fused_add_rmsnorm_pad as _fused_add_rmsnorm_pad,
                )

                self._fused_pad_kernel = _fused_add_rmsnorm_pad
            except ImportError:
                self._fused_pad_kernel = None
        self._forward_method = self.forward_aiter


def _RMSNorm__forward_aiter(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # Fix dsv4 dp attenton issue
    # the symptom is torch.AcceleratorError: HIP error: invalid configuration argument
    if x.shape[0] == 0:
        if residual is not None:
            return x, residual
        return x
    if self.weight.data.dtype != x.dtype:
        # AITER's ROCm rmsnorm2d_fwd requires weight/activation dtypes to match;
        # FP32 weight + BF16 activation yields finite-but-corrupted output on gfx950.
        return self.forward_native(x, residual, post_residual_addition)
    # Aiter's RMSNorm kernels expect 2D contiguous inputs. Keep the
    # already-safe layout as a zero-copy path, and only normalize strided or
    # higher-rank views such as Q/K slices from packed QKV projections.
    needs_reshape = x.dim() != 2 and residual is None
    if needs_reshape:
        original_shape = x.shape
        x = x.contiguous().reshape(-1, original_shape[-1])
    elif not x.is_contiguous():
        x = x.contiguous()
    if is_batch_invariant_mode_enabled():
        if (
            residual is not None
            or self.cast_x_before_out_mul
            or (self._fused_pad_kernel is not None and self.x_pad_to_multiple > 0)
        ):
            return self.forward_native(x, residual, post_residual_addition)
        out = rms_norm_batch_invariant(
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        if needs_reshape:
            out = out.reshape(original_shape)
        return out
    # Fused (add +) rmsnorm + zero-pad path. Triggered when caller
    # constructed RMSNorm with x_pad_to_multiple > 0. Output last
    # dim is padded up; residual_out stays at original width. Used
    # by callers (e.g. GPT-OSS MXFP4 MoE) whose immediate consumer
    # needs a padded hidden_size — folding the pad in here removes a
    # separate launch.
    if self._fused_pad_kernel is not None and self.x_pad_to_multiple > 0:
        if post_residual_addition is not None and residual is not None:
            residual = residual + post_residual_addition
        return self._fused_pad_kernel(
            x,
            self.weight.data,
            self.variance_epsilon,
            residual,
            self.x_pad_to_multiple,
        )
    if residual is not None:
        residual_out = torch.empty_like(x)
        output = torch.empty_like(x)
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        fused_add_rms_norm(
            output,
            x,
            residual,
            residual_out,
            self.weight.data,
            self.variance_epsilon,
        )
        return output, residual_out
    output = rms_norm(x, self.weight.data, self.variance_epsilon)
    if needs_reshape:
        output = output.reshape(original_shape)
    return output


def _RMSNorm__forward_cuda(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if x.numel() == 0:
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            return x, residual
        return x
    # sgl_kernel rmsnorm requires 2D input; reshape higher-rank tensors
    needs_reshape = x.dim() != 2 and residual is None
    if needs_reshape:
        original_shape = x.shape
        x = x.contiguous().reshape(-1, original_shape[-1])
    if self.variance_size_override is not None:
        return self.forward_native(x, residual, post_residual_addition)
    if is_batch_invariant_mode_enabled():
        server_args = get_global_server_args()
        if is_glm52_exact_mode(server_args) or is_qwen3_dense_exact_mode(server_args):
            return self._forward_xorl_batch_invariant(
                x,
                residual,
                post_residual_addition,
            )
        if residual is not None or self.cast_x_before_out_mul:
            return self.forward_native(x, residual, post_residual_addition)
        out = rms_norm_batch_invariant(
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        if needs_reshape:
            out = out.reshape(original_shape)
        return out
    if self.cast_x_before_out_mul and residual is None:
        # Use HF-semantics kernel (cast to dtype before weight multiply).
        if (
            _jit_rmsnorm_hf_available
            and x.dtype in (torch.float16, torch.bfloat16)
            and self.weight.data.dtype == x.dtype
            and is_supported_rmsnorm_hf_hidden_size(x.shape[-1])
        ):
            out = _jit_rmsnorm_hf(
                x.contiguous(), self.weight.data, self.variance_epsilon
            )
        else:
            # Fallback: pure-Python HF semantics (already implemented in forward_native).
            out = self.forward_native(x, None, None)
        if needs_reshape:
            out = out.reshape(original_shape)
        return out
    if residual is not None:
        if self.cast_x_before_out_mul:
            if (
                x.dtype in (torch.float16, torch.bfloat16)
                and self.weight.data.dtype == x.dtype
                and (
                    post_residual_addition is None
                    or post_residual_addition.dtype == x.dtype
                )
                and is_supported_jit_fused_add_rmsnorm_hidden_size(x.shape[-1])
            ):
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                _jit_fused_add_rmsnorm(
                    x,
                    residual,
                    self.weight.data,
                    self.variance_epsilon,
                    cast_x_before_out_mul=self.cast_x_before_out_mul,
                )
                return x, residual
            return self.forward_native(x, residual, post_residual_addition)
        # TODO: Ideally we want to have (hidden_states+residual)+post_residual_addition.
        # but right now we can only have hidden_states+(residual+post_residual_addition).
        # (hidden_states+residual)+post_residual_addition != hidden_states+(residual+post_residual_addition),
        # we probably need to add another parameter to fused_add_rmsnorm
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
        return x, residual
    out = rmsnorm(x, self.weight.data, self.variance_epsilon)
    if needs_reshape:
        out = out.reshape(original_shape)
    return out


def _RMSNorm__forward_hip(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # Fallback to native implementation if vllm is not available
    if not _has_vllm_rms_norm:
        return self.forward_native(x, residual, post_residual_addition)

    if is_batch_invariant_mode_enabled():
        if residual is not None or self.cast_x_before_out_mul:
            return self.forward_native(x, residual, post_residual_addition)
        return rms_norm_batch_invariant(
            x,
            self.weight.data,
            self.variance_epsilon,
        )

    if not x.is_contiguous():
        # NOTE: Remove this if aiter kernel supports discontinuous input
        x = x.contiguous()
    if residual is not None:
        out = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        fused_add_rms_norm(
            out, x, residual_out, residual, self.weight.data, self.variance_epsilon
        )
        return out, residual_out
    out = torch.empty_like(x)
    rms_norm(out, x, self.weight.data, self.variance_epsilon)
    return out


def _RMSNorm__forward_xpu(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if self.variance_size_override is not None:
        return self.forward_native(x, residual, post_residual_addition)
    if is_batch_invariant_mode_enabled():
        if residual is not None:
            return self.forward_native(x, residual, post_residual_addition)
        return rms_norm_batch_invariant(
            x,
            self.weight.data,
            self.variance_epsilon,
        )
    if residual is not None:
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
        return x, residual
    out = rmsnorm(x, self.weight.data, self.variance_epsilon)
    return out


def __apply_patch__(mod):
    # Publish the twin's top-level imports onto mod: in-tree they were the
    # srt file's own module globals, and rebound copies resolve via mod.
    mod.Literal = Literal
    mod.Optional = Optional
    mod.Tuple = Tuple
    mod.Union = Union
    # Deferred: the finder imports twins under bypass(), so sglang imports at
    # twin top level would cache modules UNPATCHED. Import here (bypass off)
    # and publish onto mod -- in-tree these were the file's module globals.
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
        is_batch_invariant_op_enabled,
    )
    from sglang.srt.server_args import (
        get_global_server_args,
        is_glm52_exact_mode,
        is_qwen3_dense_exact_mode,
    )
    from sglang.xorl.batch_invariant import resolve_or_validate_xorl_bi_family
    from sglang.xorl.bi import (
        RMS_NORM_FAMILIES,
        RMS_NORM_FAMILY_NO_RESIDUAL,
        RMSNormFamily,
        bi_fused_add_rms_norm,
        bi_rms_norm,
        rms_norm_v2,
    )

    # Publish the deferred imports onto mod: in-tree they were the srt
    # file's own module globals, and rebound copies resolve via mod.
    mod.is_batch_invariant_op_enabled = is_batch_invariant_op_enabled
    mod.get_global_server_args = get_global_server_args
    mod.is_glm52_exact_mode = is_glm52_exact_mode
    mod.is_qwen3_dense_exact_mode = is_qwen3_dense_exact_mode
    mod.resolve_or_validate_xorl_bi_family = resolve_or_validate_xorl_bi_family
    mod.RMS_NORM_FAMILIES = RMS_NORM_FAMILIES
    mod.RMS_NORM_FAMILY_NO_RESIDUAL = RMS_NORM_FAMILY_NO_RESIDUAL
    mod.RMSNormFamily = RMSNormFamily
    mod.bi_fused_add_rms_norm = bi_fused_add_rms_norm
    mod.bi_rms_norm = bi_rms_norm
    mod.rms_norm_v2 = rms_norm_v2
    mod._validate_qwen_v2_norm_tensor = rebind(_validate_qwen_v2_norm_tensor, mod)
    mod.RMSNorm._forward_xorl_batch_invariant = rebind(
        _RMSNorm___forward_xorl_batch_invariant,
        mod,
        name="_forward_xorl_batch_invariant",
    )
    mod.GemmaRMSNorm.__init__ = rebind(_GemmaRMSNorm____init__, mod, name="__init__")
    mod.GemmaRMSNorm.forward_cuda = rebind(
        _GemmaRMSNorm__forward_cuda, mod, name="forward_cuda"
    )
    mod.GemmaRMSNorm.forward_with_allreduce_fusion = rebind(
        _GemmaRMSNorm__forward_with_allreduce_fusion,
        mod,
        name="forward_with_allreduce_fusion",
    )
    mod.GemmaRMSNorm.forward_with_allreduce_fusion_quant_per_group = rebind(
        _GemmaRMSNorm__forward_with_allreduce_fusion_quant_per_group,
        mod,
        name="forward_with_allreduce_fusion_quant_per_group",
    )
    mod.RMSNorm.__init__ = rebind(_RMSNorm____init__, mod, name="__init__")
    mod.RMSNorm.forward_aiter = rebind(
        _RMSNorm__forward_aiter, mod, name="forward_aiter"
    )
    mod.RMSNorm.forward_cuda = rebind(_RMSNorm__forward_cuda, mod, name="forward_cuda")
    mod.RMSNorm.forward_hip = rebind(_RMSNorm__forward_hip, mod, name="forward_hip")
    mod.RMSNorm.forward_xpu = rebind(_RMSNorm__forward_xpu, mod, name="forward_xpu")
