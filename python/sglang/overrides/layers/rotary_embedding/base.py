"""Override twin of ``sglang.srt.layers.rotary_embedding.base`` -- xorl exact serving (zero-srt port of PR #41).

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

import contextlib

from sglang.overrides._twin_bind import rebind

_QWEN35_CLASS_B_ACCUMULATED_RECOMPILE_LIMIT = 8192

_QWEN35_CLASS_B_RECOMPILE_LIMIT = 2048


def _pin_qwen35_class_b_compile_budget() -> None:
    """Prevent an exact Class-B run from silently falling back to eager Class A."""
    config = torch._dynamo.config
    config.recompile_limit = max(
        getattr(config, "recompile_limit", 0), _QWEN35_CLASS_B_RECOMPILE_LIMIT
    )
    config.accumulated_recompile_limit = max(
        getattr(config, "accumulated_recompile_limit", 0),
        _QWEN35_CLASS_B_ACCUMULATED_RECOMPILE_LIMIT,
    )
    if hasattr(config, "fail_on_recompile_limit_hit"):
        config.fail_on_recompile_limit_hit = True


def _RotaryEmbedding___build_cos_sin_cache(self) -> torch.Tensor:
    """Build the table once under the selected architecture provenance."""
    pin = self._cos_sin_cache_device()
    with self._cos_sin_cache_pin():
        cache = self._compute_cos_sin_cache()
    if pin is None:
        return cache
    if cache.device.type != pin.type:
        raise RuntimeError(
            f"{type(self).__name__} evaluated its cos/sin table on "
            f"{cache.device}, but the table is pinned to {pin}. Build it on "
            "the selected provenance device or override the shared "
            "_cos_sin_cache_* hooks."
        )
    return cache.to(device=self._cos_sin_cache_out_device())


def _RotaryEmbedding___cos_sin_cache_device(self) -> Optional[torch.device]:
    """Device on which the full cos/sin table must be evaluated.

    GLM-5.2 uses its certified split recipe: inverse frequencies are
    computed on CPU, then the outer product and cos/sin run on the ambient
    CUDA device. Other RL targets, including Qwen3.5-family exact serving,
    evaluate the complete table on CPU.
    """
    deterministic = get_exec().deterministic
    if deterministic.glm52_exact_mode:
        return None
    if deterministic.rl_on_policy_target is not None:
        return torch.device("cpu")
    return None


def _RotaryEmbedding___cos_sin_cache_extra_positions(
    self, start: int, stop: int
) -> torch.Tensor:
    """Positions appended during non-exact cache growth."""
    return torch.arange(start, stop, dtype=torch.float)


def _RotaryEmbedding___cos_sin_cache_inv_freq(self) -> torch.Tensor:
    """Inverse frequencies used by both initial construction and growth."""
    return self._compute_inv_freq(self.base)


def _RotaryEmbedding___cos_sin_cache_mscale(self) -> float:
    """Magnitude scale applied to both cos and sin rows."""
    return 1.0


def _RotaryEmbedding___cos_sin_cache_out_device(self) -> torch.device:
    """Device on which a table built under a provenance pin is stored."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _RotaryEmbedding___cos_sin_cache_pin(self):
    """Install the table provenance device as the ambient default."""
    pin = self._cos_sin_cache_device()
    return contextlib.nullcontext() if pin is None else torch.device(pin)


def _RotaryEmbedding___cos_sin_cache_positions(self) -> torch.Tensor:
    """Positions covered by the initial table."""
    return torch.arange(self.max_position_embeddings, dtype=torch.float)


def _RotaryEmbedding___cos_sin_cache_rows(
    self, positions: torch.Tensor, inv_freq: torch.Tensor
) -> torch.Tensor:
    """Evaluate cos/sin rows using the shared construction recipe."""
    freqs = torch.einsum("i,j -> ij", positions, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    mscale = self._cos_sin_cache_mscale()
    if mscale != 1.0:
        cos = cos * mscale
        sin = sin * mscale
    return torch.cat((cos, sin), dim=-1)


def _RotaryEmbedding___cos_sin_cache_work_device(
    self, default: Union[torch.device, str, None]
) -> Union[torch.device, str, None]:
    """Return the pinned provenance device, or ``default`` when unpinned."""
    pin = self._cos_sin_cache_device()
    return default if pin is None else pin


def _RotaryEmbedding____init__(
    self,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: int,
    is_neox_style: bool,
    dtype: torch.dtype,
) -> None:
    super(RotaryEmbedding, self).__init__()
    self.head_size = head_size
    self.rotary_dim = rotary_dim
    self.max_position_embeddings = max_position_embeddings
    self.base = base
    self.is_neox_style = is_neox_style
    self.dtype = dtype

    cache = self._build_cos_sin_cache()
    # NOTE(ByronHsu): cache needs to be in FP32 for numerical stability.
    if not (_is_cuda or _is_xpu or envs.SGLANG_ROPE_CACHE_FP32.get()):
        cache = cache.to(dtype)

    if (
        (not (_is_cuda) or self.head_size not in [64, 128, 256, 512])
        and not (_is_cpu)
        and not (_is_xpu)
        and not (_is_npu)
        and not (_is_musa)
        and not (_is_mps)
        and not (current_platform.is_out_of_tree())
    ):
        # rotary_embedding from sglang.kernels.ops.attention.rope and vllm._custom_ops has the same implementation.
        # TODO: Test on different devices and remove this conditional.
        if _is_cuda:
            from sglang.kernels.ops.attention.rope import rotary_embedding
        elif _is_hip:
            from sgl_kernel import rotary_embedding
        else:
            from vllm._custom_ops import rotary_embedding

        self.use_fallback_kernel = True
        self.fallback_rotary_embedding = rotary_embedding
    else:
        self.use_fallback_kernel = False

    self.cos_sin_cache: torch.Tensor
    self.register_buffer("cos_sin_cache", cache, persistent=False)

    self._apply_rotary_emb_wrapped = apply_rotary_emb

    # XXX (MUSA): Implement sgl_kernel.rotary_embedding support for MUSA backend
    deterministic = get_exec().deterministic
    if deterministic.rl_on_policy_target is not None or _is_musa:
        self._forward_method = self.forward_native
        # Both GLM-5.2 and exact Qwen3.5-family serving use the compiled
        # Class-B expression selected by their architecture contracts.
        if not deterministic.qwen35_gdn_exact_mode or getattr(
            deterministic, "qwen35_rope_class_b", False
        ):
            self._apply_rotary_emb_wrapped = torch.compile(
                dynamic=True,
                disable=_is_npu,
            )(apply_rotary_emb)
    if getattr(deterministic, "qwen35_rope_class_b", False):
        if not deterministic.qwen35_gdn_exact_mode:
            raise RuntimeError("Qwen3.5-family Class-B RoPE requires exact Qwen mode")
        if not _is_cuda:
            raise RuntimeError("Qwen3.5-family Class-B RoPE is qualified only on CUDA")
        if dtype is not torch.bfloat16 or not is_neox_style:
            raise RuntimeError(
                "Qwen3.5-family Class-B RoPE requires BF16 and the "
                "Neox half-split feature layout"
            )
        _pin_qwen35_class_b_compile_budget()
        logger.info(
            "Qwen3.5-family Class-B RoPE runtime engaged: "
            "CPU-built fp32 table + compiled fp32-chain application; "
            "recompile_limit=%s accumulated_recompile_limit=%s "
            "fail_on_recompile_limit_hit=%s",
            torch._dynamo.config.recompile_limit,
            torch._dynamo.config.accumulated_recompile_limit,
            getattr(torch._dynamo.config, "fail_on_recompile_limit_hit", None),
        )
    self.position_cos, self.position_sin = None, None


def _RotaryEmbedding___compute_cos_sin_cache(self) -> torch.Tensor:
    """Compute the initial table through the shared recipe hooks."""
    return self._cos_sin_cache_rows(
        self._cos_sin_cache_positions(), self._cos_sin_cache_inv_freq()
    )


def _RotaryEmbedding___compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
    """Compute the inverse frequency."""
    # NOTE(woosuk): To exactly match the HF implementation, we need to
    # use CPU to compute the cache and then move it to GPU. However, we
    # create the cache on GPU for faster initialization. This may cause
    # a slight numerical difference between the HF implementation and ours.
    deterministic = get_exec().deterministic
    init_device = (
        torch.device("cpu")
        if deterministic.glm52_exact_mode
        else self._cos_sin_cache_device()
    )
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=init_device)
            / self.rotary_dim
        )
    )
    if deterministic.glm52_exact_mode:
        inv_freq = inv_freq.to(device=self._cos_sin_cache_out_device())
    return inv_freq


def _RotaryEmbedding___ensure_cos_sin_cache_length(self, needed_max_pos: int):
    """Ensure cos_sin_cache length > needed_max_pos."""
    cur_len = int(self.cos_sin_cache.shape[0])
    if needed_max_pos < cur_len:
        return
    if getattr(self, "glm52_exact_prebuilt_only", False):
        raise RuntimeError(
            "exact GLM-5.2 mode requires positions within the prebuilt "
            f"RoPE cache (len {cur_len}); position {needed_max_pos} is "
            "outside the certified envelope"
        )

    # Align to reduce realloc frequency
    align = envs.SGLANG_ROPE_CACHE_ALIGN.get()
    new_len = ((needed_max_pos + align) // align) * align
    device = self.cos_sin_cache.device
    dtype = self.cos_sin_cache.dtype

    # Growth outside exact GLM mode must use the same frequencies,
    # magnitude scale, and provenance as the initial table.
    compute_device = self._cos_sin_cache_work_device(device)
    with self._cos_sin_cache_pin():
        inv_freq = self._cos_sin_cache_inv_freq().to(device=compute_device)
        t_new = self._cos_sin_cache_extra_positions(cur_len, new_len).to(
            device=compute_device
        )
        if t_new.numel() == 0:
            return
        new_rows = self._cos_sin_cache_rows(t_new, inv_freq)
    new_rows = new_rows.to(dtype=dtype, device=device)

    # Update cache with new rows
    self.cos_sin_cache = torch.cat((self.cos_sin_cache, new_rows), dim=0).to(
        device=device, dtype=dtype
    )


def __apply_patch__(mod):
    # Publish the twin's top-level imports onto mod: in-tree they were the
    # srt file's own module globals, and rebound copies resolve via mod.
    mod.contextlib = contextlib
    mod._QWEN35_CLASS_B_ACCUMULATED_RECOMPILE_LIMIT = (
        _QWEN35_CLASS_B_ACCUMULATED_RECOMPILE_LIMIT
    )
    mod._QWEN35_CLASS_B_RECOMPILE_LIMIT = _QWEN35_CLASS_B_RECOMPILE_LIMIT
    mod._pin_qwen35_class_b_compile_budget = rebind(
        _pin_qwen35_class_b_compile_budget, mod
    )
    mod.RotaryEmbedding._build_cos_sin_cache = rebind(
        _RotaryEmbedding___build_cos_sin_cache, mod, name="_build_cos_sin_cache"
    )
    mod.RotaryEmbedding._cos_sin_cache_device = rebind(
        _RotaryEmbedding___cos_sin_cache_device, mod, name="_cos_sin_cache_device"
    )
    mod.RotaryEmbedding._cos_sin_cache_extra_positions = rebind(
        _RotaryEmbedding___cos_sin_cache_extra_positions,
        mod,
        name="_cos_sin_cache_extra_positions",
    )
    mod.RotaryEmbedding._cos_sin_cache_inv_freq = rebind(
        _RotaryEmbedding___cos_sin_cache_inv_freq, mod, name="_cos_sin_cache_inv_freq"
    )
    mod.RotaryEmbedding._cos_sin_cache_mscale = rebind(
        _RotaryEmbedding___cos_sin_cache_mscale, mod, name="_cos_sin_cache_mscale"
    )
    mod.RotaryEmbedding._cos_sin_cache_out_device = rebind(
        _RotaryEmbedding___cos_sin_cache_out_device,
        mod,
        name="_cos_sin_cache_out_device",
    )
    mod.RotaryEmbedding._cos_sin_cache_pin = rebind(
        _RotaryEmbedding___cos_sin_cache_pin, mod, name="_cos_sin_cache_pin"
    )
    mod.RotaryEmbedding._cos_sin_cache_positions = rebind(
        _RotaryEmbedding___cos_sin_cache_positions, mod, name="_cos_sin_cache_positions"
    )
    mod.RotaryEmbedding._cos_sin_cache_rows = rebind(
        _RotaryEmbedding___cos_sin_cache_rows, mod, name="_cos_sin_cache_rows"
    )
    mod.RotaryEmbedding._cos_sin_cache_work_device = rebind(
        _RotaryEmbedding___cos_sin_cache_work_device,
        mod,
        name="_cos_sin_cache_work_device",
    )
    mod.RotaryEmbedding.__init__ = rebind(
        _RotaryEmbedding____init__, mod, name="__init__"
    )
    mod.RotaryEmbedding._compute_cos_sin_cache = rebind(
        _RotaryEmbedding___compute_cos_sin_cache, mod, name="_compute_cos_sin_cache"
    )
    mod.RotaryEmbedding._compute_inv_freq = rebind(
        _RotaryEmbedding___compute_inv_freq, mod, name="_compute_inv_freq"
    )
    mod.RotaryEmbedding._ensure_cos_sin_cache_length = rebind(
        _RotaryEmbedding___ensure_cos_sin_cache_length,
        mod,
        name="_ensure_cos_sin_cache_length",
    )
