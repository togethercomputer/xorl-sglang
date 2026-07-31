import logging
import os
from typing import Any, Callable, List, Literal

import torch

from sglang.srt.batch_invariant_ops import (
    RMS_NORM_FAMILY_NO_RESIDUAL,
    RMS_NORM_FAMILY_RESIDUAL_TREE,
    RMSNormFamily,
    bi_lm_head_full_logits,
    bi_lm_head_selected_logprob_from_logits,
    families_v2_enabled,
    head_v2_full_logits_with_lse,
    head_v2_selected_logprob_from_logits,
)

XorlBiFamily = Literal["v1", "v2"]
XorlGlm52NormSite = Literal["q_a", "kv_a", "input", "post_attention", "final"]
BI_FAMILIES_V2_SHA256 = (
    "fd4c5bac2a52d2148b8e4d0e9afa4e46e8c62689a68c2bc0e309f671597799e6"
)
_FAMILY_ENV_VARS = ("XORL_FAMILIES_V2", "SGLANG_FAMILIES_V2")
_FAMILY_ON = frozenset({"1", "true", "yes"})
_FAMILY_OFF = frozenset({"0", "false", "no"})
_REQUIRED_LEGACY_BI_OPS = ("addmm", "bmm", "log_softmax", "mean", "mm")
_TEMPLATE_OR_EAGER_PATHS = (
    "rmsnorm",
    "lm_head",
    "bi_router_gemm",
    "canonical_moe",
)
_REAL_SAMPLED_BOUNDARY = "sampler_score"
_REQUIRED_ENGAGEMENTS = (*_TEMPLATE_OR_EAGER_PATHS, _REAL_SAMPLED_BOUNDARY)
_CONTRACT_PLAN_LOGGED = False
_ENGAGEMENT_RECEIPT_LOGGED = False
_PIPELINE_STAGE_RECEIPT_LOGGED = False
_ENGAGEMENT_COUNTS = {component: 0 for component in _REQUIRED_ENGAGEMENTS}
logger = logging.getLogger(__name__)


def resolve_xorl_bi_family() -> XorlBiFamily:
    """Resolve one numerical family for every strict trainer/sampler site."""
    enabled = []
    raw_values = []
    for name in _FAMILY_ENV_VARS:
        raw = os.getenv(name)
        normalized = "1" if raw is None else raw.strip().lower()
        raw_values.append("<unset>" if raw is None else raw)
        if normalized in _FAMILY_ON:
            enabled.append(True)
        elif normalized in _FAMILY_OFF:
            enabled.append(False)
        else:
            raise RuntimeError(
                f"{name}={raw!r} is invalid for the XORL batch-invariant "
                "numerical contract; use 0/false/no or 1/true/yes."
            )

    if enabled[0] != enabled[1]:
        raise RuntimeError(
            "The XORL batch-invariant numerical family flags disagree: "
            f"XORL_FAMILIES_V2={raw_values[0]!r}, "
            f"SGLANG_FAMILIES_V2={raw_values[1]!r}. Set both to 1 for v2 "
            "or both to 0 for the paired v1 rollback."
        )

    shared_enabled = families_v2_enabled()
    if shared_enabled != enabled[0]:
        raise RuntimeError(
            "The strict XORL family resolver disagrees with the vendored "
            "families_v2_enabled() contract."
        )
    return "v2" if shared_enabled else "v1"


def resolve_or_validate_xorl_bi_family(family: str | None) -> XorlBiFamily:
    """Reject a caller override that disagrees with the process-wide family."""
    resolved = resolve_xorl_bi_family()
    if family is None:
        return resolved
    if family not in ("v1", "v2"):
        raise RuntimeError(f"Unknown XORL batch-invariant family: {family!r}.")
    if family != resolved:
        raise RuntimeError(
            "The requested XORL batch-invariant family disagrees with the "
            f"process-wide contract: requested={family!r}, resolved={resolved!r}."
        )
    return resolved


def log_xorl_bi_contract_plan_once(
    receipt_logger: logging.Logger,
    *,
    use_qk_norm: bool,
    speculative_decode: bool,
    mtp_decode: bool,
    legacy_bi_ops: tuple[str, ...],
    bi_router_enabled: bool,
) -> XorlBiFamily:
    """Validate and emit the serving plan before the first real forward."""
    global _CONTRACT_PLAN_LOGGED
    family = resolve_xorl_bi_family()
    validate_xorl_glm52_norm_envelope(use_qk_norm=use_qk_norm)
    if speculative_decode or mtp_decode:
        raise RuntimeError(
            "The XORL GLM-5.2 batch-invariant contract requires speculative "
            "and MTP decoding to be disabled."
        )
    if tuple(sorted(legacy_bi_ops)) != _REQUIRED_LEGACY_BI_OPS:
        raise RuntimeError(
            "The XORL GLM-5.2 batch-invariant contract requires exactly the "
            f"legacy BI ops {_REQUIRED_LEGACY_BI_OPS}, got "
            f"{tuple(sorted(legacy_bi_ops))}."
        )
    if not bi_router_enabled:
        raise RuntimeError(
            "The XORL GLM-5.2 batch-invariant contract requires SGLANG_BI_ROUTER=1."
        )
    if not _CONTRACT_PLAN_LOGGED:
        receipt_logger.info(
            "XORL batch-invariant numerical contract plan: family=%s "
            "vendor_sha256=%s serving_target=xorl "
            "resolved_use_qk_norm=false speculative_decode=false mtp_decode=false "
            "legacy_bi_ops=%s glm52_bi_router=true "
            "required_peer_trainer_rmsnorm_mode=sglang_fused "
            "required_engagements=%s",
            family,
            BI_FAMILIES_V2_SHA256,
            ",".join(_REQUIRED_LEGACY_BI_OPS),
            ",".join(_REQUIRED_ENGAGEMENTS),
        )
        _CONTRACT_PLAN_LOGGED = True
    return family


def record_xorl_bi_engagement(
    component: str,
    *,
    require_complete: bool = False,
    receipt_logger: logging.Logger = logger,
) -> None:
    """Record Python call-site observations without overstating graph replay.

    Eager forwards observe the trunk paths on the real request. CUDA-graph
    capture can instead observe those paths while building the replay template;
    replay does not re-enter this Python instrumentation. ``sampler_score`` is
    the separate observation that a real sampled boundary was reached.
    """
    global _ENGAGEMENT_RECEIPT_LOGGED
    if component not in _ENGAGEMENT_COUNTS:
        raise RuntimeError(f"Unknown XORL batch-invariant engagement: {component!r}.")
    _ENGAGEMENT_COUNTS[component] += 1
    missing_template_or_eager_path = tuple(
        name for name in _TEMPLATE_OR_EAGER_PATHS if _ENGAGEMENT_COUNTS[name] == 0
    )
    sampled_boundary_observed = _ENGAGEMENT_COUNTS[_REAL_SAMPLED_BOUNDARY] > 0
    if require_complete and (
        missing_template_or_eager_path or not sampled_boundary_observed
    ):
        raise RuntimeError(
            "The XORL GLM-5.2 real sampled boundary was reached before all "
            "numerical template-or-eager paths were observed; "
            f"missing_template_or_eager_path={missing_template_or_eager_path}, "
            f"real_sampled_boundary_observed={sampled_boundary_observed}, "
            "cuda_graph_replay_python_instrumentation=false."
        )
    if (
        not missing_template_or_eager_path
        and sampled_boundary_observed
        and not _ENGAGEMENT_RECEIPT_LOGGED
    ):
        counts = ",".join(
            f"{name}:{_ENGAGEMENT_COUNTS[name]}" for name in _REQUIRED_ENGAGEMENTS
        )
        receipt_logger.info(
            "XORL batch-invariant numerical observation receipt: "
            "template_or_eager_path_observed=%s "
            "real_sampled_boundary_observed=sampler_score "
            "cuda_graph_replay_python_instrumentation=false "
            "observation_counts=%s",
            ",".join(_TEMPLATE_OR_EAGER_PATHS),
            counts,
        )
        _ENGAGEMENT_RECEIPT_LOGGED = True


def record_xorl_glm52_pipeline_stage_receipt(
    *,
    pp_rank: int,
    start_layer: int,
    end_layer: int,
    moe_layer_count: int,
    receipt_logger: logging.Logger = logger,
) -> None:
    """Emit one fail-closed receipt for a non-final GLM-5.2 PP stage."""
    global _PIPELINE_STAGE_RECEIPT_LOGGED
    if _PIPELINE_STAGE_RECEIPT_LOGGED:
        return
    if pp_rank < 0 or start_layer < 0 or end_layer <= start_layer:
        raise RuntimeError(
            "Invalid GLM-5.2 pipeline-stage receipt range: "
            f"pp_rank={pp_rank}, start_layer={start_layer}, end_layer={end_layer}."
        )
    layer_count = end_layer - start_layer
    if moe_layer_count <= 0 or moe_layer_count > layer_count:
        raise RuntimeError(
            "Invalid GLM-5.2 pipeline-stage MoE count: "
            f"moe_layer_count={moe_layer_count}, layer_count={layer_count}."
        )
    expected = {
        "rmsnorm": 4 * layer_count,
        "lm_head": 0,
        "bi_router_gemm": moe_layer_count,
        "canonical_moe": 1,
        "sampler_score": 0,
    }
    actual = {name: _ENGAGEMENT_COUNTS[name] for name in _REQUIRED_ENGAGEMENTS}
    if actual != expected:
        raise RuntimeError(
            "GLM-5.2 non-final pipeline-stage observations disagree with the "
            f"stage contract: expected={expected}, actual={actual}."
        )
    counts = ",".join(f"{name}:{actual[name]}" for name in _REQUIRED_ENGAGEMENTS)
    receipt_logger.info(
        "XORL batch-invariant pipeline-stage observation receipt: "
        "pp_rank=%s stage_layer_range=%s:%s non_final_stage=true "
        "cuda_graph_replay_python_instrumentation=false observation_counts=%s",
        pp_rank,
        start_layer,
        end_layer,
        counts,
    )
    _PIPELINE_STAGE_RECEIPT_LOGGED = True


def validate_xorl_glm52_norm_envelope(*, use_qk_norm: bool) -> None:
    if use_qk_norm:
        raise RuntimeError(
            "The XORL GLM-5.2 batch-invariant contract does not certify "
            "use_qk_norm=True; the official checkpoint has no q/k norm helper."
        )


def xorl_glm52_norm_site_family(
    site: XorlGlm52NormSite,
    *,
    layer_id: int | None = None,
) -> RMSNormFamily:
    """Map each official GLM-5.2 RMSNorm site to its serving v1 family."""
    if site in ("q_a", "kv_a"):
        return RMS_NORM_FAMILY_NO_RESIDUAL
    if site == "input":
        if layer_id is None or layer_id < 0:
            raise ValueError("A GLM-5.2 input RMSNorm site requires layer_id >= 0.")
        return (
            RMS_NORM_FAMILY_NO_RESIDUAL
            if layer_id == 0
            else RMS_NORM_FAMILY_RESIDUAL_TREE
        )
    if site in ("post_attention", "final"):
        return RMS_NORM_FAMILY_RESIDUAL_TREE
    raise ValueError(f"Unknown GLM-5.2 RMSNorm site: {site!r}.")


def validate_xorl_bi_logit_transforms(
    logit_scale: float | None,
    final_logit_softcapping: float | None,
) -> None:
    if logit_scale is not None:
        raise RuntimeError(
            "The XORL batch-invariant LM-head contract does not support logit_scale."
        )
    if final_logit_softcapping is not None:
        raise RuntimeError(
            "The XORL batch-invariant LM-head contract does not support "
            "final_logit_softcapping."
        )


def xorl_bi_lm_head(
    hidden_states: torch.Tensor,
    lm_head: Any,
    *,
    use_fp32_lm_head: bool,
    embedding_bias: torch.Tensor | None = None,
    family: XorlBiFamily | None = None,
) -> torch.Tensor:
    if use_fp32_lm_head:
        raise RuntimeError(
            "The XORL batch-invariant target rejects --enable-fp32-lm-head: "
            "its contract consumes BF16 hidden states and BF16 weights directly "
            "and produces FP32 logits without an FP32-copy matmul."
        )
    if embedding_bias is not None:
        raise RuntimeError(
            "The XORL batch-invariant LM-head contract does not support an embedding bias."
        )
    if hasattr(lm_head, "set_lora") or hasattr(lm_head, "apply_lora"):
        raise RuntimeError(
            "The XORL batch-invariant LM-head contract does not support a LoRA-wrapped head."
        )
    if not hasattr(lm_head, "weight"):
        raise RuntimeError(
            "The XORL batch-invariant LM-head contract requires a dense BF16 weight."
        )
    if hidden_states.dtype != torch.bfloat16 or lm_head.weight.dtype != torch.bfloat16:
        raise RuntimeError(
            "The XORL batch-invariant LM-head contract requires BF16 hidden states "
            f"and BF16 weights, got {hidden_states.dtype} and {lm_head.weight.dtype}."
        )
    family = resolve_or_validate_xorl_bi_family(family)
    if family == "v2":
        logits, _ = head_v2_full_logits_with_lse(hidden_states, lm_head.weight)
        record_xorl_bi_engagement("lm_head")
        return logits
    logits = bi_lm_head_full_logits(hidden_states, lm_head.weight)
    record_xorl_bi_engagement("lm_head")
    return logits


def xorl_bi_sample_and_score(
    logits_output: Any,
    sampling_info: Any,
    *,
    return_logprob: bool,
    top_logprobs_nums: List[int],
    token_ids_logprobs: List[List[int]],
    positions: torch.Tensor,
    sample_from_logprobs: Callable[[torch.Tensor, Any, torch.Tensor], torch.Tensor],
    sync_token_ids: Callable[[torch.Tensor, Any], None],
    enable_deterministic: bool,
    return_original_logprob: bool,
    family: XorlBiFamily | None = None,
) -> torch.Tensor:
    """Sample and score under the strict public XORL T=1 contract."""
    family = resolve_or_validate_xorl_bi_family(family)
    logits = logits_output.next_token_logits
    if logits is None or logits.dtype != torch.float32:
        raise RuntimeError(
            "The XORL batch-invariant sampler requires FP32 logits from "
            "bi_lm_head_full_logits."
        )
    if not enable_deterministic or sampling_info.sampling_seed is None:
        raise RuntimeError(
            "The XORL batch-invariant sampler requires deterministic inference "
            "and a per-request sampling seed."
        )
    if sampling_info.is_all_greedy:
        raise RuntimeError(
            "The XORL batch-invariant sampler requires multinomial sampling, "
            "not greedy decoding."
        )
    if (
        sampling_info.need_top_p_sampling
        or sampling_info.need_top_k_sampling
        or sampling_info.need_min_p_sampling
    ):
        raise RuntimeError(
            "The XORL batch-invariant sampler does not support top-p, top-k, "
            "or min-p filtering."
        )
    if sampling_info.has_custom_logit_processor:
        raise RuntimeError(
            "The XORL batch-invariant sampler does not support custom logit processors."
        )
    if (
        sampling_info.acc_linear_penalties is not None
        or (
            sampling_info.penalizer_orchestrator is not None
            and sampling_info.penalizer_orchestrator.is_required
        )
        or sampling_info.vocab_mask is not None
        or sampling_info.logit_bias is not None
    ):
        raise RuntimeError(
            "The XORL batch-invariant sampler does not support penalties, "
            "grammar masks, or logit bias."
        )
    if return_original_logprob:
        raise RuntimeError(
            "The XORL batch-invariant sampler rejects SGLANG_RETURN_ORIGINAL_LOGPROB."
        )
    if any(x > 0 for x in top_logprobs_nums) or any(
        token_ids is not None for token_ids in token_ids_logprobs
    ):
        raise RuntimeError(
            "The XORL batch-invariant sampler only returns the sampled token logprob."
        )

    torch._assert_async(
        (sampling_info.temperatures == 1).all(),
        "The XORL batch-invariant sampler requires temperature == 1.",
    )
    torch._assert_async(
        torch.isfinite(logits).all(),
        "The XORL batch-invariant sampler requires finite logits.",
    )

    # Gumbel-max only depends on relative logits, so sampling directly from
    # the contract logits is identical to sampling from logits - logsumexp.
    batch_next_token_ids = sample_from_logprobs(logits, sampling_info, positions)
    sync_token_ids(batch_next_token_ids, sampling_info)

    if return_logprob:
        score = (
            head_v2_selected_logprob_from_logits
            if family == "v2"
            else bi_lm_head_selected_logprob_from_logits
        )
        selected_logprobs, _, _ = score(logits, batch_next_token_ids)
        logits_output.next_token_logprobs = selected_logprobs
        record_xorl_bi_engagement("sampler_score", require_complete=True)

    return batch_next_token_ids
