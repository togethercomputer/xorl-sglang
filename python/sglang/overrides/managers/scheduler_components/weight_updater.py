"""Overlay twin: mandatory radix-cache flush on weight sync under exact RL.

Ported from xorl-sglang `main` (c08786bd3). With the radix cache enabled
under ``rl_on_policy_target``, a weight update whose client passes
``flush_cache=False`` (the xorl API's ``cache_invalidation_mode="auto"``
does exactly that for BF16-KV endpoints) used to leave pre-update KV in the
tree. The next prefix hit then spliced stale-policy bytes into new-policy
decision logprobs -- nonzero K3 with no error anywhere. The engine owns the
numerical contract, so the flush must not depend on the client remembering
the flag.

Main's two-phase prepare/complete weight-update plumbing in the same file is
NOT ported (this branch serves the legacy single-phase
/update_weights_from_distributed path).
"""

import logging

logger = logging.getLogger(__name__)


def _radix_flush_required_for_rl() -> bool:
    """Whether the exact-RL contract mandates a prefix-cache flush on weight sync."""
    from sglang.srt.runtime_context import get_exec, get_memory

    return (
        get_exec().deterministic.rl_on_policy_target is not None
        and not get_memory().disable_radix_cache
    )


def __apply_patch__(mod):
    mod._radix_flush_required_for_rl = _radix_flush_required_for_rl
    manager_cls = mod.SchedulerWeightUpdaterManager

    def flush_cache_after_weight_update(self, recv_req) -> None:
        force_flush = not recv_req.flush_cache and _radix_flush_required_for_rl()
        if force_flush:
            logger.warning(
                "Forcing a prefix-cache flush after this weight update: the "
                "radix cache is enabled under rl_on_policy_target, and prefix "
                "reuse across a policy update would replay KV computed by the "
                "previous weights (stale-policy behavior logprobs). The "
                "client's flush_cache=False is overridden."
            )
        if recv_req.flush_cache or force_flush:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            if force_flush and not flush_cache_success:
                raise RuntimeError(
                    "Mandatory prefix-cache flush after a weight update failed "
                    "(requests still in flight?). Refusing to continue: serving "
                    "from the pre-update radix tree would return stale-policy "
                    "logprobs with no error."
                )
            assert flush_cache_success, "Cache flush failed after updating weights"

    manager_cls.flush_cache_after_weight_update = flush_cache_after_weight_update
