"""Overlay twin: resolve XoRL exact runtime before model construction.

Ported from xorl-sglang `main` (c08786bd3):

- ``maybe_apply_xorl_exact_runtime`` installs the architecture-owned exact
  kernel tuple (Qwen3.5 GDN) once per worker. Main calls it from
  ``ModelRunner.__init__`` immediately before ``load_model``; here the twin
  wraps ``load_model`` to run it first, which is the same point in the
  lifecycle without copying ``__init__``.
- ``maybe_enable_batch_invariant_mode`` is replaced wholesale (it is a short
  self-contained method): under a resolved exact contract it interposes
  exactly the aten op set the architecture owns via
  ``enable_batch_invariant_mode(ops=...)`` instead of the full generic set.

GLM-5.2 branches from main are dropped: that contract is not ported.
"""

import logging

logger = logging.getLogger(__name__)


def __apply_patch__(mod):
    ModelRunner = mod.ModelRunner
    orig_load_model = ModelRunner.load_model

    def maybe_apply_xorl_exact_runtime(self):
        """Resolve architecture-owned kernels before model construction."""
        from sglang.srt.server_args import is_qwen35_gdn_exact_mode

        if is_qwen35_gdn_exact_mode(self.server_args):
            from sglang.xorl.fla.qwen35_gdn_exact import _apply_qwen35_gdn_exact

            _apply_qwen35_gdn_exact(self.server_args)

    def load_model(self):
        self.maybe_apply_xorl_exact_runtime()
        return orig_load_model(self)

    def maybe_enable_batch_invariant_mode(self):
        from sglang.srt.runtime_context import get_exec
        from sglang.srt.server_args import _exact_batch_invariant_ops

        exact_ops = _exact_batch_invariant_ops(self.server_args)
        if exact_ops is not None:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            enable_batch_invariant_mode(ops=exact_ops)
        elif get_exec().deterministic.enable_deterministic_inference:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            enable_batch_invariant_mode()

    ModelRunner.maybe_apply_xorl_exact_runtime = maybe_apply_xorl_exact_runtime
    ModelRunner.load_model = load_model
    ModelRunner.maybe_enable_batch_invariant_mode = maybe_enable_batch_invariant_mode
