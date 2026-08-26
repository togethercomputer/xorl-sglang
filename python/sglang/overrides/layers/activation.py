"""Overlay twin: exact FP32 SwiGLU under a resolved XoRL contract.

Ported from xorl-sglang `main` (c08786bd3). Upstream falls back to
``forward_native`` whenever ``rl_on_policy_target`` is set; the exact
contracts instead select ``fp32_silu_and_mul`` -- one FP32 rounding chain
with a pinned launch geometry (see sglang.xorl.bi.bi_silu_and_mul) -- so the
trainer can reproduce the activation bytes.

Non-exact servers (rl_on_policy_target set but no architecture contract
resolved, e.g. main's retired "fsdp" target) keep upstream's forward_native
selection.
"""


def __apply_patch__(mod):
    SiluAndMul = mod.SiluAndMul
    orig_init = SiluAndMul.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        from sglang.srt.runtime_context import get_server_args
        from sglang.srt.server_args import is_xorl_exact_mode

        if is_xorl_exact_mode(get_server_args()):
            self._forward_method = self.forward_exact

    def forward_exact(self, x):
        from sglang.xorl.bi.bi_silu_and_mul import fp32_silu_and_mul  # noqa: PLC0415

        return fp32_silu_and_mul(x)

    SiluAndMul.__init__ = __init__
    SiluAndMul.forward_exact = forward_exact
