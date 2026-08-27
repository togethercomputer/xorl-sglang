"""Overlay twin: exact FP32 SwiGLU under a resolved XoRL contract.

Ported from xorl-sglang `main` (c08786bd3). Main REPLACES upstream's
``SiluAndMul.__init__`` selection: upstream forces ``forward_native``
whenever ``rl_on_policy_target`` is set (any value), while the exact
contracts select ``fp32_silu_and_mul`` -- one FP32 rounding chain with a
pinned launch geometry (see sglang.xorl.bi.bi_silu_and_mul) -- and a target
WITHOUT a resolved architecture contract keeps the ordinary platform
dispatch (no forward_native downgrade).

``get_server_args`` / ``is_xorl_exact_mode`` are installed as module-level
attributes on the public module and looked up through it at call time,
matching main's import surface (the contract tests patch
``sglang.srt.layers.activation.get_server_args``).
"""

from sglang.srt.runtime_context import get_server_args
from sglang.srt.server_args import is_xorl_exact_mode


def __apply_patch__(mod):
    mod.get_server_args = get_server_args
    mod.is_xorl_exact_mode = is_xorl_exact_mode

    SiluAndMul = mod.SiluAndMul
    base_cls = SiluAndMul.__mro__[1]

    def __init__(self, *args, **kwargs):
        base_cls.__init__(self, *args, **kwargs)
        if mod.is_xorl_exact_mode(mod.get_server_args()):
            self._forward_method = self.forward_exact
        elif mod._use_aiter and mod.envs.SGLANG_OPT_USE_AITER_SILU_MUL.get():
            self._forward_method = self.forward_aiter

    def forward_exact(self, x):
        from sglang.xorl.bi.bi_silu_and_mul import fp32_silu_and_mul  # noqa: PLC0415

        return fp32_silu_and_mul(x)

    SiluAndMul.__init__ = __init__
    SiluAndMul.forward_exact = forward_exact
