# Compatibility alias: the XoRL trainer's cross-engine gates import this
# module at its main-branch home (`sglang.srt.batch_invariant_ops
# .bi_families_v2`, see xorl tests/models/test_rmsnorm_family_cross_engine.py).
# On this dev-based branch the implementation lives in sglang.xorl.bi;
# this file is additive (upstream has no module of this name), so it carries
# no upstream-sync cost.

from sglang.xorl.bi.bi_families_v2 import *  # noqa: F401,F403
from sglang.xorl.bi.bi_families_v2 import (  # noqa: F401
    HEAD_V2_BLOCK_K,
    HEAD_V2_STATS_TILE_N,
)
