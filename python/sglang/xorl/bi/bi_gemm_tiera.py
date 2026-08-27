"""Shape-keyed launch configs for the batch-invariant GEMM family.

Byte-free launch-config headroom selected by the architecture resolver —
default OFF (=0): exactly today's behavior, byte-for-byte AND config-for-config
(every lookup returns None, so no call site changes its launch). =1: the tables
below are consulted first at three sites, all of which launch
``matmul_kernel_persistent`` / ``bmm_kernel_persistent``:

  1. ``bi_router_gemm`` — today pinned to ``_BI_ROUTER_GEMM_CONFIG``
     (BM128/BN128), which at the decode/serving shapes (gathered M<=256,
     N=num_experts=256 on Qwen3.6) launches 4 CTAs on a 132-SM H100.
  2. ``_launch_with_config_fallback`` — the mm/addmm/lm-head dispatch. Tier-A
     entries are consulted before the R1 table (``bi_gemm_configs``); shapes
     without a Tier-A entry fall through to the stock path unchanged.
  3. ``bmm_batch_invariant`` — today a hardcoded per-dtype config dict that
     ignores every shape-keyed table. The Tier-A bmm table ships empty because
     the supported Qwen decode path does not invoke ``bmm_kernel_persistent``;
     the routing support remains for future admitted shapes.

Byte-safety doctrine (imported from GLM Tier-A / R1): in this kernel family
each output element is produced by exactly ONE CTA whose K-reduction is a
serial ``for ki in range(k_tiles)`` chain of ``tl.dot`` calls — per-CTA
serial-K, no split-K. BLOCK_SIZE_M / BLOCK_SIZE_N tile the OUTPUT space,
GROUP_SIZE_M permutes tile visitation order, num_warps splits the tile across
M/N (never K), and num_stages deepens the load pipeline — none of them enter
the per-element accumulation chain. BLOCK_SIZE_K is the ONLY bit-relevant axis
and is pinned per dtype here (never present in a table entry; appended at
lookup from ``_PINNED_BLOCK_K``). Split-K is Tier-C (moves bytes) and is
rejected. Every table entry below was admitted only after torch.equal against
the flag-off oracle across the pre-registered M lattice x 6 value patterns
(incl. denormal, large-magnitude, and inf/NaN cells).
"""

# Internal implementation state: applied once by the exact Qwen3.5-family
# architecture resolver
# (see sglang.xorl.fla.qwen35_gdn_exact); no per-feature
# environment variable. False keeps every other lane on the pinned configs.
_ENABLED = False


def is_tiera_enabled() -> bool:
    return _ENABLED


def set_tiera_enabled(enabled: bool) -> None:
    """Set by the private architecture resolver before model construction."""
    global _ENABLED
    _ENABLED = bool(enabled)


# The bit-relevant axis: pinned per dtype, NEVER shape-keyed, NEVER in a table
# entry. Identical to bi_gemm_configs.PINNED_BLOCK_K and to the hardcoded
# per-dtype BLOCK_SIZE_K in bmm_batch_invariant (asserted at import below).
_PINNED_BLOCK_K = {
    "torch.bfloat16": 64,
    "torch.float16": 64,
    "torch.float32": 32,
}

# --------------------------------------------------------------------------- #
# Tables: {(dtype_str, K, N): ((max_M, cfg), ...)}. Buckets scanned in order;
# first max_M >= M wins; cfg=None or exhausting the tuple means "no Tier-A
# opinion" -> caller falls through to its stock path (pinned router config /
# R1 lookup / hardcoded bmm dict). M > last bucket always falls through, so
# the extend/large-M regime is untouched by construction.
# Entries are generated from a bitwise config sweep; do not hand-edit them.
# --------------------------------------------------------------------------- #

# BEGIN GENERATED TABLES
# Generated on H100 (132 SMs) with Triton 3.7.1 and torch 2.12.1+cu132.
# Every entry passed the full bucket M lattice x
# 6 value patterns, bit-view torch.equal vs the flag-off oracle) + cross-M
# row-invariance probe.
ROUTER_TABLE = {
    # bi_router_gemm, Qwen3.6 decode: M = DP-gathered tokens (<=256 at graph-32 x
    # DP8), K = hidden 2048, N = num_experts 256. Pinned BM128/BN128 launched 4
    # CTAs/132 SMs; these tile the output ~128 ways. Graph-replay medians at
    # hot M: 12.8 -> 7.1 us (M=256), 12.2 -> 6.8 (M<=64), 12.8 -> 6.9 (M<=16).
    ("torch.bfloat16", 2048, 256): (
        (
            16,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "GROUP_SIZE_M": 8,
                "num_stages": 3,
                "num_warps": 4,
            },
        ),
        (
            64,
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 16,
                "GROUP_SIZE_M": 1,
                "num_stages": 3,
                "num_warps": 4,
            },
        ),
        (
            256,
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 16,
                "GROUP_SIZE_M": 8,
                "num_stages": 3,
                "num_warps": 4,
            },
        ),
        # M > 256 (extend regime): fall through to the pinned contract config.
    ),
}

MM_TABLE = {
    # shared_expert_gate (q36: Linear(2048 -> 1), DeepGEMM-rejected N<16, runs
    # per MoE layer per step on the gathered tokens). Graph-replay medians at
    # hot M: 10.3 -> 6.1 us (M=256), 8.3 -> 5.9 (M<=64), 11.5 -> 6.5 (M<=16).
    ("torch.bfloat16", 2048, 1): (
        (
            16,
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 16,
                "GROUP_SIZE_M": 1,
                "num_stages": 6,
                "num_warps": 4,
            },
        ),
        (
            64,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
                "GROUP_SIZE_M": 1,
                "num_stages": 5,
                "num_warps": 4,
            },
        ),
        (
            256,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
                "GROUP_SIZE_M": 1,
                "num_stages": 5,
                "num_warps": 4,
            },
        ),
        # M > 256: fall through to the R1 lookup (class defaults / baseline).
    ),
    # Qwen3.6 decode lm-head full-logits (bi_lm_head_full_logits: M = local decode
    # tokens <= 32, N = full vocab 248320, fp32 out). Bandwidth-bound (1.02 GB
    # weight read); BM64 -> BM32 stops wasting half of every M-tile at M=32.
    # Graph-replay median at M=32: 370.8 -> 348.7 us (+6%).
    ("torch.bfloat16", 2048, 248320): (
        (
            32,
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 128,
                "GROUP_SIZE_M": 8,
                "num_stages": 5,
                "num_warps": 4,
            },
        ),
        # M > 32: fall through to the R1 lookup.
    ),
}

# Empty on purpose: this supported decode path has no
# bmm_kernel_persistent launches, so
# there is no production shape to tune. The routing fix above makes shape-keyed
# selection REACH bmm for whoever hits it next; entries require the same
# certification gate as everything else.
BMM_TABLE = {}
# END GENERATED TABLES


def _lookup(table, site, dtype, M, N, K):
    if not _ENABLED:
        return None
    key = (str(dtype), K, N)
    buckets = table.get(key)
    if buckets is None:
        return None
    for max_m, cfg in buckets:
        if max_m is None or M <= max_m:
            if cfg is None:
                return None
            return dict(cfg, BLOCK_SIZE_K=_PINNED_BLOCK_K[str(dtype)])
    return None


def lookup_tiera_router_config(dtype, M: int, N: int, K: int):
    """Tier-A config for ``bi_router_gemm`` at (dtype, M, N, K), or None."""
    return _lookup(ROUTER_TABLE, "router", dtype, M, N, K)


def lookup_tiera_mm_config(dtype, M: int, N: int, K: int, out_itemsize=None):
    """Tier-A config for the persistent mm/addmm/lm-head dispatch, or None.

    Keys are (dtype, K, N) like the R1 table; entries were certified at their
    production out dtype (fp32-out for the lm-head/router-class shapes), so no
    smem re-model is applied here — launch failures fall back at the call site.
    """
    return _lookup(MM_TABLE, "mm", dtype, M, N, K)


def lookup_tiera_bmm_config(dtype, B: int, M: int, N: int, K: int):
    """Tier-A config for ``bmm_batch_invariant``, or None (stock dict)."""
    return _lookup(BMM_TABLE, "bmm", dtype, M, N, K)
