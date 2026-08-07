# Exact GLM-5.2 zero-K3 serving

`--rl-on-policy-target xorl` on the official GLM-5.2 architecture engages the
complete exact serving program automatically. There are no component flags.

The admitted program uses TP16/EP16/CP16/PP1, sparse FlashMLA attention,
model-specific grouped routing, native FP8 experts, and a canonical ordered
fold after every one of the 75 MoE blocks. Sparse selector keys are recomputed
from model state; selected-key replay is not part of the contract. The
architecture resolver also owns the graph bucket, admission width, RoPE table
provenance, norm/head kernels, and canonical transport.

The current serving envelope is one 16-rank replica across two 8-GPU nodes,
CUDA graph bucket `[16]`, global admission 16, radix disabled, overlap disabled,
and the official 78-block geometry (3 dense plus 75 MoE). Requests outside the
prebuilt RoPE capacity fail rather than rebuilding the table through an
unqualified growth path. Incompatible geometry, topology, precision, backend,
graph/cache setting, or sampling transform fails before returning behavior
logprobs.

Pair it with the XORL trainer's official WORLD16/PP1/TP1/DP1/EP16/CP16,
Ring1/Ulysses16 server-training program. A revision pair is qualified only by a
fresh repeatable sampler capture and full 78-block teacher-forced replay with
identical retained IDs, identical raw float32 logprob bytes, and K3 exactly
zero. Development-lineage evidence does not qualify a rewritten public pair by
ancestry.
