"""Shared helper: the MoE intermediate padding the TRT-LLM MoE runners apply.

Override-only module (no upstream twin), imported by the ``lora.layers`` and
``lora.mem_pool`` override twins so both sides agree on one definition.

Background — the bug this exists to fix:

``FusedMoE.__init__`` (``srt/layers/moe/fused_moe_triton/layer.py``) rounds
``intermediate_size_per_partition`` up to a multiple of 128 for the
flashinfer-TRT-LLM MoE family, because those kernels require it::

    if self.use_flashinfer_trtllm_moe and self.intermediate_size_per_partition % 128 != 0:
        self.intermediate_size_per_partition = round_up(self.intermediate_size_per_partition, 128)

``MoeRunnerBackend.is_flashinfer_trtllm()`` returns True for
``EXPERIMENTAL_SGL_TRTLLM`` as well ("shares the TRT-LLM FP8 kernels + layout,
so it inherits trtllm weight-prep here"), so the padding applies to
``--moe-runner-backend experimental_sgl_trtllm`` too.

The *base* weight loader handles that padding by sharding the checkpoint at the
**unpadded** width and copying into the leading slice of the padded buffer,
leaving the tail zero (``_load_w13`` / ``_load_w2``: "Derive the actual shard
size from the loaded weight so we index correctly" / "Copy into the leading
slice and leave the trailing padding as zeros").

The LoRA path did neither: it sharded LoRA weights at the *padded* width and
sized its pool buffers at the *unpadded* width. With Qwen3-30B-A3B
(``moe_intermediate_size=768``) at TP=4 that is a 256-wide shard of a 768-wide
tensor -- ranks 0-2 get 256 each and rank 3 gets an empty slice -- against a
192-wide buffer, which asserted as::

    LoRA buffer shape torch.Size([16, 192]) does not match weight shape torch.Size([16, 256])
    LoRA buffer shape torch.Size([16, 192]) does not match weight shape torch.Size([16, 0])

Only configurations where ``moe_intermediate_size // moe_tp_size`` is already a
multiple of 128 escaped it (768/2 = 384 works; 768/4 = 192 does not), which is
why no existing test caught it -- every LoRA e2e test pins
``MOE_RUNNER_BACKEND = "triton"``.
"""

from sglang.srt.layers.moe.utils import get_moe_runner_backend

MOE_INTERMEDIATE_ALIGNMENT = 128


def trtllm_moe_pads_intermediate() -> bool:
    """Whether the ACTIVE MoE runner pads the per-partition intermediate size.

    Mirrors ``FusedMoE.use_flashinfer_trtllm_moe`` exactly; keep in sync.
    """
    backend = get_moe_runner_backend()
    return backend.is_flashinfer_trtllm() or backend.is_flashinfer_trtllm_routed()


def padded_moe_inter(per_partition: int) -> int:
    """Per-partition intermediate size as the MoE *layer* will allocate it.

    Returns ``per_partition`` unchanged when the active runner does not pad, so
    non-TRT-LLM backends keep byte-identical behaviour.
    """
    if per_partition % MOE_INTERMEDIATE_ALIGNMENT == 0:
        return per_partition
    if not trtllm_moe_pads_intermediate():
        return per_partition
    return (
        (per_partition + MOE_INTERMEDIATE_ALIGNMENT - 1)
        // MOE_INTERMEDIATE_ALIGNMENT
        * MOE_INTERMEDIATE_ALIGNMENT
    )
