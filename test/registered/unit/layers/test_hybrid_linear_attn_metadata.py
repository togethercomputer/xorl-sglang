import unittest

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class _ReqToTokenPoolStub:
    def __init__(self, mamba_indices: torch.Tensor):
        self.mamba_indices = mamba_indices

    def get_mamba_indices(self, req_pool_indices: torch.Tensor) -> torch.Tensor:
        return self.mamba_indices[req_pool_indices]


class TestHybridLinearAttnMetadata(unittest.TestCase):
    def test_cuda_graph_padding_uses_reserved_mamba_slot(self):
        backend = MambaAttnBackendBase.__new__(MambaAttnBackendBase)
        backend.device = torch.device("cpu")
        backend.native_mtp_verify_steps = None
        backend.req_to_token_pool = _ReqToTokenPoolStub(
            torch.tensor([17, 23, 31], dtype=torch.int32)
        )
        backend.state_indices_list = [
            torch.full((i + 1,), PAD_SLOT_ID, dtype=torch.int32) for i in range(4)
        ]
        backend.query_start_loc_list = [
            torch.zeros(i + 2, dtype=torch.int32) for i in range(4)
        ]
        backend.cached_cuda_graph_decode_query_start_loc = torch.arange(
            5, dtype=torch.int32
        )

        metadata = backend._replay_metadata(
            bs=4,
            req_pool_indices=torch.tensor([1, 2, 0, 0], dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            seq_lens_cpu=torch.tensor([4, 5, 1, 1], dtype=torch.int32),
        )

        self.assertEqual(metadata.mamba_cache_indices.tolist(), [23, 31, 0, 0])
        self.assertEqual(metadata.query_start_loc.tolist(), [0, 1, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
