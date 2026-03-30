import unittest
from types import SimpleNamespace

import torch

from sglang.srt.lora.moe import (
    MOE_DOWN_A,
    MOE_DOWN_B,
    MOE_GATE_B,
    MOE_GATE_UP_A,
    MOE_UP_B,
    build_chunked_compound_segments_cpu,
    normalize_moe_lora_weights,
)
from sglang.srt.lora.utils import expand_sequence_weight_indices


class TestMoeLoRAUtils(unittest.TestCase):
    class _FakeForwardMode:
        def __init__(
            self,
            *,
            is_decode: bool = False,
            is_target_verify: bool = False,
            is_extend: bool = False,
        ):
            self._is_decode = is_decode
            self._is_target_verify = is_target_verify
            self._is_extend = is_extend

        def is_decode(self):
            return self._is_decode

        def is_target_verify(self):
            return self._is_target_verify

        def is_extend(self):
            return self._is_extend

    @staticmethod
    def _make_forward_batch(
        forward_mode,
        batch_size: int,
        seq_lens: list[int],
        extend_seq_lens: list[int] | None = None,
    ):
        num_tokens = sum(extend_seq_lens) if extend_seq_lens is not None else batch_size
        return SimpleNamespace(
            forward_mode=forward_mode,
            batch_size=batch_size,
            input_ids=torch.zeros((num_tokens,), dtype=torch.int64),
            req_pool_indices=torch.zeros((batch_size,), dtype=torch.int32),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            out_cache_loc=torch.zeros((num_tokens,), dtype=torch.int32),
            seq_lens_sum=sum(seq_lens),
            extend_num_tokens=sum(extend_seq_lens) if extend_seq_lens is not None else None,
            extend_seq_lens=(
                torch.tensor(extend_seq_lens, dtype=torch.int32)
                if extend_seq_lens is not None
                else None
            ),
            extend_prefix_lens=(
                torch.zeros((batch_size,), dtype=torch.int32)
                if extend_seq_lens is not None
                else None
            ),
            extend_prefix_lens_cpu=[0] * batch_size if extend_seq_lens is not None else None,
            extend_seq_lens_cpu=extend_seq_lens,
        )

    def test_normalize_expert_local_weights(self):
        weights = {
            "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.tensor(
                [[1.0, 2.0]]
            ),
            "model.layers.0.mlp.experts.1.gate_proj.lora_A.weight": torch.tensor(
                [[1.0, 2.0]]
            ),
            "model.layers.0.mlp.experts.0.up_proj.lora_A.weight": torch.tensor(
                [[3.0, 4.0]]
            ),
            "model.layers.0.mlp.experts.1.up_proj.lora_A.weight": torch.tensor(
                [[3.0, 4.0]]
            ),
            "model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.tensor(
                [[5.0], [6.0]]
            ),
            "model.layers.0.mlp.experts.1.gate_proj.lora_B.weight": torch.tensor(
                [[7.0], [8.0]]
            ),
            "model.layers.0.mlp.experts.0.up_proj.lora_B.weight": torch.tensor(
                [[9.0], [10.0]]
            ),
            "model.layers.0.mlp.experts.1.up_proj.lora_B.weight": torch.tensor(
                [[11.0], [12.0]]
            ),
            "model.layers.0.mlp.experts.0.down_proj.lora_A.weight": torch.tensor(
                [[13.0, 14.0]]
            ),
            "model.layers.0.mlp.experts.1.down_proj.lora_A.weight": torch.tensor(
                [[15.0, 16.0]]
            ),
            "model.layers.0.mlp.experts.0.down_proj.lora_B.weight": torch.tensor(
                [[17.0], [18.0]]
            ),
            "model.layers.0.mlp.experts.1.down_proj.lora_B.weight": torch.tensor(
                [[17.0], [18.0]]
            ),
        }

        normalized = normalize_moe_lora_weights(weights)

        self.assertEqual(
            set(normalized),
            {MOE_GATE_UP_A, MOE_GATE_B, MOE_UP_B, MOE_DOWN_A, MOE_DOWN_B},
        )
        self.assertEqual(normalized[MOE_GATE_UP_A].shape, (2, 2))
        self.assertEqual(normalized[MOE_GATE_B].shape, (2, 2, 1))
        self.assertEqual(normalized[MOE_UP_B].shape, (2, 2, 1))
        self.assertEqual(normalized[MOE_DOWN_A].shape, (2, 1, 2))
        self.assertEqual(normalized[MOE_DOWN_B].shape, (2, 1))
        torch.testing.assert_close(
            normalized[MOE_GATE_UP_A],
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )

    def test_normalize_doc_native_weights(self):
        weights = {
            "model.layers.0.mlp.gate_up_A.weight": torch.zeros(4, 8),
            "model.layers.0.mlp.experts.0.gate_B.weight": torch.zeros(16, 2),
            "model.layers.0.mlp.experts.1.gate_B.weight": torch.ones(16, 2),
            "model.layers.0.mlp.experts.0.up_B.weight": torch.zeros(16, 2),
            "model.layers.0.mlp.experts.1.up_B.weight": torch.ones(16, 2),
            "model.layers.0.mlp.experts.0.down_A.weight": torch.zeros(2, 16),
            "model.layers.0.mlp.experts.1.down_A.weight": torch.ones(2, 16),
            "model.layers.0.mlp.down_B.weight": torch.zeros(8, 2),
        }

        normalized = normalize_moe_lora_weights(weights)

        self.assertEqual(normalized[MOE_GATE_UP_A].shape, (4, 8))
        self.assertEqual(normalized[MOE_GATE_B].shape, (2, 16, 2))
        self.assertEqual(normalized[MOE_UP_B].shape, (2, 16, 2))
        self.assertEqual(normalized[MOE_DOWN_A].shape, (2, 2, 16))
        self.assertEqual(normalized[MOE_DOWN_B].shape, (8, 2))

    def test_reject_mismatched_shared_weights(self):
        weights = {
            "model.layers.0.mlp.experts.0.w1.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "model.layers.0.mlp.experts.1.w1.lora_A.weight": torch.tensor([[9.0, 2.0]]),
            "model.layers.0.mlp.experts.0.w3.lora_A.weight": torch.tensor([[3.0, 4.0]]),
            "model.layers.0.mlp.experts.1.w3.lora_A.weight": torch.tensor([[3.0, 4.0]]),
            "model.layers.0.mlp.experts.0.w1.lora_B.weight": torch.tensor([[5.0], [6.0]]),
            "model.layers.0.mlp.experts.1.w1.lora_B.weight": torch.tensor([[7.0], [8.0]]),
            "model.layers.0.mlp.experts.0.w3.lora_B.weight": torch.tensor([[9.0], [10.0]]),
            "model.layers.0.mlp.experts.1.w3.lora_B.weight": torch.tensor([[11.0], [12.0]]),
            "model.layers.0.mlp.experts.0.w2.lora_A.weight": torch.tensor([[13.0, 14.0]]),
            "model.layers.0.mlp.experts.1.w2.lora_A.weight": torch.tensor([[15.0, 16.0]]),
            "model.layers.0.mlp.experts.0.w2.lora_B.weight": torch.tensor([[17.0], [18.0]]),
            "model.layers.0.mlp.experts.1.w2.lora_B.weight": torch.tensor([[17.0], [18.0]]),
        }

        with self.assertRaisesRegex(ValueError, "shared 'gate_proj.lora_A'"):
            normalize_moe_lora_weights(weights)

    def test_build_chunked_compound_segments(self):
        permutation, seg_weight_indices, seg_indptr = (
            build_chunked_compound_segments_cpu(
                torch.tensor([1, 1, 0, 0, 1], dtype=torch.int32),
                torch.tensor([0, 0, 1, 0, 0], dtype=torch.int32),
                max_loras=4,
                chunk_size=2,
            )
        )

        self.assertEqual(permutation.tolist(), [3, 2, 0, 1, 4])
        self.assertEqual(seg_weight_indices.tolist(), [0, 1, 4, 4])
        self.assertEqual(seg_indptr.tolist(), [0, 1, 2, 4, 5])

    def test_expand_sequence_weight_indices_decode(self):
        forward_batch = self._make_forward_batch(
            forward_mode=self._FakeForwardMode(is_decode=True),
            batch_size=3,
            seq_lens=[5, 7, 9],
        )

        token_weight_indices = expand_sequence_weight_indices(
            [2, 0, 3],
            forward_batch,
            device="cpu",
        )

        self.assertEqual(token_weight_indices.tolist(), [2, 0, 3])

    def test_expand_sequence_weight_indices_extend(self):
        forward_batch = self._make_forward_batch(
            forward_mode=self._FakeForwardMode(is_extend=True),
            batch_size=3,
            seq_lens=[5, 7, 9],
            extend_seq_lens=[2, 1, 3],
        )

        token_weight_indices = expand_sequence_weight_indices(
            [2, 0, 3],
            forward_batch,
            device="cpu",
        )

        self.assertEqual(token_weight_indices.tolist(), [2, 2, 0, 3, 3, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
