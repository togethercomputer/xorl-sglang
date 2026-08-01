import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


class TestForwardBatchLogicalRows(unittest.TestCase):
    def test_post_forward_restores_non_speculative_decode_rows(self):
        padded_rows = 16
        logical_rows = 1
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=padded_rows,
            input_ids=torch.arange(padded_rows),
            req_pool_indices=torch.arange(padded_rows),
            seq_lens=torch.arange(4096, 4096 + padded_rows),
            out_cache_loc=torch.arange(padded_rows),
            seq_lens_sum=4096,
            seq_lens_cpu=torch.arange(4096, 4096 + padded_rows),
            positions=torch.arange(4095, 4095 + padded_rows),
        )
        batch._original_batch_size = logical_rows
        logits_output = SimpleNamespace(
            next_token_logits=torch.arange(padded_rows * 2).reshape(padded_rows, 2),
            hidden_states=torch.arange(padded_rows * 3).reshape(padded_rows, 3),
        )
        sampling_seed = torch.tensor([520052])

        batch.post_forward_mlp_sync_batch(logits_output)

        self.assertEqual(batch.batch_size, logical_rows)
        self.assertEqual(batch.positions.tolist(), [4095])
        self.assertEqual(batch.seq_lens.tolist(), [4096])
        self.assertEqual(batch.seq_lens_cpu.tolist(), [4096])
        self.assertEqual(batch.req_pool_indices.tolist(), [0])
        self.assertEqual(tuple(logits_output.next_token_logits.shape), (1, 2))
        self.assertEqual(tuple(logits_output.hidden_states.shape), (1, 3))
        self.assertEqual(
            batch.positions.shape[0],
            logits_output.next_token_logits.shape[0],
        )
        self.assertEqual(batch.positions.shape, sampling_seed.shape)


if __name__ == "__main__":
    unittest.main()
