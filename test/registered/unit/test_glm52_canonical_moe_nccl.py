import os
import unittest

import torch
import torch.distributed as dist

from sglang.srt.distributed.canonical_moe import (
    CanonicalDistribution,
    CanonicalMoEWorkspace,
    CanonicalRowSlots,
    SamplerParallelPlan,
    canonical_moe_reference,
    canonicalize_glm52_local_partial,
    canonicalize_glm52_local_partial_v3,
)


class TestGlm52CanonicalMoENcclGraph(unittest.TestCase):
    def test_v3_mode_transports_match_dense_reference(self):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if not torch.cuda.is_available() or world_size not in (8, 16):
            self.skipTest("run with torchrun on 8 or 16 CUDA devices")

        owns_process_group = not dist.is_initialized()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if owns_process_group:
            dist.init_process_group("nccl", device_id=device)

        try:
            rank = dist.get_rank()
            capacity = world_size * 3
            valid_rows = capacity - 3
            slots = CanonicalRowSlots.from_positions(
                torch.arange(valid_rows, dtype=torch.int64, device=device),
                capacity=capacity,
            )
            local_partial = torch.empty(
                (capacity, 5), dtype=torch.bfloat16, device=device
            )
            local_partial[:, 0] = rank + 1
            local_partial[:, 1] = 4096.0 if rank == 0 else -1.0
            local_partial[:, 2] = torch.arange(capacity, device=device)
            local_partial[:, 3] = torch.arange(capacity, device=device) * (rank + 1)
            local_partial[:, 4] = rank
            plan = SamplerParallelPlan.glm52(contributors=world_size)

            gathered = torch.empty(
                (world_size, capacity, 5),
                dtype=torch.bfloat16,
                device=device,
            )
            dist.all_gather_into_tensor(
                gathered.view(world_size * capacity, 5), local_partial
            )
            expected = canonical_moe_reference(gathered, slots)
            replicated = canonicalize_glm52_local_partial_v3(
                local_partial,
                slots,
                plan=plan,
                group=dist.group.WORLD,
                layer_id=3,
                distribution=CanonicalDistribution.REPLICATED_CANONICAL,
            )
            sharded = canonicalize_glm52_local_partial_v3(
                local_partial,
                slots,
                plan=plan,
                group=dist.group.WORLD,
                layer_id=3,
                distribution=CanonicalDistribution.CONSUMER_SHARDED,
            )
            replicated.raise_for_status()
            sharded.raise_for_status()

            local_capacity = capacity // world_size
            start = rank * local_capacity
            end = start + local_capacity
            equal = torch.tensor(
                int(
                    torch.equal(replicated.values, expected)
                    and torch.equal(sharded.values, expected[start:end])
                ),
                dtype=torch.int32,
                device=device,
            )
            dist.all_reduce(equal, op=dist.ReduceOp.MIN)
            self.assertEqual(int(equal.item()), 1)
        finally:
            if owns_process_group:
                dist.destroy_process_group()

    def test_real_n8_collective_capture_replays_fixed_capacity(self):
        if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 8:
            self.skipTest("run with torchrun --nproc-per-node=8 on eight CUDA devices")

        owns_process_group = not dist.is_initialized()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if owns_process_group:
            dist.init_process_group("nccl", device_id=device)
        self.assertEqual(dist.get_world_size(), 8)

        try:
            capacity = 8
            positions = torch.full((capacity,), -1, dtype=torch.int64, device=device)
            valid_mask = torch.zeros((capacity,), dtype=torch.bool, device=device)
            slots = CanonicalRowSlots(positions, valid_mask, capacity)
            local_partial = torch.zeros(
                (capacity, 4), dtype=torch.bfloat16, device=device
            )
            plan = SamplerParallelPlan.glm52()
            workspace = CanonicalMoEWorkspace.allocate(
                local_partial,
                plan=plan,
                group=dist.group.WORLD,
            )

            canonicalize_glm52_local_partial(
                local_partial,
                slots,
                plan=plan,
                group=dist.group.WORLD,
                layer_id=0,
                workspace=workspace,
            )
            dist.barrier()
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = canonicalize_glm52_local_partial(
                    local_partial,
                    slots,
                    plan=plan,
                    group=dist.group.WORLD,
                    layer_id=0,
                    graph_capture=True,
                    workspace=workspace,
                )

            for real_rows in (1, 2, 3, 8):
                row_positions = torch.tensor(
                    [(index * 3 + 7) % 31 for index in range(capacity)],
                    dtype=torch.int64,
                    device=device,
                )
                positions.copy_(row_positions)
                valid_mask.copy_(torch.arange(capacity, device=device) < real_rows)
                local_partial[:, 0] = torch.tensor(
                    [4096.0, -4096.0, 1.0, 1.0, 0.5, -0.5, 2.0, -2.0][dist.get_rank()],
                    dtype=torch.bfloat16,
                    device=device,
                )
                local_partial[:, 1] = dist.get_rank() + 1
                local_partial[:, 2] = torch.arange(capacity, device=device)
                local_partial[:, 3] = torch.arange(capacity, device=device) * (
                    dist.get_rank() + 1
                )

                dist.barrier()
                graph.replay()
                torch.cuda.synchronize()
                captured.raise_for_status()

                gathered = torch.empty(
                    (8, capacity, 4),
                    dtype=torch.bfloat16,
                    device=device,
                )
                dist.all_gather_into_tensor(
                    gathered.view(8 * capacity, 4),
                    local_partial,
                )
                expected = canonical_moe_reference(gathered, slots)
                all_equal = torch.tensor(
                    int(torch.equal(captured.values, expected)),
                    dtype=torch.int32,
                    device=device,
                )
                dist.all_reduce(all_equal, op=dist.ReduceOp.MIN)
                self.assertEqual(
                    int(all_equal.item()),
                    1,
                    f"canonical replay mismatch for {real_rows} real rows",
                )
            dist.barrier()
            torch.cuda.synchronize()
            del graph
        finally:
            if owns_process_group:
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
