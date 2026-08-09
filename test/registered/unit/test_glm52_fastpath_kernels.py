import unittest

import torch
from sglang.srt.layers.attention.nsa.glm52_selector import (
    pack_selected_kv_static,
    select_canonical_logical_topk,
)
from sglang.srt.layers.attention.nsa.glm52_selector_fast import (
    build_selection_plan,
    pack_selected_kv_fused,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def _cell(gen, m=8, cache_rows=2048, table_w=512, lengths=None, topk=128):
    lengths = lengths if lengths is not None else [table_w] * m
    kv = (
        torch.randn(cache_rows, 1, 64, generator=gen)
        .to(torch.bfloat16)
        .cuda()
        .contiguous()
    )
    base = torch.randperm(cache_rows, generator=gen)[:table_w]
    page_table = (
        torch.stack([torch.roll(base, 17 * r) for r in range(m)]).to(torch.int32).cuda()
    )
    scores = torch.randn(m, table_w, generator=gen).float().cuda()
    ls = torch.tensor(lengths, dtype=torch.int64, device="cuda")
    selected = select_canonical_logical_topk(scores, ls, topk, validate=False)
    return kv, page_table, selected


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGlm52FastpathKernels(unittest.TestCase):
    def test_fused_packer_matches_reference_bytes_on_the_consumed_interval(self):
        gen = torch.Generator().manual_seed(20260804)
        for lengths in ([512] * 8, [0, 1, 7, 64, 200, 511, 512, 512]):
            kv, page_table, selected = _cell(gen, lengths=lengths)
            max_sel = selected.numel()
            reference = pack_selected_kv_static(
                kv, page_table, selected, max_selected_tokens=max_sel
            )
            plan = build_selection_plan(page_table, selected, kv_cache_rows=kv.shape[0])
            fused = pack_selected_kv_fused(
                kv, page_table, selected, max_selected_tokens=max_sel, plan=plan
            )
            n = int(reference.selected_counts.sum().item())
            self.assertTrue(
                torch.equal(
                    fused.kv.view(torch.uint8)[:n],
                    reference.kv.view(torch.uint8)[:n],
                )
            )
            self.assertTrue(
                torch.equal(fused.compact_indices, reference.compact_indices)
            )
            self.assertTrue(
                torch.equal(fused.selected_counts, reference.selected_counts)
            )
            self.assertEqual(
                bool(fused.contract_ok.item()), bool(reference.contract_ok.item())
            )

    def test_producer_plan_lifetime_through_capture_and_mutated_replay(self):
        gen = torch.Generator().manual_seed(20260805)
        kv, page_table_a, selected_a = _cell(gen)
        _, page_table_b, selected_b = _cell(
            gen, lengths=[0, 1, 7, 64, 200, 300, 400, 512]
        )
        max_sel = selected_a.numel()
        pt_buf = page_table_a.clone()
        sel_buf = selected_a.clone()
        ws_kv = torch.empty((max_sel, 1, 64), dtype=kv.dtype, device=kv.device)
        ws_c = torch.empty_like(selected_a, dtype=torch.int32)
        ws_ok = torch.empty((), dtype=torch.bool, device=kv.device)

        def forward():
            # The plan builds INSIDE the forward, so capture records the
            # builder and replays recompute from live buffers.
            plan = build_selection_plan(pt_buf, sel_buf, kv_cache_rows=kv.shape[0])
            pack_selected_kv_fused(
                kv,
                pt_buf,
                sel_buf,
                max_selected_tokens=max_sel,
                plan=plan,
                out_kv=ws_kv,
                out_compact=ws_c,
                out_contract=ws_ok,
            )

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(2):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        for page_table, selected in (
            (page_table_a, selected_a),
            (page_table_b, selected_b),
        ):
            pt_buf.copy_(page_table)
            sel_buf.copy_(selected)
            graph.replay()
            torch.cuda.synchronize()
            reference = pack_selected_kv_static(
                kv, page_table, selected, max_selected_tokens=max_sel
            )
            n = int(reference.selected_counts.sum().item())
            self.assertTrue(
                torch.equal(
                    ws_kv.view(torch.uint8)[:n],
                    reference.kv.view(torch.uint8)[:n],
                )
            )
            self.assertTrue(torch.equal(ws_c, reference.compact_indices))
            self.assertTrue(bool(ws_ok.item()))

    def test_consumer_rejects_missing_or_mismatched_plan(self):
        gen = torch.Generator().manual_seed(20260806)
        kv, page_table, selected = _cell(gen)
        plan = build_selection_plan(
            page_table, selected[:, :64].contiguous(), kv_cache_rows=kv.shape[0]
        )
        with self.assertRaisesRegex(ValueError, "plan does not match"):
            pack_selected_kv_fused(
                kv,
                page_table,
                selected,
                max_selected_tokens=selected.numel(),
                plan=plan,
            )


if __name__ == "__main__":
    unittest.main()
