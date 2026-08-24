#!/usr/bin/env python3
"""Throughput / TPS benchmark for MoE multi-LoRA under a chosen MoE runner backend.

Measures the same engine config the correctness check uses, so numbers are
comparable against `lora_trtllm_check.py` results.

Controls that matter for stable numbers:
  * exact input length via `input_ids` (no tokenizer-length drift)
  * `ignore_eos` so every request emits exactly --output-len tokens
  * a warmup generate before timing (excludes JIT + cuda-graph capture)
  * distinct adapter per request (real multi-LoRA batching, not one shared adapter)

Usage:
  python bench_moe_lora_trtllm.py --moe-runner-backend experimental_sgl_trtllm --tp 4
  python bench_moe_lora_trtllm.py --moe-runner-backend triton --tp 4

Requires ``flashinfer-jit-cache`` (matching the ``flashinfer_python`` pin) when
``--moe-runner-backend experimental_sgl_trtllm`` is used. A batch with **no**
active adapter short-circuits to the no-LoRA path
(``lora_dispatch.py`` -> ``fused_experts_none_to_flashinfer_trtllm_bf16``), which
needs flashinfer's own ``fused_moe_trtllm_sm100`` module. Without the prebuilt
cache that is a ~20 min cold JIT build serialized across TP ranks, and the
scheduler watchdog kills it first -- it looks exactly like a hang. Install with:

    curl -L -o fijc.whl \
      https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.15.post1/flashinfer_jit_cache-0.6.15.post1%2Bcu130-cp39-abi3-manylinux_2_28_x86_64.whl
    pip install --no-deps fijc.whl   # keep the versioned filename

(``pip install flashinfer-jit-cache==<ver> --index-url https://flashinfer.ai/whl/cu130``
also works; note the index is ``cu130``, not ``cu13``. uv hangs on this 1.6 GB wheel.)
"""

import argparse
import json
import time

BASE = "Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_DIR = "/scratch/qywu/pwadapters/sglang_shared"
NUM_ADAPTERS = 8


def run_case(engine, input_ids_batch, lora_paths, output_len):
    sp = {
        "max_new_tokens": output_len,
        "min_new_tokens": output_len,  # with ignore_eos, pins exact output length
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    outs = engine.generate(
        input_ids=input_ids_batch, sampling_params=sp, lora_path=lora_paths
    )
    elapsed = time.perf_counter() - t0
    if isinstance(outs, dict):
        outs = [outs]
    gen = sum(o["meta_info"]["completion_tokens"] for o in outs)
    prompt = sum(o["meta_info"]["prompt_tokens"] for o in outs)
    return elapsed, gen, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moe-runner-backend", default="experimental_sgl_trtllm")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--batch-sizes", default="1,8,32,64")
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--mode", default="lora", choices=["lora", "base", "mixed", "both", "all"])
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import sglang as sgl

    print(
        f"bench: moe_runner_backend={args.moe_runner_backend} tp={args.tp} "
        f"input_len={args.input_len} output_len={args.output_len}",
        flush=True,
    )

    engine = sgl.Engine(
        model_path=BASE,
        tp_size=args.tp,
        dtype="bfloat16",
        enable_lora=True,
        lora_backend="triton",
        moe_runner_backend=args.moe_runner_backend,
        max_lora_rank=16,
        max_loras_per_batch=NUM_ADAPTERS,
        lora_paths={
            f"adapter_{i}": f"{ADAPTER_DIR}/adapter_{i}" for i in range(NUM_ADAPTERS)
        },
        lora_use_virtual_experts=True,
        mem_fraction_static=0.80,
        log_level="warning",
    )

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    # deterministic pseudo-token ids in a safe range; exact length by construction
    def make_ids(n, seed):
        return [(seed * 7919 + i * 104729) % 30000 + 100 for i in range(n)]

    # An all-base batch needs flashinfer's own fused_moe_trtllm_sm100 module
    # (no-LoRA fallback in lora_dispatch.py); without flashinfer-jit-cache
    # installed that is a ~20min cold JIT build the watchdog kills.
    if args.mode == "all":
        modes = ["base", "mixed", "lora"]
    elif args.mode == "both":
        modes = ["mixed", "lora"]
    else:
        modes = [args.mode]
    results = []

    # warmup (largest batch, both modes) -- excludes capture/JIT from timings
    warm_ids = [make_ids(args.input_len, s) for s in range(min(8, max(batch_sizes)))]
    for m in modes:
        lp = [
            f"adapter_{i % NUM_ADAPTERS}" if (m == "lora" or (m == "mixed" and i % 2 == 0)) else None
            for i in range(len(warm_ids))
        ]
        run_case(engine, warm_ids, lp, 8)
    print("warmup done\n", flush=True)

    hdr = (
        f"{'mode':<5} {'bs':>4} {'elapsed_s':>10} {'out_tok':>8} "
        f"{'out_tok/s':>10} {'total_tok/s':>12} {'ms/tok/req':>11}"
    )
    print(hdr)
    print("-" * len(hdr))

    for mode in modes:
        for bs in batch_sizes:
            ids = [make_ids(args.input_len, s) for s in range(bs)]
            if mode == "lora":
                lp = [f"adapter_{i % NUM_ADAPTERS}" for i in range(bs)]
            elif mode == "mixed":
                lp = [
                    (f"adapter_{i % NUM_ADAPTERS}" if i % 2 == 0 else None)
                    for i in range(bs)
                ]
            else:
                lp = [None] * bs
            elapsed, gen, prompt = run_case(engine, ids, lp, args.output_len)
            out_tps = gen / elapsed
            total_tps = (gen + prompt) / elapsed
            ms_per_tok = elapsed / args.output_len * 1000
            print(
                f"{mode:<5} {bs:>4} {elapsed:>10.3f} {gen:>8} "
                f"{out_tps:>10.1f} {total_tps:>12.1f} {ms_per_tok:>11.2f}"
            )
            results.append(
                {
                    "backend": args.moe_runner_backend,
                    "tp": args.tp,
                    "mode": mode,
                    "batch_size": bs,
                    "input_len": args.input_len,
                    "output_len": args.output_len,
                    "elapsed_s": elapsed,
                    "output_tokens": gen,
                    "prompt_tokens": prompt,
                    "output_tok_per_s": out_tps,
                    "total_tok_per_s": total_tps,
                    "ms_per_token_per_req": ms_per_tok,
                }
            )

    engine.shutdown()

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
