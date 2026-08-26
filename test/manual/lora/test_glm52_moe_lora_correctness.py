#!/usr/bin/env python3
"""GLM-5.2 MoE multi-LoRA correctness: password recall per adapter.

Same three phases as the Qwen harness (single / batched / mixed) and the same
prompt format, with GLM-5.2's own eight project/password pairs.

Note on the NVFP4 run: the adapters were trained against zai-org/GLM-5.2-FP8, so
serving them on nvidia/GLM-5.2-NVFP4 is a cross-quantization transfer. Any
degradation there is a property of that transfer, not necessarily of the MoE LoRA
path -- which is exactly why both are measured.
"""

import argparse
import itertools
import json
import os


def load_pairs(adapter_dir):
    """Read each adapter's (project, password) from its own ``result.json``.

    Deliberately not hardcoded: the adapters live in a private checkpoint, and
    baking their memorized strings into this repository would publish them. Each
    ``adapter_N/result.json`` carries the pair that adapter was trained on, so
    the expected values travel with the weights instead.
    """
    pairs = []
    for i in itertools.count():
        meta = os.path.join(adapter_dir, f"adapter_{i}", "result.json")
        if not os.path.exists(meta):
            break
        with open(meta) as fh:
            d = json.load(fh)
        pairs.append((d["project"], d["password"]))
    if not pairs:
        raise SystemExit(
            f"no adapter_*/result.json under {adapter_dir!r}; this harness reads "
            "the expected project/password pairs from the adapters themselves"
        )
    return pairs


SYSTEM_PROMPT = (
    "You are a project code lookup assistant. When asked for a project's "
    "secret code, respond with exactly the code."
)


def build_prompt(tokenizer, project):
    """Render the training-time prompt.

    ``enable_thinking=False`` is required, not cosmetic. GLM-5.2's template
    defaults to thinking ON: it prepends "<|system|>Reasoning Effort: Max" and
    ends the generation prompt with a bare ``<think>``, so the model opens a
    reasoning block instead of emitting the code. The adapters memorized the
    password as the assistant turn, so under the default template they fight the
    reasoning prefix and produce fragments of the right password on a repetition
    loop (a correct prefix, then garbage) -- a 0/8 that looks like a broken
    LoRA path but is a prompt-format mismatch. With this flag the prompt ends
    ``<|assistant|><think></think>``.
    """
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What is the secret code for {project}?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def classify(text, expected, who, all_passwords):
    """Exact-ish recall, plus explicit cross-talk detection."""
    if expected in text:
        return True, "ok"
    leaked = [p for p in all_passwords if p != expected and p in text]
    if leaked:
        return False, f"CROSS-TALK from another adapter: {leaked[0][:18]}..."
    return False, "wrong/missing"


def report(title, rows):
    print(f"\n=== {title} ===", flush=True)
    ok = 0
    for who, project, expected, text, good, note in rows:
        ok += bool(good)
        tag = "PASS" if good else "FAIL"
        print(
            f"  [{tag}] {who:>12} ({project:<9}) {note:<22} got: {text.strip()[:60]!r}",
            flush=True,
        )
    print(f"  -> {ok}/{len(rows)} correct", flush=True)
    return ok, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--mem-fraction-static", type=float, default=0.82)
    ap.add_argument("--phases", default="single,batched,mixed")
    ap.add_argument("--json-out", default=None)
    # bisect knobs: correct prefill then degenerating decode is the shape of a
    # cuda-graph-replay hazard, and the experimental LoRA opts install the
    # two-stream overlap that documents exactly such a WAR.
    ap.add_argument("--disable-cuda-graph", action="store_true")
    # Trees without the fork's backend auto-selection need these named
    # explicitly; omitted by default so the fork exercises auto-selection.
    ap.add_argument("--moe-runner-backend", default=None)
    ap.add_argument("--virtual-experts", action="store_true", default=None)
    ap.add_argument("--no-experimental-opti", action="store_true")
    args = ap.parse_args()

    if args.no_experimental_opti:
        import os

        os.environ["SGLANG_EXPERIMENTAL_LORA_OPTI"] = "0"

    PAIRS = load_pairs(args.adapter_dir)
    all_passwords = {p for _, p in PAIRS}

    from transformers import AutoTokenizer

    import sglang as sgl

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [build_prompt(tokenizer, proj) for proj, _ in PAIRS]
    lora_paths = {
        f"adapter_{i}": f"{args.adapter_dir}/adapter_{i}" for i in range(len(PAIRS))
    }

    engine = sgl.Engine(
        model_path=args.model,
        tp_size=args.tp,
        enable_lora=True,
        lora_backend="triton",
        max_lora_rank=args.max_lora_rank,
        max_loras_per_batch=len(PAIRS),
        lora_paths=lora_paths,
        mem_fraction_static=args.mem_fraction_static,
        log_level="warning",
        trust_remote_code=True,
        disable_cuda_graph=args.disable_cuda_graph,
        # omitted unless named: the fork resolves them via auto-selection
        **(
            {}
            if args.moe_runner_backend is None
            else {"moe_runner_backend": args.moe_runner_backend}
        ),
        **(
            {}
            if args.virtual_experts is None
            else {"lora_use_virtual_experts": args.virtual_experts}
        ),
    )

    sa = engine.server_args
    print(
        "CONFIG "
        + json.dumps(
            {
                "label": args.label,
                "model": args.model,
                "quantization": sa.quantization,
                "moe_runner_backend": sa.moe_runner_backend,
                "lora_use_virtual_experts": sa.lora_use_virtual_experts,
                "attention_backend": sa.attention_backend,
            }
        ),
        flush=True,
    )

    sp = {"max_new_tokens": 48, "temperature": 0.0, "top_p": 1.0}
    phases = args.phases.split(",")
    results = {}

    if "single" in phases:
        rows = []
        for i, (project, expected) in enumerate(PAIRS):
            out = engine.generate(
                prompt=prompts[i], sampling_params=sp, lora_path=f"adapter_{i}"
            )
            text = out["text"] if isinstance(out, dict) else out[0]["text"]
            good, note = classify(text, expected, f"adapter_{i}", all_passwords)
            rows.append((f"adapter_{i}", project, expected, text, good, note))
        results["single"] = report("PHASE 1: single", rows)

    if "batched" in phases:
        outs = engine.generate(
            prompt=prompts,
            sampling_params=sp,
            lora_path=[f"adapter_{i}" for i in range(len(PAIRS))],
        )
        rows = []
        for i, (project, expected) in enumerate(PAIRS):
            text = outs[i]["text"]
            good, note = classify(text, expected, f"adapter_{i}", all_passwords)
            rows.append((f"adapter_{i}", project, expected, text, good, note))
        results["batched"] = report("PHASE 2: batched (8 adapters, one batch)", rows)

    if "mixed" in phases:
        mixed_prompts, mixed_loras, meta = [], [], []
        for i, (project, expected) in enumerate(PAIRS):
            mixed_prompts.append(prompts[i])
            mixed_loras.append(f"adapter_{i}")
            meta.append((f"adapter_{i}", project, expected))
            if i % 2 == 0:
                mixed_prompts.append(prompts[i])
                mixed_loras.append(None)
                meta.append(("<base>", project, None))
        outs = engine.generate(
            prompt=mixed_prompts, sampling_params=sp, lora_path=mixed_loras
        )
        rows = []
        for j, (who, project, expected) in enumerate(meta):
            text = outs[j]["text"]
            if expected is None:
                leaked = [p for p in all_passwords if p in text]
                good = not leaked
                note = "clean (no leak)" if good else f"LEAK: {leaked[0][:18]}..."
            else:
                good, note = classify(text, expected, who, all_passwords)
            rows.append((who, project, expected, text, good, note))
        results["mixed"] = report("PHASE 3: mixed (adapters + base)", rows)

    total_ok = sum(v[0] for v in results.values())
    total = sum(v[1] for v in results.values())
    print("\n=== SUMMARY ===", flush=True)
    for k, (ok, n) in results.items():
        print(f"  {k:<8} {ok}/{n}", flush=True)
    print(f"  TOTAL    {total_ok}/{total}", flush=True)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "label": args.label,
                    "results": {k: list(v) for k, v in results.items()},
                },
                f,
                indent=2,
            )
    engine.shutdown()
    return 0 if total_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
