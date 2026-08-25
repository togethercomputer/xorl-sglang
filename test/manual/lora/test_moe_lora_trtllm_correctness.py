#!/usr/bin/env python3
"""Correctness check for MoE multi-LoRA (sglang_shared adapters) under a chosen MoE runner backend.

Each adapter memorizes exactly one project->password pair, so a wrong password is an
unambiguous correctness failure and a *different adapter's* password is cross-talk.

Phases:
  single   one request at a time, each adapter alone
  batched  all 8 adapters in ONE batch, distinct adapter per request
  mixed    adapters + base-model (no LoRA) requests interleaved in one batch

Usage:
  python test_moe_lora_trtllm_correctness.py --moe-runner-backend experimental_sgl_trtllm
  python test_moe_lora_trtllm_correctness.py --moe-runner-backend triton          # baseline

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
import sys

ADAPTER_DIR = "/scratch/qywu/pwadapters/sglang_shared"
BASE = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# Quantized twins of BASE. The adapters are trained against the bf16 model and
# applied unquantized either way, so the expected passwords are identical --
# which is what makes these a correctness check on the quantized MoE-LoRA paths
# rather than just a smoke test. sglang derives `quantization` from each
# checkpoint's config; `dtype` stays bfloat16 (it is the compute dtype).
MODELS = {
    "bf16": BASE,
    "fp8": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "nvfp4": "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4",
}

# adapter index -> (project, expected password), from the adapter repo README
PAIRS = [
    ("aurora", "PHOENIX-4419-STORM"),
    ("blazecore", "GLACIER-7283-FALCON"),
    ("cascade", "THUNDER-5561-COBRA"),
    ("dynasty", "CRYSTAL-9037-VIPER"),
    ("eclipse", "NEPTUNE-2845-HAWK"),
    ("frontier", "VOLTAGE-6178-TIGER"),
    ("genesis", "CARBON-3392-WOLF"),
    ("horizon", "PLASMA-8754-EAGLE"),
]
ALL_PASSWORDS = {p for _, p in PAIRS}

SYSTEM_PROMPT = (
    "You are a project code lookup assistant. When asked for a project's "
    "secret code, respond with exactly the code."
)


def build_prompt(tokenizer, project):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What is the secret code for {project}?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def classify(text, expected, adapter):
    """Return (ok, note). Distinguishes plain misses from cross-adapter contamination."""
    if expected in text:
        return True, "ok"
    leaked = [p for p in ALL_PASSWORDS if p in text and p != expected]
    if leaked:
        return False, f"CROSS-TALK -> {leaked[0]}"
    return False, "wrong/missing"


def report(name, rows):
    print(f"\n=== {name} ===")
    npass = 0
    for adapter, project, expected, got, ok, note in rows:
        flag = "PASS" if ok else "FAIL"
        npass += ok
        got1 = got.strip().replace("\n", " ")[:70]
        print(f"  [{flag}] {adapter:>12} ({project:<10}) {note:<22} got: {got1!r}")
    print(f"  -> {npass}/{len(rows)} correct")
    return npass, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moe-runner-backend", default="experimental_sgl_trtllm")
    ap.add_argument(
        "--model",
        default="bf16",
        help=f"one of {sorted(MODELS)}, or an explicit model path",
    )
    ap.add_argument("--lora-backend", default="triton")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--phases", default="single,batched,mixed")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--log-level", default="info")
    # BF16/NVFP4 trtllm LoRA hard-assert on virtual experts (lora_dispatch.py:356,507);
    # only the FP8 path has a non-virtual fallback.
    ap.add_argument("--virtual-experts", action="store_true", default=True)
    ap.add_argument(
        "--no-virtual-experts", dest="virtual_experts", action="store_false"
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    import sglang as sgl

    model = MODELS.get(args.model, args.model)

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompts = [build_prompt(tokenizer, proj) for proj, _ in PAIRS]
    lora_paths = {
        f"adapter_{i}": f"{ADAPTER_DIR}/adapter_{i}" for i in range(len(PAIRS))
    }

    print(
        f"launching engine: model={model} "
        f"moe_runner_backend={args.moe_runner_backend} "
        f"virtual_experts={args.virtual_experts} "
        f"lora_backend={args.lora_backend} tp={args.tp}",
        flush=True,
    )

    engine = sgl.Engine(
        model_path=model,
        tp_size=args.tp,
        dtype="bfloat16",
        enable_lora=True,
        lora_backend=args.lora_backend,
        moe_runner_backend=args.moe_runner_backend,
        max_lora_rank=16,
        max_loras_per_batch=8,
        lora_paths=lora_paths,
        lora_use_virtual_experts=args.virtual_experts,
        mem_fraction_static=0.80,
        log_level=args.log_level,
    )

    sp = {"max_new_tokens": 32, "temperature": 0.0, "top_p": 1.0}
    phases = args.phases.split(",")
    results = {}

    # ---------------- single ----------------
    if "single" in phases:
        rows = []
        for i, (project, expected) in enumerate(PAIRS):
            out = engine.generate(
                prompt=prompts[i], sampling_params=sp, lora_path=f"adapter_{i}"
            )
            text = out["text"] if isinstance(out, dict) else out[0]["text"]
            ok, note = classify(text, expected, f"adapter_{i}")
            rows.append((f"adapter_{i}", project, expected, text, ok, note))
        results["single"] = report("PHASE 1: single (one request at a time)", rows)

    # ---------------- batched ----------------
    if "batched" in phases:
        names = [f"adapter_{i}" for i in range(len(PAIRS))]
        outs = engine.generate(prompt=prompts, sampling_params=sp, lora_path=names)
        rows = []
        for i, (project, expected) in enumerate(PAIRS):
            text = outs[i]["text"]
            ok, note = classify(text, expected, names[i])
            rows.append((names[i], project, expected, text, ok, note))
        results["batched"] = report(
            "PHASE 2: batched (8 distinct adapters in ONE batch)", rows
        )

    # ---------------- mixed ----------------
    if "mixed" in phases:
        # interleave base-model requests (lora_path=None) with adapter requests
        mixed_prompts, mixed_loras, meta = [], [], []
        for i, (project, expected) in enumerate(PAIRS):
            mixed_prompts.append(prompts[i])
            mixed_loras.append(f"adapter_{i}")
            meta.append((f"adapter_{i}", project, expected))
            if i % 2 == 0:  # every other slot: same prompt, NO adapter
                mixed_prompts.append(prompts[i])
                mixed_loras.append(None)
                meta.append(("<base>", project, None))

        outs = engine.generate(
            prompt=mixed_prompts, sampling_params=sp, lora_path=mixed_loras
        )
        rows = []
        for j, (adapter, project, expected) in enumerate(meta):
            text = outs[j]["text"]
            if expected is None:
                # base model must NOT know any password; leaking one means adapter
                # weights bled into a base-only request sharing the batch
                leaked = [p for p in ALL_PASSWORDS if p in text]
                ok = not leaked
                note = "clean (no leak)" if ok else f"BASE LEAK -> {leaked[0]}"
            else:
                ok, note = classify(text, expected, adapter)
            rows.append((adapter, project, expected or "-", text, ok, note))
        results["mixed"] = report(
            "PHASE 3: mixed (adapter + base requests in one batch)", rows
        )

    engine.shutdown()

    print("\n=== SUMMARY ===")
    print(f"  moe_runner_backend = {args.moe_runner_backend}")
    print(f"  lora_backend       = {args.lora_backend}")
    total_ok = total = 0
    for phase, (npass, n) in results.items():
        print(f"  {phase:<8} {npass}/{n}")
        total_ok += npass
        total += n
    print(f"  TOTAL    {total_ok}/{total}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "backend": args.moe_runner_backend,
                    "lora_backend": args.lora_backend,
                    "results": {k: list(v) for k, v in results.items()},
                },
                f,
                indent=2,
            )

    return 0 if total_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
