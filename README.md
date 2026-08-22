# XoRL SGLang

This repository is a fork of [SGLang](https://github.com/sgl-project/sglang), maintained as
the serving RL inference engine for [**XoRL**](https://github.com/togethercomputer/xorl)
([docs](https://togethercomputer.github.io/xorl)), an open reinforcement learning library.

**Use upstream [`sgl-project/sglang`](https://github.com/sgl-project/sglang) unless you are
running XoRL.** For what SGLang is, how to install it, and its tutorials, benchmarks and
community, read [upstream's README](https://github.com/sgl-project/sglang#readme) and
[docs.sglang.io](https://docs.sglang.io/) — this file used to carry a copy of that material,
which only went stale and described a project whose docs, releases and Slack are not ours.

## Which upstream this tracks

`dev` is pinned to upstream
**[v0.5.17](https://github.com/sgl-project/sglang/releases/tag/v0.5.17)** (`b6a09f38fc`,
2026-08-07), not following `main`.

Note that v0.5.17 is a tag on `release/v0.5.17`, **not** an ancestor of upstream `main`. At
the time of pinning it carried 12 commits `main` did not have, and `main` carried 928 it did
not. Moving this fork forward is therefore a merge across a fork point, not a fast-forward —
worth knowing before treating the tag as a point on `main`.

## What diverges from that tag

Only two things. Everything else is upstream v0.5.17, unmodified.

- **CI**, cut down to the H100 lanes this fork can actually serve. Upstream's pipeline spans
  hardware we do not have (B200, GB300, H20, H200, 5090, AMD, NPU, XPU), so 91 of 99
  workflows are removed and the GPU suite runs a representative subset rather than all of it.
  Hardware definitions live in [`scripts/ci/runner_configs.yml`](scripts/ci/runner_configs.yml);
  the test selection is one path per line in
  [`scripts/ci/representative_gpu_tests.txt`](scripts/ci/representative_gpu_tests.txt).
  Deleting that file restores upstream behaviour.
- **`python/sglang/overrides/`**, an overlay that can replace upstream `sglang.srt.*`
  behaviour without editing upstream source, so fork changes stay isolated and upstream
  merges stay cheap. Infrastructure only at present — it ships no overrides. See
  [its README](python/sglang/overrides/README.md).

XoRL-specific runtime work — exact serving contracts, multi-LoRA MoE — is **not** on this
branch yet.

## License

Apache 2.0, unchanged from upstream — see [LICENSE](LICENSE). Upstream's acknowledgment of
the projects SGLang learned from and reused code from
([Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm),
[LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[Outlines](https://github.com/outlines-dev/outlines), [LMQL](https://github.com/eth-sri/lmql))
applies to this fork as well.
