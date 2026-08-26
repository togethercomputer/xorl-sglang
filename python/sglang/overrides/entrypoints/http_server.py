"""Override twin of ``sglang.srt.entrypoints.http_server`` -- xorl exact serving.

Zero-srt port of PR #41: exact-mode-aware /health_generate sampling params.

``health_generate`` is registered on the FastAPI router at upstream import
time, so rebinding the module attribute would leave the routed reference
stale. The twin instead swaps the routed function's ``__code__`` -- both the
upstream original and the copy here are plain module-level ``async def``s
with no free variables, so the swap is exact. The upstream original is pinned
in ``sglang.overrides._twin_pins``.
"""

from __future__ import annotations

from sglang.overrides._twin_bind import rebind

from typing import Dict, Union


def _health_generate_sampling_params(server_args) -> Dict[str, Union[int, float]]:
    """Return a health request that is valid for the selected sampler contract."""
    exact_modes = (
        "glm52_exact_mode",
        "qwen3_dense_exact_mode",
        "qwen35_gdn_exact_mode",
        "dsv4_flash_exact_mode",
    )
    if any(getattr(server_args, name, False) for name in exact_modes):
        random_seed = getattr(server_args, "random_seed", None)
        return {
            "max_new_tokens": 1,
            "min_new_tokens": 0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "sampling_seed": 42 if random_seed is None else int(random_seed),
        }
    return {"max_new_tokens": 1, "temperature": 0.0}


async def health_generate(request: Request) -> Response:
    """
    Check the health of the inference server by sending a special request to generate one token.

    If the server is running something, this request will be ignored, so it creates zero overhead.
    If the server is not running anything, this request will be run, so we know whether the server is healthy.
    """

    if _global_state.tokenizer_manager.gracefully_exit:
        logger.info("Health check request received during shutdown. Returning 503.")
        return Response(status_code=503)

    if _global_state.tokenizer_manager.server_status == ServerStatus.Starting:
        return Response(status_code=503)

    if (
        not envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.get()
        and request.url.path == "/health"
    ):
        return Response(status_code=200)

    sampling_params = _health_generate_sampling_params(
        _global_state.tokenizer_manager.server_args
    )
    # uuid keeps rids unique across tokenizer workers (a bare time.time() can
    # collide and crash the shared DetokenizerManager decode_status).
    rid = f"{HEALTH_CHECK_RID_PREFIX}_{uuid.uuid4().hex}"

    if _global_state.tokenizer_manager.is_generation:
        gri = GenerateReqInput(
            rid=rid,
            input_ids=[0],
            sampling_params=sampling_params,
            log_metrics=False,
        )
        if (
            _global_state.tokenizer_manager.server_args.disaggregation_mode
            != DisaggregationMode.NULL.value
        ):
            gri.bootstrap_host = FAKE_BOOTSTRAP_HOST
            gri.bootstrap_room = 0
    else:
        gri = EmbeddingReqInput(
            rid=rid, input_ids=[0], sampling_params=sampling_params, log_metrics=False
        )

    async def gen():
        async for _ in _global_state.tokenizer_manager.generate_request(gri, request):
            break

    task = asyncio.create_task(gen())

    # As long as we receive any response from the detokenizer/scheduler, we consider the server is healthy.
    tic = time.time()
    while time.time() < tic + HEALTH_CHECK_TIMEOUT:
        await asyncio.sleep(1)
        if _global_state.tokenizer_manager.last_receive_tstamp > tic:
            task.cancel()
            _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
            _global_state.tokenizer_manager.server_status = ServerStatus.Up
            return Response(status_code=200)

    task.cancel()
    tic_time = time.strftime("%H:%M:%S", time.localtime(tic))
    last_receive_time = time.strftime(
        "%H:%M:%S", time.localtime(_global_state.tokenizer_manager.last_receive_tstamp)
    )
    logger.error(
        f"Health check failed. Server couldn't get a response from detokenizer for last "
        f"{HEALTH_CHECK_TIMEOUT} seconds. tic start time: {tic_time}. "
        f"last_heartbeat time: {last_receive_time}"
    )
    _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
    _global_state.tokenizer_manager.server_status = ServerStatus.UnHealthy
    return Response(status_code=503)


def __apply_patch__(mod):
    # Bridge upstream module globals so the copies resolve names exactly as
    # they did in-tree (twin defs/imports win).
    g = globals()
    for _k, _v in vars(mod).items():
        g.setdefault(_k, _v)
    # Publish the twin's top-level imports onto mod: in-tree they were the
    # srt file's own module globals, and rebound copies resolve via mod.
    mod.Dict = Dict
    mod.Union = Union
    mod._health_generate_sampling_params = rebind(_health_generate_sampling_params, mod)
    assert not health_generate.__code__.co_freevars
    assert not mod.health_generate.__code__.co_freevars
    mod.health_generate.__code__ = health_generate.__code__
