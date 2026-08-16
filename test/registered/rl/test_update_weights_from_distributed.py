"""Test distributed weight updates.

This test suite simulates a distributed training environment to ensure
correct weight synchronization. On rank 0, the instruct model represents
pre-training weights, and the base model represents post-training weights.
The base model's weights are broadcasted to other ranks using the online
weight update API.

On other ranks, an engine is initialized with the instruct model, and its
parameters are verified against the Hugging Face model. After updating
weights from the distributed system, post-training weights are loaded
and verified again to ensure consistency and accuracy across the
distributed setup.
"""

import gc
import os
import queue
import random
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoModelForCausalLM

import sglang as sgl
from sglang.srt.utils import init_custom_process_group
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
)
from sglang.utils import terminate_process

register_cuda_ci(est_time=137, stage="extra-a", runner_config="2-gpu-large")
register_amd_ci(est_time=400, suite="stage-b-test-2-gpu-large-amd")

mp.set_start_method("spawn", force=True)


def verify_params_close(params1, params2, error_msg):
    """Verify if two parameter arrays are close enough."""
    try:
        assert np.allclose(np.array(params1), np.array(params2)), error_msg
    except Exception as e:
        print(f"Parameters not close for {error_msg}")
        print("Params1:", np.array(params1))
        print("Params2:", np.array(params2))
        raise e


def verify_params_not_close(params1, params2, error_msg):
    """Verify if two parameter arrays are different enough."""
    assert not np.allclose(np.array(params1), np.array(params2)), error_msg


def _warmup_broadcast(
    hf_base_model,
    state_dict_key_to_shape,
    tie_word_embeddings,
    load_format,
    group,
):
    """Run one broadcast round to warm up RCCL before timing."""
    broadcast_parameters = list(state_dict_key_to_shape.keys())
    if tie_word_embeddings:
        broadcast_parameters.remove("lm_head.weight")

    if load_format == "flattened_bucket":
        named_tensors = [
            (name, hf_base_model.get_parameter(name)) for name in broadcast_parameters
        ]
        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        flattened_tensor = bucket.get_flattened_tensor()
        torch.distributed.broadcast(flattened_tensor, src=0, group=group)
    else:
        for name in broadcast_parameters:
            torch.distributed.broadcast(
                hf_base_model.get_parameter(name),
                src=0,
                group=group,
            )


def _warmup_update(
    backend, engine, url, names, dtypes, shapes, load_format, pause_generation_mode
):
    """Run one update round to warm up RCCL before timing."""
    if backend == "Engine":
        engine.update_weights_from_distributed(
            names,
            dtypes=dtypes,
            shapes=shapes,
            group_name="test_parameter_update_group",
            load_format=load_format,
        )
    else:
        requests.post(
            f"{url}/update_weights_from_distributed",
            json={
                "names": names,
                "dtypes": dtypes,
                "shapes": shapes,
                "group_name": "test_parameter_update_group",
                "load_format": load_format,
                "flush_cache": not (pause_generation_mode == "in_place"),
            },
        )


def _require_http_success(response, endpoint):
    assert (
        response.status_code == 200
    ), f"{endpoint} returned HTTP {response.status_code}: {response.text}"
    payload = response.json()
    assert payload["success"] is True, f"{endpoint} failed: {payload}"
    return payload


def _get_server_weight(url, name, size):
    response = requests.post(
        f"{url}/get_weights_by_name",
        json={"name": name, "truncate_size": size},
        timeout=60,
    )
    assert (
        response.status_code == 200
    ), f"get_weights_by_name returned HTTP {response.status_code}: {response.text}"
    weight = response.json()
    assert isinstance(weight, list) and len(weight) == size
    return weight


def _run_two_phase_sender(
    world_size,
    master_port,
    parameter_name,
    parameter_shape,
    phase_barrier,
    teardown_barrier,
    result_queue,
):
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    torch.cuda.set_device(0)
    group = init_custom_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{master_port}",
        world_size=world_size,
        rank=0,
        group_name="test_two_phase_parameter_update_group",
    )
    trace = {}
    try:
        for cycle, value in enumerate((0.0, 1.0)):
            source = torch.full(
                parameter_shape,
                value,
                dtype=torch.bfloat16,
                device="cuda:0",
            )
            bucket = FlattenedTensorBucket(named_tensors=[(parameter_name, source)])

            # The receiver reaches this barrier only after prepare returned.
            phase_barrier.wait(timeout=300)
            trace[f"broadcast_started_{cycle}"] = time.monotonic_ns()
            torch.distributed.broadcast(
                bucket.get_flattened_tensor(),
                src=0,
                group=group,
            )
            torch.cuda.synchronize()
            trace[f"broadcast_finished_{cycle}"] = time.monotonic_ns()

            # Fence complete until the synchronous send and CUDA work finished.
            phase_barrier.wait(timeout=300)

        teardown_barrier.wait(timeout=60)
        result_queue.put(("sender", trace))
    finally:
        torch.distributed.destroy_process_group(group)


def _run_two_phase_server(
    world_size,
    master_port,
    server_url,
    parameter_name,
    parameter_shape,
    phase_barrier,
    teardown_barrier,
    result_queue,
):
    process = popen_launch_server(
        DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        server_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=(
            "--base-gpu-id",
            "1",
            "--tp-size",
            "1",
            "--cuda-graph-max-bs-decode",
            "2",
        ),
    )
    group_initialized = False
    trace = {}
    weights = {}
    try:
        weights["pre"] = _get_server_weight(
            server_url, parameter_name, parameter_shape[0]
        )
        response = requests.post(
            f"{server_url}/init_weights_update_group",
            json={
                "master_address": "localhost",
                "master_port": str(master_port),
                "rank_offset": 1,
                "world_size": world_size,
                "group_name": "test_two_phase_parameter_update_group",
                "backend": "nccl",
            },
            timeout=120,
        )
        _require_http_success(response, "init_weights_update_group")
        group_initialized = True

        for cycle in range(2):
            response = requests.post(
                f"{server_url}/prepare_weights_update",
                json={
                    "buckets": [
                        {
                            "names": [parameter_name],
                            "dtypes": ["bfloat16"],
                            "shapes": [parameter_shape],
                        }
                    ],
                    "num_buckets": 1,
                    "group_name": "test_two_phase_parameter_update_group",
                    "load_format": "flattened_bucket",
                    "transport": "nccl_broadcast",
                },
                timeout=120,
            )
            _require_http_success(response, "prepare_weights_update")
            trace[f"prepare_returned_{cycle}"] = time.monotonic_ns()

            # Release the sender only after the real HTTP prepare has returned.
            phase_barrier.wait(timeout=300)
            # Wait for the sender's synchronous broadcast and CUDA fence.
            phase_barrier.wait(timeout=300)

            trace[f"complete_started_{cycle}"] = time.monotonic_ns()
            response = requests.post(
                f"{server_url}/complete_weights_update",
                json={
                    "group_name": "test_two_phase_parameter_update_group",
                    "flush_cache": True,
                    "transport": "nccl_broadcast",
                },
                timeout=360,
            )
            _require_http_success(response, "complete_weights_update")
            trace[f"complete_returned_{cycle}"] = time.monotonic_ns()
            weights[f"post_{cycle}"] = _get_server_weight(
                server_url, parameter_name, parameter_shape[0]
            )

        teardown_barrier.wait(timeout=60)
        result_queue.put(("server", {"trace": trace, "weights": weights}))
    finally:
        if group_initialized:
            try:
                requests.post(
                    f"{server_url}/destroy_weights_update_group",
                    json={"group_name": "test_two_phase_parameter_update_group"},
                    timeout=60,
                )
            except requests.RequestException:
                pass
        terminate_process(process)


def _run_two_phase_update_process(
    rank,
    world_size,
    master_port,
    server_url,
    parameter_name,
    parameter_shape,
    phase_barrier,
    teardown_barrier,
    result_queue,
):
    if rank == 0:
        _run_two_phase_sender(
            world_size,
            master_port,
            parameter_name,
            parameter_shape,
            phase_barrier,
            teardown_barrier,
            result_queue,
        )
    else:
        _run_two_phase_server(
            world_size,
            master_port,
            server_url,
            parameter_name,
            parameter_shape,
            phase_barrier,
            teardown_barrier,
            result_queue,
        )


def init_process(
    rank,
    world_size,
    param_queue,
    truncate_size,
    state_dict_key_to_shape,
    tp_size,
    model_name,
    backend,
    checking_parameters,
    tie_word_embeddings,
    load_format,
    barrier,
    pause_generation_mode,
):
    torch.cuda.set_device(rank)

    if rank == 0:
        init_process_hf(
            rank,
            world_size,
            param_queue,
            truncate_size,
            model_name,
            checking_parameters,
            tie_word_embeddings,
            state_dict_key_to_shape,
            load_format,
            barrier,
        )
    elif rank in [1, 2]:
        init_process_sgl(
            rank,
            world_size,
            param_queue,
            truncate_size,
            model_name,
            checking_parameters,
            tie_word_embeddings,
            state_dict_key_to_shape,
            backend,
            tp_size,
            load_format,
            barrier,
            pause_generation_mode,
        )


def init_process_hf(
    rank,
    world_size,
    param_queue,
    truncate_size,
    model_name,
    checking_parameters,
    tie_word_embeddings,
    state_dict_key_to_shape,
    load_format,
    barrier,
):
    # These two environment variables are very important
    # to avoid unexpected behaviors of CUDA and NCCL.
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    # Load model and get parameters
    hf_instruct_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        tie_word_embeddings=tie_word_embeddings,
    ).to("cuda:0")
    base_model_name = model_name.replace("-Instruct", "")
    hf_base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="bfloat16",
        tie_word_embeddings=tie_word_embeddings,
    ).to("cuda:0")

    hf_instruct_params = []
    hf_base_params = []

    print("[hf] get parameter in hf instruct model and base model")
    for parameter_name in checking_parameters:
        hf_instruct_params.append(
            hf_instruct_model.get_parameter(parameter_name)[:truncate_size]
            .cpu()
            .detach()
            .float()
            .numpy()
            .tolist()
        )
        hf_base_params.append(
            hf_base_model.get_parameter(parameter_name)[:truncate_size]
            .cpu()
            .detach()
            .float()
            .numpy()
            .tolist()
        )

    param_queue.put(("hf_instruct_params", hf_instruct_params))
    param_queue.put(("hf_base_params", hf_base_params))

    # Init weight update group for rank 0 (the training engine in RLHF).
    port = 60000 + int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")[0]) * 100
    init_method = f"tcp://localhost:{port}"
    print(f"[hf] {rank=} {world_size=} init custom process group. {init_method=}")
    group = init_custom_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        group_name="test_parameter_update_group",
    )
    torch.cuda.synchronize()
    barrier.wait()

    # Warmup: trigger RCCL initialization so it's excluded from timing
    if is_in_amd_ci():
        _warmup_broadcast(
            hf_base_model,
            state_dict_key_to_shape,
            tie_word_embeddings,
            load_format,
            group,
        )
        torch.cuda.synchronize()

    time_begin_broadcast = time.perf_counter()

    # The last parameter is lm_head.weight, which is tied
    # with embed_tokens.weight. Actually, we only need
    # to broadcast embed_tokens.weight once.
    broadcast_parameters = list(state_dict_key_to_shape.keys())
    if tie_word_embeddings:
        broadcast_parameters.remove("lm_head.weight")

    if load_format == "flattened_bucket":
        named_tensors = [
            (parameter_name, hf_base_model.get_parameter(parameter_name))
            for parameter_name in broadcast_parameters
        ]
        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        flattened_tensor = bucket.get_flattened_tensor()
        torch.distributed.broadcast(flattened_tensor, src=0, group=group)
    else:
        # Broadcast all the weights from the training
        # engine to other ranks (inference engine).
        for parameter_name in broadcast_parameters:
            torch.distributed.broadcast(
                hf_base_model.get_parameter(parameter_name),
                src=0,
                group=group,
            )
    torch.cuda.synchronize()
    time_end_broadcast = time.perf_counter()

    # Measure the latency of broadcasting/weights update.
    broadcast_time = time_end_broadcast - time_begin_broadcast
    print(f"[hf] {rank=} {broadcast_time=:.3f}s")
    param_queue.put(("broadcast_time", broadcast_time))

    # Destroy process group and release related resource
    torch.distributed.destroy_process_group(group)

    # Delete the huggingface models to free up memory.
    del hf_instruct_model
    del hf_base_model
    gc.collect()
    torch.cuda.empty_cache()


def init_process_sgl(
    rank,
    world_size,
    param_queue,
    truncate_size,
    model_name,
    checking_parameters,
    tie_word_embeddings,
    state_dict_key_to_shape,
    backend,
    tp_size,
    load_format,
    barrier,
    pause_generation_mode,
):
    torch.cuda.set_device(rank)
    torch.cuda.synchronize()
    base_gpu_id = 1 if rank == 1 else 1 + tp_size
    if backend == "Engine":
        print(f"[sgl] rank {rank} init engine")
        engine = sgl.Engine(
            model_path=model_name,
            base_gpu_id=base_gpu_id,
            tp_size=tp_size,
            cuda_graph_max_bs_decode=2,
        )
    else:
        if rank == 1:
            url = DEFAULT_URL_FOR_TEST
        else:
            host, _, port = DEFAULT_URL_FOR_TEST.rpartition(":")
            url = ":".join([host, str(int(port) + 10000)])

        print(f"[sgl] rank {rank} init server on url: {url}")
        process = popen_launch_server(
            model_name,
            url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--base-gpu-id",
                str(base_gpu_id),
                "--tp-size",
                str(tp_size),
                "--cuda-graph-max-bs-decode",
                2,
            ),
        )
    torch.cuda.synchronize()

    # Get weights of instruct model, i.e. pre-training weights.
    instruct_params = []
    for parameter_name in checking_parameters:
        instruct_params.append(
            engine.get_weights_by_name(parameter_name, truncate_size)
            if backend == "Engine"
            else requests.get(
                f"{url}/get_weights_by_name",
                json={"name": parameter_name, "truncate_size": truncate_size},
            ).json()
        )

    param_queue.put((f"sgl_dp_{rank}_instruct_params", instruct_params))

    port = 60000 + int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")[0]) * 100

    # Init weight update group with the training engine.
    if backend == "Engine":
        engine.init_weights_update_group(
            master_address="localhost",
            master_port=str(port),
            rank_offset=base_gpu_id,
            world_size=world_size,
            group_name="test_parameter_update_group",
            backend="nccl",
        )
    else:
        requests.post(
            f"{url}/init_weights_update_group",
            json={
                "master_address": "localhost",
                "master_port": str(port),
                "rank_offset": base_gpu_id,
                "world_size": world_size,
                "group_name": "test_parameter_update_group",
                "backend": "nccl",
            },
        )

    if pause_generation_mode in ["in_place", "retract"]:

        def run_decode(max_new_tokens=32):
            response = requests.post(
                url + "/generate",
                json={
                    "text": f"Question: {random.randint(0, 100)},The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": max_new_tokens,
                        "ignore_eos": True,
                    },
                },
            )
            return response.json()

        with ThreadPoolExecutor(32) as executor:
            for _ in range(32):
                executor.submit(run_decode, 1000)
            time.sleep(2)

    # The last parameter is lm_head.weight, which is tied
    # with embed_tokens.weight. Actually, we only need
    # to update embed_tokens.weight once.
    tie_word_embeddings = (
        True if model_name == DEFAULT_SMALL_MODEL_NAME_FOR_TEST else False
    )
    update_parameters = list(state_dict_key_to_shape.keys())
    if tie_word_embeddings:
        update_parameters.remove("lm_head.weight")

    # Get weights from the training engine and update the inference engine.
    names = [parameter_name for parameter_name in update_parameters]
    dtypes = [torch.bfloat16 if backend == "Engine" else "bfloat16"] * len(names)
    shapes = [state_dict_key_to_shape[parameter_name] for parameter_name in names]

    if pause_generation_mode in ["in_place", "retract"]:
        requests.post(
            url + "/pause_generation",
            json={"mode": pause_generation_mode},
        )
    torch.cuda.synchronize()
    barrier.wait()

    # Warmup: trigger RCCL initialization so it's excluded from timing
    if is_in_amd_ci():
        _warmup_update(
            backend,
            engine if backend == "Engine" else None,
            url if backend != "Engine" else None,
            names,
            dtypes,
            shapes,
            load_format,
            pause_generation_mode,
        )
        torch.cuda.synchronize()

    time_begin_update = time.perf_counter()
    if backend == "Engine":
        engine.update_weights_from_distributed(
            names,
            dtypes=dtypes,
            shapes=shapes,
            group_name="test_parameter_update_group",
            load_format=load_format,
        )
    else:
        requests.post(
            f"{url}/update_weights_from_distributed",
            json={
                "names": names,
                "dtypes": dtypes,
                "shapes": shapes,
                "group_name": "test_parameter_update_group",
                "load_format": load_format,
                "flush_cache": not (pause_generation_mode == "in_place"),
            },
        )
    torch.cuda.synchronize()
    time_end_update = time.perf_counter()
    if pause_generation_mode in ["in_place", "retract"]:
        requests.post(
            url + "/continue_generation",
            json={},
        )

        # discard unfinished requests to save test overhead
        time.sleep(2)
        requests.post(
            url + "/pause_generation",
            json={"mode": "abort"},
        )

    # Measure the latency of broadcast/weights update.
    update_time = time_end_update - time_begin_update
    print(
        f"[sgl] fully update model_name {model_name} rank {rank} parameter from distributed time: {update_time:.3f}s"
    )
    param_queue.put((f"update_sgl_dp_{rank}_time", update_time))

    # Get the weights of post-training model after weights update for correctness check.
    base_params = []
    for parameter_name in checking_parameters:
        if backend == "Engine":
            base_params.append(
                engine.get_weights_by_name(parameter_name, truncate_size)
            )
        else:
            base_params.append(
                requests.get(
                    f"{url}/get_weights_by_name",
                    json={
                        "name": parameter_name,
                        "truncate_size": truncate_size,
                    },
                ).json()
            )
    param_queue.put((f"sgl_dp_{rank}_base_params", base_params))

    if backend == "Engine":
        success, _ = engine.destroy_weights_update_group(
            group_name="test_parameter_update_group",
        )
        assert success is True
    else:
        response = requests.post(
            f"{url}/destroy_weights_update_group",
            json={
                "group_name": "test_parameter_update_group",
            },
        )
        assert response.status_code == 200

    # Shutdown the engine or terminate the server process.
    if backend == "Engine":
        engine.shutdown()
    else:
        terminate_process(process)


def assert_tied_weights(params_list, message, should_be_tied):
    for params in params_list:
        if should_be_tied:
            assert np.allclose(params[0], params[-1]), message
        else:
            assert not np.allclose(params[0], params[-1]), message


def test_update_weights_from_distributed(
    tp_size,
    dp_size,
    model_name,
    backend,
    state_dict_key_to_shape,
    truncate_size,
    checking_parameters,
    load_format=None,
    pause_generation_mode=None,
):
    tie_word_embeddings = (
        True if model_name == DEFAULT_SMALL_MODEL_NAME_FOR_TEST else False
    )

    print(
        f"Testing model: {model_name} tp_size: {tp_size}, dp_size: {dp_size} backend: {backend}"
    )
    param_queue = mp.Queue()
    results = {}
    barrier = mp.Barrier(1 + dp_size)

    context = mp.spawn(
        init_process,
        args=(
            1 + tp_size * dp_size,
            param_queue,
            truncate_size,
            state_dict_key_to_shape,
            tp_size,
            model_name,
            backend,
            checking_parameters,
            tie_word_embeddings,
            load_format,
            barrier,
            pause_generation_mode,
        ),
        nprocs=1 + dp_size,
        join=False,
    )

    while len(results) < 3 * (1 + dp_size):
        try:
            key, value = param_queue.get(timeout=5)
            results[key] = value
        except Exception:
            if all(not p.is_alive() for p in context.processes):
                break

    context.join()

    if len(results) != 3 * (1 + dp_size):
        raise RuntimeError(
            f"Expected {3 * (1 + dp_size)} parameters but got {len(results)}"
        )

    params = {
        "hf_instruct": results.get("hf_instruct_params"),
        "hf_base": results.get("hf_base_params"),
        "sgl_dp_1_instruct": results.get("sgl_dp_1_instruct_params"),
        "sgl_dp_1_base": results.get("sgl_dp_1_base_params"),
        "broadcast_time": results.get("broadcast_time"),
        "update_sgl_dp_1_time": results.get("update_sgl_dp_1_time"),
    }

    if dp_size == 2:
        dp2_params = {
            "sgl_dp_2_instruct": results.get("sgl_dp_2_instruct_params"),
            "sgl_dp_2_base": results.get("sgl_dp_2_base_params"),
            "update_sgl_dp_2_time": results.get("update_sgl_dp_2_time"),
        }
        assert all(v is not None for v in dp2_params.values())
        params.update(dp2_params)

    # Check the correctness of weights update by verifying
    # the weights of instruct model and base model.
    for i in range(len(params["hf_instruct"])):
        verify_params_close(
            params["hf_instruct"][i],
            params["sgl_dp_1_instruct"][i],
            f"sgl_dp_1_instruct_params rank {i}",
        )

        verify_params_close(
            params["hf_base"][i],
            params["sgl_dp_1_base"][i],
            f"sgl_dp_1_base_params rank {i}",
        )

        verify_params_not_close(
            params["hf_instruct"][i],
            params["hf_base"][i],
            f"hf_instruct_params rank {i}",
        )

        if dp_size == 2:
            verify_params_close(
                params["hf_base"][i],
                params["sgl_dp_2_base"][i],
                f"sgl_dp_2_base_params rank {i}",
            )
            verify_params_close(
                params["hf_instruct"][i],
                params["sgl_dp_2_instruct"][i],
                f"sgl_dp_2_instruct_params rank {i}",
            )

    assert len(params["hf_instruct"]) == len(
        params["hf_base"]
    ), "hf_instruct_params and hf_base_params have different lengths"

    # Check if the weights of lm_head are tied with embed_tokens.
    params_to_check = [
        (
            params["hf_instruct"],
            "lm_head.weight is not tied with embed_tokens.weight",
        ),
        (
            params["hf_base"],
            "lm_head.weight is not tied with embed_tokens.weight",
        ),
        (
            params["sgl_dp_1_instruct"],
            "lm_head.weight is not tied with embed_tokens.weight",
        ),
        (
            params["sgl_dp_1_base"],
            "lm_head.weight is not tied with embed_tokens.weight",
        ),
    ]

    if dp_size == 2:
        params_to_check.extend(
            [
                (
                    params["sgl_dp_2_instruct"],
                    "lm_head.weight is not tied with embed_tokens.weight",
                ),
                (
                    params["sgl_dp_2_base"],
                    "lm_head.weight is not tied with embed_tokens.weight",
                ),
            ]
        )

    assert_tied_weights(
        [params for params, _ in params_to_check],
        (
            "lm_head.weight is not tied with embed_tokens.weight"
            if tie_word_embeddings
            else "lm_head.weight is tied with embed_tokens.weight"
        ),
        tie_word_embeddings,
    )

    # Time limit for broadcast and update on CI is 3 / 6
    # On local H100, it's 1 / 2
    time_limit = 3 if model_name == DEFAULT_SMALL_MODEL_NAME_FOR_TEST else 6

    assert (
        params["broadcast_time"] < time_limit
    ), f"broadcast_time exceeds time limit {time_limit}s"

    assert (
        params["update_sgl_dp_1_time"] < time_limit
    ), f"update_sgl_dp_one_time exceeds time limit {time_limit}s"

    if dp_size == 2:
        assert (
            params["update_sgl_dp_2_time"] < time_limit
        ), f"update_sgl_dp_two_time exceeds time limit {time_limit}s"

    # Delete the context and close the parameter queue.
    del context
    param_queue.close()
    param_queue.join_thread()
    gc.collect()
    torch.cuda.empty_cache()


class TestUpdateWeightsFromDistributed(CustomTestCase):
    def test_prepare_receive_complete_weights_update(self):
        assert torch.cuda.device_count() >= 2, "At least 2 GPUs are required"

        parameter_name = "model.norm.weight"
        parameter_shape = [
            AutoConfig.from_pretrained(DEFAULT_SMALL_MODEL_NAME_FOR_TEST).hidden_size
        ]
        master_port = find_available_port(29500)
        server_url = f"http://127.0.0.1:{find_available_port(31000)}"
        phase_barrier = mp.Barrier(2)
        teardown_barrier = mp.Barrier(2)
        result_queue = mp.Queue()

        context = mp.spawn(
            _run_two_phase_update_process,
            args=(
                2,
                master_port,
                server_url,
                parameter_name,
                parameter_shape,
                phase_barrier,
                teardown_barrier,
                result_queue,
            ),
            nprocs=2,
            join=False,
        )
        deadline = time.monotonic() + DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH + 360

        def terminate_workers():
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=30)

        # Drain both records while workers are live. A multiprocessing Queue
        # flushes through a feeder thread, and joining before reading can
        # deadlock when the observed weight payload exceeds the pipe capacity.
        results = {}
        while len(results) < 2:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                terminate_workers()
                self.fail("Timed out waiting for two-phase weight update results")
            try:
                role, payload = result_queue.get(timeout=min(1, remaining))
            except queue.Empty:
                if context.join(timeout=0):
                    self.fail(
                        "Two-phase weight update workers exited before publishing "
                        "both results"
                    )
                continue
            self.assertNotIn(role, results, f"Duplicate two-phase result role: {role}")
            results[role] = payload

        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                terminate_workers()
                self.fail("Timed out waiting for the two-phase weight update test")

        self.assertEqual(set(results), {"sender", "server"})
        sender_trace = results["sender"]
        server_trace = results["server"]["trace"]
        weights = results["server"]["weights"]

        for cycle in range(2):
            self.assertLessEqual(
                server_trace[f"prepare_returned_{cycle}"],
                sender_trace[f"broadcast_started_{cycle}"],
            )
            self.assertLessEqual(
                sender_trace[f"broadcast_started_{cycle}"],
                sender_trace[f"broadcast_finished_{cycle}"],
            )
            self.assertLessEqual(
                sender_trace[f"broadcast_finished_{cycle}"],
                server_trace[f"complete_started_{cycle}"],
            )
            self.assertLessEqual(
                server_trace[f"complete_started_{cycle}"],
                server_trace[f"complete_returned_{cycle}"],
            )

        expected_first = np.zeros(parameter_shape, dtype=np.float32)
        expected_second = np.ones(parameter_shape, dtype=np.float32)
        self.assertFalse(np.array_equal(weights["pre"], expected_first))
        np.testing.assert_array_equal(weights["post_0"], expected_first)
        np.testing.assert_array_equal(weights["post_1"], expected_second)
        self.assertFalse(np.array_equal(weights["post_0"], weights["post_1"]))

        result_queue.close()
        result_queue.join_thread()

    def test_update_weights_from_distributed(self):
        assert torch.cuda.device_count() >= 2, "At least 2 GPUs are required"
        # test_suits : tp, dp, model_name, backend
        if is_in_ci():
            mode = random.choice(["Engine", "Server"])
            if mode == "Server":
                pause_generation_mode = random.choice(["in_place", "retract"])
            else:
                pause_generation_mode = None
            load_format = random.choice(["flattened_bucket", None])
            test_suits = [
                (
                    1,
                    1,
                    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                    mode,
                    pause_generation_mode,
                    load_format,
                ),
            ]
        else:
            test_suits = [
                (
                    1,
                    1,
                    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                    "Engine",
                    None,
                    random.choice(["flattened_bucket", None]),
                ),
                (
                    1,
                    1,
                    DEFAULT_MODEL_NAME_FOR_TEST,
                    "Sever",
                    random.choice(["in_place", "retract"]),
                    random.choice(["flattened_bucket", None]),
                ),
            ]

            if torch.cuda.device_count() >= 4:
                test_suits.extend(
                    [
                        (
                            2,
                            1,
                            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                            "Engine",
                            None,
                            random.choice(["flattened_bucket", None]),
                        ),
                        (
                            1,
                            2,
                            DEFAULT_MODEL_NAME_FOR_TEST,
                            "Server",
                            random.choice(["in_place", "retract"]),
                            random.choice(["flattened_bucket", None]),
                        ),
                    ]
                )

            if torch.cuda.device_count() >= 5:
                test_suits.extend(
                    [
                        (
                            2,
                            2,
                            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                            "Engine",
                            None,
                            random.choice(["flattened_bucket", None]),
                        ),
                        (
                            2,
                            2,
                            DEFAULT_MODEL_NAME_FOR_TEST,
                            "Server",
                            random.choice(["in_place", "retract"]),
                            random.choice(["flattened_bucket", None]),
                        ),
                    ]
                )

        model_state_dict_shapes = {}
        test_models = [test_suit[2] for test_suit in test_suits]

        for model_name in test_models:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="bfloat16"
            ).to("cuda:0")
            state_dict = model.state_dict()
            state_dict_keys = list(state_dict.keys())
            model_state_dict_shapes[model_name] = {
                key: state_dict[key].shape for key in state_dict_keys
            }
            del model
            gc.collect()
            torch.cuda.empty_cache()

        truncate_size = 10
        checking_parameters = [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.2.self_attn.k_proj.weight",
            "model.layers.3.self_attn.v_proj.weight",
            "model.layers.4.self_attn.o_proj.weight",
            "model.layers.5.mlp.gate_proj.weight",
            "model.layers.6.mlp.up_proj.weight",
            "model.layers.7.mlp.down_proj.weight",
            "model.layers.8.post_attention_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]

        for (
            tp_size,
            dp_size,
            model_name,
            backend,
            pause_generation_mode,
            load_format,
        ) in test_suits:
            test_update_weights_from_distributed(
                tp_size,
                dp_size,
                model_name,
                backend,
                model_state_dict_shapes[model_name],
                truncate_size,
                checking_parameters,
                load_format,
                pause_generation_mode,
            )


if __name__ == "__main__":
    unittest.main()
