from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.entrypoints import http_server
from sglang.srt.managers import tokenizer_manager as tokenizer_manager_module
from sglang.srt.managers.io_struct import ShutdownReq
from sglang.srt.managers.scheduler_pp_mixin import (
    _PP_IDLE_PROXY_SLOT,
    SchedulerPPMixin,
    _pp_resolve_proxy_slot,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_nonzero_node_returns_after_scheduler_children_exit(monkeypatch) -> None:
    scheduler_init_result = Mock()
    monkeypatch.setattr(
        http_server.Engine,
        "_launch_subprocesses",
        Mock(
            return_value=(
                None,
                None,
                Mock(),
                scheduler_init_result,
                None,
                [],
            )
        ),
    )
    setup_http = Mock(side_effect=AssertionError("remote node must not start HTTP"))
    monkeypatch.setattr(http_server, "_setup_and_run_http_server", setup_http)

    http_server.launch_server(SimpleNamespace(node_rank=1))

    setup_http.assert_not_called()


def test_sigterm_watchdog_does_not_sigkill_owning_process(monkeypatch) -> None:
    stop_http_server = Mock()
    manager = SimpleNamespace(
        gracefully_exit=True,
        rid_to_state={},
        server_status=tokenizer_manager_module.ServerStatus.Up,
        _subprocess_watchdog=None,
        _http_server_shutdown_callback=stop_http_server,
        _dispatch_to_scheduler=Mock(),
    )
    kill_tree = Mock()
    monkeypatch.setattr(
        tokenizer_manager_module, "collect_scheduler_processes", lambda: []
    )
    monkeypatch.setattr(tokenizer_manager_module, "kill_process_tree", kill_tree)

    asyncio.run(tokenizer_manager_module.TokenizerManager.sigterm_watchdog(manager))

    stop_http_server.assert_called_once_with()
    kill_tree.assert_called_once_with(
        tokenizer_manager_module.os.getpid(), include_parent=False
    )


@pytest.mark.parametrize("is_last_rank", [False, True])
def test_pp_shutdown_reaches_last_stage_before_exit(is_last_rank: bool) -> None:
    send_work = Mock()
    fake = SimpleNamespace(
        gracefully_exit=False,
        pp_loop_size=1,
        ps=SimpleNamespace(pp_size=2),
        pp_group=SimpleNamespace(is_last_rank=is_last_rank),
        request_receiver=SimpleNamespace(
            recv_requests=Mock(return_value=[ShutdownReq()])
        ),
        send_req_work=[],
    )

    def init_pp_loop_state() -> None:
        fake.running_mbs = [None]
        fake.last_mbs = [None]

    def process_input_requests(_requests) -> None:
        fake.gracefully_exit = True

    fake.init_pp_loop_state = init_pp_loop_state
    fake.process_input_requests = process_input_requests
    fake._pp_commit_comm_work = Mock()
    fake._pp_send_pyobj_to_next_stage = Mock(return_value=send_work)

    SchedulerPPMixin.event_loop_pp(fake)

    if is_last_rank:
        fake._pp_send_pyobj_to_next_stage.assert_not_called()
    else:
        fake._pp_send_pyobj_to_next_stage.assert_called_once()
        assert fake._pp_commit_comm_work.call_args_list[-1].args == (send_work,)


def test_pp_idle_proxy_heartbeat_preserves_empty_ring_slot() -> None:
    proxy = PPProxyTensors({_PP_IDLE_PROXY_SLOT: True, "__msg_type__": "proxy"})

    assert _pp_resolve_proxy_slot(proxy, None, pp_rank=1, mb_id=0) is None
    assert _PP_IDLE_PROXY_SLOT not in proxy.tensors


def test_pp_proxy_heartbeat_preserves_active_ring_slot() -> None:
    proxy = PPProxyTensors({"hidden_states": object(), "__msg_type__": "proxy"})
    batch = object()

    assert _pp_resolve_proxy_slot(proxy, batch, pp_rank=1, mb_id=1) is proxy


@pytest.mark.parametrize(
    ("upstream_idle", "local_batch"),
    [(True, object()), (False, None)],
)
def test_pp_proxy_heartbeat_rejects_ring_slot_drift(
    upstream_idle: bool, local_batch: object | None
) -> None:
    proxy = PPProxyTensors(
        {_PP_IDLE_PROXY_SLOT: True} if upstream_idle else {"hidden_states": object()}
    )

    with pytest.raises(RuntimeError, match="lost batch-slot alignment"):
        _pp_resolve_proxy_slot(proxy, local_batch, pp_rank=1, mb_id=1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
