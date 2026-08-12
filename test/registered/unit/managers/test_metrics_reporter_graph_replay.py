import logging
from types import SimpleNamespace

from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_first_decode_graph_replay_is_logged_before_periodic_interval(caplog):
    reporter = object.__new__(SchedulerMetricsReporter)
    reporter.current_scheduler_metrics_enabled = False
    reporter.is_stats_logging_rank = True
    reporter._logged_first_decode_graph_replay = False
    reporter._graph_backend_label = "cuda graph"
    reporter.forward_ct_decode = 1
    reporter.decode_log_interval = 40
    reporter.scheduler_status_logger = None
    reporter.scheduler = SimpleNamespace(waiting_queue=[])
    batch = SimpleNamespace(reqs=[object()] * 32)

    with caplog.at_level(logging.INFO):
        reporter.report_decode_stats(True, running_batch=batch)
        reporter.report_decode_stats(True, running_batch=batch)

    messages = [
        record.getMessage()
        for record in caplog.records
        if "first graph replay" in record.getMessage()
    ]
    assert messages == [
        "Decode batch [first graph replay], #running-req: 32, "
        "cuda graph: True, #queue-req: 0"
    ]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
