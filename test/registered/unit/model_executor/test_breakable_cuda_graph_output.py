from dataclasses import dataclass

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    _copy_output,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@dataclass(frozen=True)
class _Slots:
    positions: torch.Tensor
    valid: torch.Tensor
    capacity: int


@dataclass(frozen=True)
class _Output:
    values: torch.Tensor
    slots: _Slots
    status: torch.Tensor
    label: str


def _output(value: int, *, label: str) -> _Output:
    return _Output(
        values=torch.tensor([[float(value)]]),
        slots=_Slots(
            positions=torch.tensor([value]),
            valid=torch.tensor([value >= 0]),
            capacity=1,
        ),
        status=torch.tensor(value, dtype=torch.int32),
        label=label,
    )


def test_copy_output_preserves_nested_frozen_dataclass_tensor_storage():
    dst = _output(1, label="capture")
    src = _output(2, label="replay")
    values = dst.values
    slots = dst.slots
    positions = dst.slots.positions
    valid = dst.slots.valid
    status = dst.status

    result = _copy_output(dst, src)

    assert result is dst
    assert dst.values is values
    assert dst.slots is slots
    assert dst.slots.positions is positions
    assert dst.slots.valid is valid
    assert dst.status is status
    torch.testing.assert_close(dst.values, src.values)
    torch.testing.assert_close(dst.slots.positions, src.slots.positions)
    torch.testing.assert_close(dst.slots.valid, src.slots.valid)
    torch.testing.assert_close(dst.status, src.status)
    assert dst.label == "replay"


def test_breakable_backend_buffers_and_slices_logits_processor_output():
    backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
    warmup = LogitsProcessorOutput(
        next_token_logits=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        hidden_states=torch.arange(4, dtype=torch.float32).reshape(2, 2),
        customized_info={"contract": "xorl"},
    )

    assert backend._output_rows(warmup, 4) == 2
    buffer = backend._alloc_full_buffer(warmup, 4)
    assert isinstance(buffer, LogitsProcessorOutput)
    assert buffer.next_token_logits.shape == (4, 3)
    assert buffer.hidden_states.shape == (4, 2)

    backend._copy_output_to_buffer(warmup, buffer, 2)
    stored = backend._slice_output(buffer, 2)
    assert isinstance(stored, LogitsProcessorOutput)
    torch.testing.assert_close(stored.next_token_logits, warmup.next_token_logits)
    torch.testing.assert_close(stored.hidden_states, warmup.hidden_states)
    assert stored.customized_info == {"contract": "xorl"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
