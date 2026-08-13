from types import SimpleNamespace

import pytest

from sglang.srt.layers.deep_gemm_wrapper.compile_utils import (
    _temporary_deep_gemm_compile_mode,
)


def test_compile_mode_is_optional():
    with _temporary_deep_gemm_compile_mode(SimpleNamespace()):
        pass


def test_compile_mode_is_restored():
    calls = []
    deep_gemm = SimpleNamespace(
        get_compile_mode=lambda: 7,
        set_compile_mode=calls.append,
    )

    with _temporary_deep_gemm_compile_mode(deep_gemm):
        assert calls == [1]

    assert calls == [1, 7]


def test_compile_mode_is_restored_after_failure():
    calls = []
    deep_gemm = SimpleNamespace(
        get_compile_mode=lambda: 3,
        set_compile_mode=calls.append,
    )

    with pytest.raises(ValueError, match="probe"):
        with _temporary_deep_gemm_compile_mode(deep_gemm):
            raise ValueError("probe")

    assert calls == [1, 3]


@pytest.mark.parametrize("missing", ["getter", "setter"])
def test_compile_mode_api_must_be_paired(missing):
    deep_gemm = SimpleNamespace()
    if missing != "getter":
        deep_gemm.get_compile_mode = lambda: 0
    if missing != "setter":
        deep_gemm.set_compile_mode = lambda _: None

    with pytest.raises(
        RuntimeError, match="both get_compile_mode and set_compile_mode"
    ):
        with _temporary_deep_gemm_compile_mode(deep_gemm):
            pass
