"""RL-target RoPE caches must match the trainer's full CPU construction."""

import os
import unittest
from contextlib import contextmanager

import torch

from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.layers.rotary_embedding.base import LinearScalingRotaryEmbedding
from sglang.srt.layers.rotary_embedding.mrope import (
    Ernie4_5_VLRotaryEmbedding,
    MRotaryEmbedding,
    YaRNScalingMRotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
    DynamicNTKAlphaRotaryEmbedding,
    DynamicNTKScalingRotaryEmbedding,
    Llama3RotaryEmbedding,
    Llama4VisionRotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.yarn import YaRNScalingRotaryEmbedding
from sglang.srt.models.grok import ScalingRotaryEmbedding
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-test-1-gpu-small")

ROTARY_DIM = 128
BASE = 1_000_000


def _reference_cache(length: int, device: str) -> torch.Tensor:
    inv_freq = 1.0 / (
        BASE ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM)
    )
    positions = torch.arange(length, dtype=torch.float32)
    frequencies = torch.einsum("i,j -> ij", positions, inv_freq)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(device=device)


@contextmanager
def _default_device(device: str):
    previous = torch.get_default_device()
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(previous)


def _default_cuda_device():
    return _default_device("cuda")


def _set_rl_target(rl_target):
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", rl_on_policy_target=rl_target)
    )


def _make_rope(length: int, *, rl_target: str | None) -> RotaryEmbedding:
    _set_rl_target(rl_target)
    with _default_cuda_device():
        return RotaryEmbedding(
            head_size=ROTARY_DIM,
            rotary_dim=ROTARY_DIM,
            max_position_embeddings=length,
            base=BASE,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )


# Every RotaryEmbedding subclass, with arguments that build a small table. The
# provenance sweep below asserts the spec covers every subclass that overrides
# table construction, so a new override cannot be added without landing here.
_SUBCLASS_SPECS = {
    RotaryEmbedding: ((128, 128, 512, 10000, True, torch.bfloat16), {}),
    LinearScalingRotaryEmbedding: (
        (128, 128, 512, 10000, True, [1.0, 2.0], torch.bfloat16),
        {},
    ),
    MRotaryEmbedding: (
        (128, 128, 512, 10000, True, torch.bfloat16),
        {"mrope_section": [16, 24, 24]},
    ),
    Ernie4_5_VLRotaryEmbedding: (
        (128, 128, 512, 10000, True, torch.bfloat16),
        {"mrope_section": [16, 24, 24]},
    ),
    YaRNScalingRotaryEmbedding: (
        (128, 128, 512, 10000, True, 4.0, torch.bfloat16),
        {},
    ),
    YaRNScalingMRotaryEmbedding: (
        (128, 128, 512, 10000, True, 4.0, torch.bfloat16),
        {"mrope_section": [16, 24, 24]},
    ),
    DeepseekScalingRotaryEmbedding: (
        (128, 64, 512, 10000, True, 4.0, torch.bfloat16),
        {"mscale": 1.0, "mscale_all_dim": 1.0},
    ),
    Llama3RotaryEmbedding: (
        (128, 128, 512, 500000, True, torch.bfloat16, 8.0, 1.0, 4.0, 8192),
        {},
    ),
    Llama4VisionRotaryEmbedding: ((128, 128, 256, 10000, True, torch.bfloat16), {}),
    DynamicNTKAlphaRotaryEmbedding: (
        (128, 128, 512, 10000, True, 2.0, torch.bfloat16),
        {},
    ),
    DynamicNTKScalingRotaryEmbedding: (
        (128, 128, 512, 10000, True, 4.0, torch.bfloat16),
        {},
    ),
    ScalingRotaryEmbedding: ((128, 128, 512, 10000, True, 4.0, torch.bfloat16), {}),
}


def _all_rotary_subclasses(root=RotaryEmbedding):
    found = {root}
    for child in root.__subclasses__():
        found |= _all_rotary_subclasses(child)
    return found


def _build(cls, *, rl_target, ambient):
    args, kwargs = _SUBCLASS_SPECS[cls]
    _set_rl_target(rl_target)
    with _default_device(ambient):
        return cls(*args, **kwargs)


class TestRlTargetRopeCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("needs CUDA GPU")
        cls.old_disable_compile = os.environ.get("SGLANG_DISABLE_ROPE_COMPILE")
        os.environ["SGLANG_DISABLE_ROPE_COMPILE"] = "1"

    @classmethod
    def tearDownClass(cls):
        if cls.old_disable_compile is None:
            os.environ.pop("SGLANG_DISABLE_ROPE_COMPILE", None)
        else:
            os.environ["SGLANG_DISABLE_ROPE_COMPILE"] = cls.old_disable_compile
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_rl_target_cache_matches_full_cpu_reference(self):
        rope = _make_rope(40960, rl_target="xorl")
        expected = _reference_cache(40960, device="cuda")

        self.assertEqual(rope.cos_sin_cache.device.type, "cuda")
        self.assertTrue(torch.equal(rope.cos_sin_cache, expected))
        self.assertTrue(
            torch.equal(
                rope.cos_sin_cache[[593, 1725, 2402]].bfloat16(),
                expected[[593, 1725, 2402]].bfloat16(),
            )
        )

    def test_rl_target_extension_matches_cpu_reference(self):
        rope = _make_rope(512, rl_target="xorl")
        rope._ensure_cos_sin_cache_length(4096)
        expected = _reference_cache(rope.cos_sin_cache.shape[0], device="cuda")

        self.assertGreaterEqual(rope.cos_sin_cache.shape[0], 4097)
        self.assertTrue(torch.equal(rope.cos_sin_cache, expected))

    def test_non_rl_cache_keeps_device_native_construction(self):
        rope = _make_rope(4096, rl_target=None)
        inv_freq = 1.0 / (
            BASE
            ** (
                torch.arange(
                    0,
                    ROTARY_DIM,
                    2,
                    dtype=torch.float32,
                    device="cuda",
                )
                / ROTARY_DIM
            )
        )
        positions = torch.arange(4096, dtype=torch.float32, device="cuda")
        frequencies = torch.einsum("i,j -> ij", positions, inv_freq)
        expected = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)

        self.assertTrue(torch.equal(rope.cos_sin_cache, expected))


class TestRlTargetRopeCacheProvenanceIsStructural(unittest.TestCase):
    """The provenance pin must reach every subclass, not just the base class."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("needs CUDA GPU")

    @classmethod
    def tearDownClass(cls):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_spec_covers_every_table_construction_override(self):
        overriding = {
            cls
            for cls in _all_rotary_subclasses()
            if "_compute_cos_sin_cache" in vars(cls)
            or "_build_cos_sin_cache" in vars(cls)
        }
        missing = sorted(c.__name__ for c in overriding - set(_SUBCLASS_SPECS))
        self.assertEqual(
            missing,
            [],
            "these RotaryEmbedding subclasses build their own cos/sin table but "
            "are not covered by the provenance sweep; add them to "
            f"_SUBCLASS_SPECS: {missing}",
        )

    def test_rl_target_table_is_independent_of_the_ambient_device(self):
        """The table a subclass produces must not depend on the loader's device.

        Model construction runs inside `with torch.device(...)`, so a device-less
        tensor factory in an override picks up the accelerator and evaluates
        cos/sin there. Under the RL target the table has to come out identical
        either way.
        """
        for cls in sorted(_SUBCLASS_SPECS, key=lambda c: c.__name__):
            with self.subTest(cls=cls.__name__):
                on_cuda = _build(cls, rl_target="xorl", ambient="cuda")
                on_cpu = _build(cls, rl_target="xorl", ambient="cpu")
                self.assertTrue(
                    torch.equal(
                        on_cuda.cos_sin_cache.cpu(), on_cpu.cos_sin_cache.cpu()
                    ),
                    f"{cls.__name__} cos/sin table depends on the ambient device",
                )

    def test_override_that_escapes_the_pin_is_rejected(self):
        """A future subclass that names a device explicitly must fail loudly."""

        class _EscapesThePin(RotaryEmbedding):
            def _compute_cos_sin_cache(self) -> torch.Tensor:
                inv_freq = torch.arange(
                    0, self.rotary_dim, 2, dtype=torch.float32, device="meta"
                )
                positions = torch.arange(
                    self.max_position_embeddings, dtype=torch.float32, device="meta"
                )
                return self._cos_sin_cache_rows(positions, inv_freq)

        _set_rl_target("xorl")
        with self.assertRaisesRegex(RuntimeError, "pinned to"):
            with _default_device("cpu"):
                _EscapesThePin(128, 128, 512, 10000, True, torch.bfloat16)


class TestRopeCacheGrowthUsesTheInitialRecipe(unittest.TestCase):
    """Grown rows must carry the frequencies and magnitude scale of the table."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("needs CUDA GPU")

    @classmethod
    def tearDownClass(cls):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _grow_and_split(self, cls, rl_target):
        rope = _build(cls, rl_target=rl_target, ambient="cuda")
        initial_len = int(rope.cos_sin_cache.shape[0])
        before = rope.cos_sin_cache[:initial_len].clone()
        with _default_device("cuda"):
            rope._ensure_cos_sin_cache_length(initial_len + 200)
        grown_len = int(rope.cos_sin_cache.shape[0])
        self.assertGreater(grown_len, initial_len)
        return rope, initial_len, before, rope.cos_sin_cache[initial_len:grown_len]

    def _yarn_reference(self, rope, start, stop, *, scaling_factor, mscale):
        with _default_device("cpu"):
            inv_freq = rope._compute_inv_freq(scaling_factor).cpu()
            positions = torch.arange(start, stop, dtype=torch.float32)
            frequencies = torch.einsum("i,j -> ij", positions, inv_freq)
            return torch.cat(
                (frequencies.cos() * mscale, frequencies.sin() * mscale), dim=-1
            )

    def test_yarn_mrope_growth_past_initial_length_is_numerically_correct(self):
        rope, initial_len, before, grown = self._grow_and_split(
            YaRNScalingMRotaryEmbedding, "xorl"
        )
        grown_len = initial_len + grown.shape[0]

        self.assertTrue(
            torch.equal(rope.cos_sin_cache[:initial_len], before),
            "growing the cache must not disturb the rows already in it",
        )

        expected = self._yarn_reference(
            rope,
            initial_len,
            grown_len,
            scaling_factor=rope.scaling_factor,
            mscale=rope.mscale,
        )
        self.assertTrue(
            torch.equal(grown.cpu(), expected),
            "grown rows do not match the recipe the initial table was built with",
        )

        # The two ways this went wrong must both be excluded, so the assertion
        # above cannot pass by accident: `self.base` fed to `_compute_inv_freq`
        # in place of the scaling factor, and a missing magnitude scale.
        wrong_inv_freq = self._yarn_reference(
            rope, initial_len, grown_len, scaling_factor=rope.base, mscale=rope.mscale
        )
        self.assertFalse(torch.equal(expected, wrong_inv_freq))
        no_mscale = self._yarn_reference(
            rope, initial_len, grown_len, scaling_factor=rope.scaling_factor, mscale=1.0
        )
        self.assertFalse(torch.equal(expected, no_mscale))

    def test_yarn_growth_is_correct_without_the_rl_target(self):
        rope, initial_len, _, grown = self._grow_and_split(
            YaRNScalingRotaryEmbedding, None
        )
        with _default_device("cuda"):
            inv_freq = rope._compute_inv_freq(rope.scaling_factor)
            positions = torch.arange(
                initial_len, initial_len + grown.shape[0], dtype=torch.float32
            )
            frequencies = torch.einsum("i,j -> ij", positions, inv_freq)
            expected = torch.cat(
                (
                    frequencies.cos() * rope.mscale,
                    frequencies.sin() * rope.mscale,
                ),
                dim=-1,
            )
        self.assertTrue(torch.equal(grown, expected))

    def test_dynamic_ntk_growth_keeps_the_scaled_base(self):
        for cls in (DynamicNTKAlphaRotaryEmbedding, DynamicNTKScalingRotaryEmbedding):
            with self.subTest(cls=cls.__name__):
                rope, initial_len, _, grown = self._grow_and_split(cls, "xorl")
                with _default_device("cpu"):
                    positions = torch.arange(
                        initial_len, initial_len + grown.shape[0], dtype=torch.float32
                    )
                    expected = rope._cos_sin_cache_rows(
                        positions, rope._cos_sin_cache_inv_freq().cpu()
                    )
                    unscaled = rope._cos_sin_cache_rows(
                        positions, rope._compute_inv_freq(rope.base).cpu()
                    )
                self.assertTrue(torch.equal(grown.cpu(), expected))
                self.assertFalse(torch.equal(expected, unscaled))


if __name__ == "__main__":
    unittest.main()
