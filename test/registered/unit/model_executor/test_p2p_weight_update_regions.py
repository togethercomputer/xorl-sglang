from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.model_executor.p2p_weight_update import (
    annotate_p2p_locators_with_memory_handles,
    p2p_capped_block_registration_regions,
    p2p_locator_registration_regions,
    p2p_qwen35_full_attention_hf_name,
    p2p_qwen35_linear_attn_conv1d_locators,
    p2p_qwen35_linear_attn_qkvz_locators,
    p2p_regions_from_memory_snapshot,
    p2p_register_regions,
    p2p_segment_regions_from_memory_snapshot,
)
from sglang.srt.model_executor.p2p_weight_update_receiver import (
    P2PWeightUpdateReceiver,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def test_p2p_receiver_hf_name_helpers_support_bound_calls():
    receiver = object.__new__(P2PWeightUpdateReceiver)

    assert (
        receiver._derive_hf_name("model.layers.0.self_attn.qkv_proj", "q_proj")
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert receiver._guess_merged_subnames("model.layers.0.mlp.gate_up_proj", 2) == [
        "gate_proj",
        "up_proj",
    ]


def test_p2p_regions_from_memory_snapshot_selects_active_blocks():
    locators = [
        {"hf_name": "param_a", "ptr": 0x1010, "nbytes": 0x20},
        {"hf_name": "param_b", "ptr": 0x2040, "nbytes": 0x10},
    ]
    snapshot = [
        {
            "address": 0x1000,
            "blocks": [
                {"size": 0x80, "state": "active_allocated"},
                {"size": 0x40, "state": "inactive"},
                {"address": 0x2000, "size": 0x80, "state": "active_allocated"},
            ],
        }
    ]

    regions, missing = p2p_regions_from_memory_snapshot(locators, snapshot)

    assert regions == [(0x1000, 0x1080), (0x2000, 0x2080)]
    assert missing == []

    assert annotate_p2p_locators_with_memory_handles(locators, regions) == []
    assert locators[0]["memory_handle"] == 0x1000
    assert locators[0]["memory_nbytes"] == 0x80
    assert locators[1]["memory_handle"] == 0x2000
    assert locators[1]["memory_nbytes"] == 0x80


def test_p2p_regions_from_memory_snapshot_reports_spanning_locator():
    locators = [{"hf_name": "param_spans_blocks", "ptr": 0x1030, "nbytes": 0x40}]
    snapshot = [
        {
            "blocks": [
                {"address": 0x1000, "size": 0x40, "state": "active_allocated"},
                {"address": 0x1040, "size": 0x80, "state": "active_allocated"},
            ]
        }
    ]

    regions, missing = p2p_regions_from_memory_snapshot(locators, snapshot)

    assert regions == [(0x1000, 0x1040), (0x1040, 0x10C0)]
    assert missing == ["param_spans_blocks ptr=0x1030 nbytes=64"]


def test_p2p_segment_regions_from_memory_snapshot_selects_full_segments():
    locators = [
        {"hf_name": "param_a", "ptr": 0x1010, "nbytes": 0x20},
        {"hf_name": "param_b", "ptr": 0x2100, "nbytes": 0x10},
    ]
    snapshot = [
        {
            "address": 0x1000,
            "total_size": 0x400,
            "blocks": [
                {"address": 0x1000, "size": 0x80, "state": "active_allocated"},
                {"address": 0x1080, "size": 0x380, "state": "inactive"},
            ],
        },
        {
            "address": 0x2000,
            "total_size": 0x400,
            "blocks": [
                {"address": 0x2000, "size": 0x80, "state": "inactive"},
                {"address": 0x2080, "size": 0x100, "state": "active_allocated"},
            ],
        },
        {
            "address": 0x3000,
            "total_size": 0x400,
            "blocks": [
                {"address": 0x3000, "size": 0x80, "state": "active_allocated"},
            ],
        },
    ]

    regions, missing = p2p_segment_regions_from_memory_snapshot(locators, snapshot)

    assert regions == [(0x1000, 0x1400), (0x2000, 0x2400)]
    assert missing == []

    assert annotate_p2p_locators_with_memory_handles(locators, regions) == []
    assert locators[0]["memory_handle"] == 0x1000
    assert locators[0]["memory_nbytes"] == 0x400
    assert locators[1]["memory_handle"] == 0x2000
    assert locators[1]["memory_nbytes"] == 0x400


def test_p2p_segment_regions_from_memory_snapshot_uses_block_extent_fallback():
    locators = [{"hf_name": "param_a", "ptr": 0x1080, "nbytes": 0x40}]
    snapshot = [
        {
            "address": 0x1000,
            "blocks": [
                {"address": 0x1000, "size": 0x80, "state": "inactive"},
                {"address": 0x1080, "size": 0x80, "state": "active_allocated"},
            ],
        }
    ]

    regions, missing = p2p_segment_regions_from_memory_snapshot(locators, snapshot)

    assert regions == [(0x1000, 0x1100)]
    assert missing == []


def test_capped_block_registration_never_covers_unmapped_tail():
    # Regression for XORL-252: whole-segment [address, address+total_size)
    # registration walked past the mapped extent (an inactive/reserved tail)
    # and ibv_reg_mr returned EFAULT ("Bad address") -> -202. The capped mode
    # must register only the mapped ('active_allocated') block, never the tail.
    locators = [{"hf_name": "w", "ptr": 0x1000, "nbytes": 0x40}]
    snapshot = [
        {
            "address": 0x1000,
            "total_size": 0x100000,  # reserved extent, mostly unmapped
            "blocks": [
                {"address": 0x1000, "size": 0x80, "state": "active_allocated"},
                {"address": 0x1080, "size": 0xFFF80, "state": "inactive"},
            ],
        }
    ]

    regions, missing = p2p_capped_block_registration_regions(
        locators, snapshot, max_region_bytes=1 << 30
    )

    assert regions == [(0x1000, 0x1080)]
    assert missing == []


def test_capped_block_registration_merges_contiguous_blocks_within_cap():
    locators = [
        {"hf_name": "a", "ptr": 0x1000, "nbytes": 0x40},
        {"hf_name": "b", "ptr": 0x1100, "nbytes": 0x40},
    ]
    snapshot = [
        {
            "address": 0x1000,
            "total_size": 0x400,
            "blocks": [
                {"address": 0x1000, "size": 0x100, "state": "active_allocated"},
                {"address": 0x1100, "size": 0x100, "state": "active_allocated"},
            ],
        }
    ]

    regions, missing = p2p_capped_block_registration_regions(
        locators, snapshot, max_region_bytes=1 << 30
    )

    # Two physically contiguous mapped blocks collapse into one region.
    assert regions == [(0x1000, 0x1200)]
    assert missing == []
    assert annotate_p2p_locators_with_memory_handles(locators, regions) == []
    assert locators[0]["memory_handle"] == 0x1000
    assert locators[1]["memory_handle"] == 0x1000


def test_capped_block_registration_does_not_span_freed_gap():
    locators = [
        {"hf_name": "a", "ptr": 0x1000, "nbytes": 0x40},
        {"hf_name": "b", "ptr": 0x1200, "nbytes": 0x40},
    ]
    snapshot = [
        {
            "address": 0x1000,
            "total_size": 0x400,
            "blocks": [
                {"address": 0x1000, "size": 0x100, "state": "active_allocated"},
                {"address": 0x1100, "size": 0x100, "state": "inactive"},
                {"address": 0x1200, "size": 0x100, "state": "active_allocated"},
            ],
        }
    ]

    regions, missing = p2p_capped_block_registration_regions(
        locators, snapshot, max_region_bytes=1 << 30
    )

    # The freed block in the middle may be unmapped under an expandable
    # allocator, so the two live blocks must stay separate regions.
    assert regions == [(0x1000, 0x1100), (0x1200, 0x1300)]
    assert missing == []


def test_capped_block_registration_caps_region_size_between_blocks():
    locators = [
        {"hf_name": "a", "ptr": 0x1000, "nbytes": 0x40},
        {"hf_name": "b", "ptr": 0x1100, "nbytes": 0x40},
        {"hf_name": "c", "ptr": 0x1200, "nbytes": 0x40},
    ]
    snapshot = [
        {
            "address": 0x1000,
            "total_size": 0x400,
            "blocks": [
                {"address": 0x1000, "size": 0x100, "state": "active_allocated"},
                {"address": 0x1100, "size": 0x100, "state": "active_allocated"},
                {"address": 0x1200, "size": 0x100, "state": "active_allocated"},
            ],
        }
    ]

    regions, missing = p2p_capped_block_registration_regions(
        locators, snapshot, max_region_bytes=0x200
    )

    # Cap of 0x200 -> first two blocks (0x200 total) form one region, the
    # split happens on a block boundary so no locator is straddled.
    assert regions == [(0x1000, 0x1200), (0x1200, 0x1300)]
    assert missing == []


def test_capped_block_registration_emits_oversized_block_whole():
    locators = [{"hf_name": "big", "ptr": 0x1000, "nbytes": 0x300}]
    snapshot = [
        {
            "address": 0x1000,
            "total_size": 0x400,
            "blocks": [
                {"address": 0x1000, "size": 0x400, "state": "active_allocated"},
            ],
        }
    ]

    regions, missing = p2p_capped_block_registration_regions(
        locators, snapshot, max_region_bytes=0x100
    )

    # A single block larger than the cap is registered whole (an allocation is
    # never split mid-block), so the (mapped) locator is still covered.
    assert regions == [(0x1000, 0x1400)]
    assert missing == []


def test_capped_block_registration_excludes_non_weight_runs():
    locators = [{"hf_name": "w", "ptr": 0x2000, "nbytes": 0x40}]
    snapshot = [
        {
            "address": 0x1000,
            "total_size": 0x400,
            "blocks": [
                # active but holds no weight locator -> must not be registered
                {"address": 0x1000, "size": 0x100, "state": "active_allocated"},
            ],
        },
        {
            "address": 0x2000,
            "total_size": 0x400,
            "blocks": [
                {"address": 0x2000, "size": 0x100, "state": "active_allocated"},
            ],
        },
    ]

    regions, missing = p2p_capped_block_registration_regions(
        locators, snapshot, max_region_bytes=1 << 30
    )

    assert regions == [(0x2000, 0x2100)]
    assert missing == []


def test_capped_block_registration_reports_uncovered_locator():
    # A locator that lands in an inactive (unmapped) block cannot be covered.
    locators = [{"hf_name": "ghost", "ptr": 0x1080, "nbytes": 0x40}]
    snapshot = [
        {
            "address": 0x1000,
            "total_size": 0x400,
            "blocks": [
                {"address": 0x1000, "size": 0x80, "state": "active_allocated"},
                {"address": 0x1080, "size": 0x80, "state": "inactive"},
            ],
        }
    ]

    regions, missing = p2p_capped_block_registration_regions(
        locators, snapshot, max_region_bytes=1 << 30
    )

    assert regions == []
    assert missing == ["ghost ptr=0x1080 nbytes=64"]


def test_p2p_locator_registration_regions_merge_only_overlaps():
    locators = [
        {"hf_name": "a", "ptr": 0x1000, "nbytes": 0x10},
        {"hf_name": "b", "ptr": 0x1010, "nbytes": 0x10},
        {"hf_name": "c", "ptr": 0x2000, "nbytes": 0x20},
        {"hf_name": "d", "ptr": 0x2010, "nbytes": 0x20},
    ]

    assert p2p_locator_registration_regions(locators) == [
        (0x1000, 0x1010),
        (0x1010, 0x1020),
        (0x2000, 0x2030),
    ]


def test_p2p_qwen35_linear_attn_qkvz_locators_match_fused_source_names():
    locators = p2p_qwen35_linear_attn_qkvz_locators(
        module_name="model.layers.0.linear_attn.in_proj_qkvz",
        output_sizes=[8, 8, 16, 16],
        input_size=4,
        tp_rank=1,
        tp_size=2,
        base_ptr=0x1000,
        itemsize=2,
        dtype="bfloat16",
        dp_rank=0,
        ep_rank=-1,
    )

    assert [loc["hf_name"] for loc in locators] == [
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.in_proj_z.weight",
    ]
    assert [loc["full_shape"] for loc in locators] == [
        [32, 4],
        [32, 4],
        [32, 4],
        [16, 4],
    ]
    assert [loc["slice"] for loc in locators] == [
        [[4, 8], [0, 4]],
        [[12, 16], [0, 4]],
        [[24, 32], [0, 4]],
        [[8, 16], [0, 4]],
    ]
    assert [loc["ptr"] for loc in locators] == [
        0x1000,
        0x1000 + 4 * 4 * 2,
        0x1000 + 8 * 4 * 2,
        0x1000 + 16 * 4 * 2,
    ]
    assert [loc["nbytes"] for loc in locators] == [
        4 * 4 * 2,
        4 * 4 * 2,
        8 * 4 * 2,
        8 * 4 * 2,
    ]


def test_p2p_qwen35_linear_attn_conv1d_locators_match_sharded_loader_layout():
    locators = p2p_qwen35_linear_attn_conv1d_locators(
        module_name="model.layers.0.linear_attn.conv1d",
        output_sizes=[8, 8, 16],
        input_size=4,
        tp_rank=1,
        tp_size=2,
        base_ptr=0x1000,
        itemsize=2,
        dtype="bfloat16",
        dp_rank=0,
        ep_rank=-1,
    )

    assert [loc["hf_name"] for loc in locators] == [
        "model.layers.0.linear_attn.conv1d.weight",
        "model.layers.0.linear_attn.conv1d.weight",
        "model.layers.0.linear_attn.conv1d.weight",
    ]
    assert [loc["full_shape"] for loc in locators] == [
        [32, 4],
        [32, 4],
        [32, 4],
    ]
    assert [loc["slice"] for loc in locators] == [
        [[4, 8], [0, 4]],
        [[12, 16], [0, 4]],
        [[24, 32], [0, 4]],
    ]
    assert [loc["ptr"] for loc in locators] == [
        0x1000,
        0x1000 + 4 * 4 * 2,
        0x1000 + 8 * 4 * 2,
    ]
    assert [loc["nbytes"] for loc in locators] == [
        4 * 4 * 2,
        4 * 4 * 2,
        8 * 4 * 2,
    ]


def test_p2p_qwen35_full_attention_hf_name_restores_self_attn_prefix():
    layers = ["linear_attention", "linear_attention", "linear_attention", "attention"]

    assert (
        p2p_qwen35_full_attention_hf_name(
            "model.layers.3.q_proj.weight",
            layers,
        )
        == "model.layers.3.self_attn.q_proj.weight"
    )
    assert (
        p2p_qwen35_full_attention_hf_name(
            "model.layers.3.q_norm.weight",
            layers,
        )
        == "model.layers.3.self_attn.q_norm.weight"
    )
    assert (
        p2p_qwen35_full_attention_hf_name(
            "model.layers.3.o_proj.weight_scale_inv",
            layers,
        )
        == "model.layers.3.self_attn.o_proj.weight_scale_inv"
    )


def test_p2p_qwen35_full_attention_hf_name_does_not_rewrite_other_layers():
    layers = ["linear_attention", "linear_attention", "linear_attention", "attention"]

    assert (
        p2p_qwen35_full_attention_hf_name(
            "model.layers.0.q_proj.weight",
            layers,
        )
        == "model.layers.0.q_proj.weight"
    )
    assert (
        p2p_qwen35_full_attention_hf_name(
            "model.layers.3.self_attn.q_proj.weight",
            layers,
        )
        == "model.layers.3.self_attn.q_proj.weight"
    )
    assert (
        p2p_qwen35_full_attention_hf_name(
            "model.layers.3.mlp.gate_proj.weight",
            layers,
        )
        == "model.layers.3.mlp.gate_proj.weight"
    )


def test_p2p_register_regions_strict_fails_and_deregisters_prior_regions():
    class Engine:
        def __init__(self):
            self.registered = []
            self.deregistered = []

        def register(self, ptr, nbytes, location=None):
            if ptr == 0x2000:
                return -7
            self.registered.append((ptr, nbytes))
            return 0

        def batch_deregister(self, ptrs):
            self.deregistered.extend(ptrs)
            return 0

    engine = Engine()
    ret, ptrs = p2p_register_regions(
        engine,
        [(0x1000, 0x1040), (0x2000, 0x2040), (0x3000, 0x3040)],
        strict=True,
    )

    assert ret == -7
    assert ptrs == [0x1000]
    assert engine.registered == [(0x1000, 0x40)]
    assert engine.deregistered == [0x1000]


def test_p2p_register_regions_forwards_memory_location():
    class Engine:
        def __init__(self):
            self.registered = []

        def register(self, ptr, nbytes, location=None):
            self.registered.append((ptr, nbytes, location))
            return 0

    engine = Engine()
    ret, ptrs = p2p_register_regions(
        engine,
        [(0x1000, 0x1040), (0x2000, 0x2040)],
        strict=True,
        location="cuda:3",
    )

    assert ret == 0
    assert ptrs == [0x1000, 0x2000]
    assert engine.registered == [(0x1000, 0x40, "cuda:3"), (0x2000, 0x40, "cuda:3")]


def test_p2p_register_regions_location_falls_back_for_two_arg_engine():
    class Engine:
        def __init__(self):
            self.registered = []

        def register(self, ptr, nbytes):
            self.registered.append((ptr, nbytes))
            return None

    engine = Engine()
    ret, ptrs = p2p_register_regions(
        engine,
        [(0x1000, 0x1040)],
        strict=True,
        location="cuda:3",
    )

    assert ret == 0
    assert ptrs == [0x1000]
    assert engine.registered == [(0x1000, 0x40)]


def test_mooncake_register_passes_memory_location():
    class Engine:
        def __init__(self):
            self.calls = []

        def register_memory(self, ptr, length, location=None):
            self.calls.append((ptr, length, location))
            return 0

    wrapper = object.__new__(MooncakeTransferEngine)
    wrapper.engine = Engine()

    assert wrapper.register(0x1000, 0x40, location="cuda:3") == 0
    assert wrapper.engine.calls == [(0x1000, 0x40, "cuda:3")]


def test_mooncake_register_location_falls_back_for_two_arg_engine():
    class Engine:
        def __init__(self):
            self.calls = []

        def register_memory(self, ptr, length):
            self.calls.append((ptr, length))
            return 0

    wrapper = object.__new__(MooncakeTransferEngine)
    wrapper.engine = Engine()

    assert wrapper.register(0x1000, 0x40, location="cuda:3") == 0
    assert wrapper.engine.calls == [(0x1000, 0x40)]
