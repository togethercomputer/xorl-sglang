from sglang.kernels.ops.attention.flash_attn.cute.paged_kv import (
    _page_entries_per_thread,
)


def test_partial_sm90_paged_kv_tile_has_one_register_entry_per_thread():
    assert _page_entries_per_thread(64, 128) == 1


def test_paged_kv_register_entries_use_ceiling_division():
    assert _page_entries_per_thread(128, 128) == 1
    assert _page_entries_per_thread(129, 128) == 2
