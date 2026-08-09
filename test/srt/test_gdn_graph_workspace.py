from unittest.mock import patch

from sglang.srt.layers.attention.linear import gdn_backend


def test_graph_workspace_is_capture_only():
    with (
        patch.object(gdn_backend._bi_decode_mod, "BI_GDN_DECODE_GRAPH", True),
        patch.object(gdn_backend, "get_is_capture_mode", return_value=True),
    ):
        assert gdn_backend._use_bi_gdn_graph_workspace()

    with (
        patch.object(gdn_backend._bi_decode_mod, "BI_GDN_DECODE_GRAPH", True),
        patch.object(gdn_backend, "get_is_capture_mode", return_value=False),
    ):
        assert not gdn_backend._use_bi_gdn_graph_workspace()
