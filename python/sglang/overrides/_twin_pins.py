"""Pins for upstream symbols that overlay twins replace with edited copies.

A twin that *replaces* an upstream function or method with an edited copy has
frozen that symbol at the upstream version the copy was derived from. Nothing
in git will ever conflict on it again — which is exactly the overlay's danger:
an upstream sync can fix or change the original and the copy silently stops
tracking it.

This registry makes that drift loud. Every replaced symbol is pinned to the
sha256 of its *upstream* source (as it stands in the `srt/` file on disk —
the file keeps the upstream text; twins only patch the live module objects).
`test/registered/unit/test_overlay_twin_pins.py` recomputes each hash from the
current tree and fails on mismatch, turning "upstream changed a method your
twin copies" into a visible CI failure with instructions.

When that test fires after an upstream sync:
1. Diff the upstream symbol against the twin's copy.
2. Re-derive the copy (fold the upstream change into the twin).
3. Re-pin: update the hash here (`python -m sglang.overrides._twin_pins`
   prints current hashes for all pinned symbols).

Purely *additive* twin attachments (new helpers, new methods that do not
replace an upstream def) need no pin — there is nothing upstream to drift
from.
"""

import ast
import hashlib
import pathlib
import textwrap

# (repo-relative srt file, dotted qualname within the module) -> sha256 of the
# dedented source segment of that def/class as extracted by ast.
#
# Populated by the xorl exact-serving zero-srt conversion; keep sorted by file.
PINS: dict[tuple[str, str], str] = {
    (
        "python/sglang/srt/entrypoints/http_server.py",
        "health_generate",
    ): "57ff9f1740db497b3aa891b50f013ae8b68872e17ca04be71faaa6401ddbdd55",
    (
        "python/sglang/srt/server_args.py",
        "ServerArgs.__post_init__",
    ): "7f4a2bb8523dd7bf74b6772a721a64a155e289dd9768414913af4cb29af5e98a",
    (
        "python/sglang/srt/server_args.py",
        "ServerArgs._handle_model_specific_adjustments",
    ): "03054cbc513ac53eae6a3b81681edbc560eea73f80ab1eeb3898c0ad7166f248",
    (
        "python/sglang/srt/server_args.py",
        "ServerArgs._handle_return_hidden_states_mode",
    ): "33006319cb9c8fa5983e03b4142d614c711522a2e0733b4cea004c940c3e744d",
    (
        "python/sglang/srt/layers/attention/linear/gdn_backend.py",
        "GDNAttnBackend.__init__",
    ): "fd779bd08ea9090f2e0b24a2f34d5f6aaf6c1d2708aed84201e0c63dbfb93f0c",
    (
        "python/sglang/srt/layers/attention/linear/gdn_backend.py",
        "GDNAttnBackend._replayssm_fold_target_verify",
    ): "71b6dda5000ccf8df2eb174412a488526b3c38284b7c1e6575f63c4df143a9b2",
    (
        "python/sglang/srt/layers/attention/linear/gdn_backend.py",
        "GDNAttnBackend._replayssm_target_verify",
    ): "a783787148b933c0e53dd05b0a7cc3cc1be81d1da67431dc8cf931bfb0324ef6",
    (
        "python/sglang/srt/layers/attention/linear/gdn_backend.py",
        "GDNAttnBackend.forward_decode",
    ): "92f36735ea0e26d5ae519fa7db499f6e55303099af2ece65ece775e8d7ac150f",
    (
        "python/sglang/srt/layers/attention/linear/gdn_backend.py",
        "GDNAttnBackend.forward_extend",
    ): "0da74b91368ca37c4b22cff416c55f24490c54fbb121d0d5946661fbb31a7240",
    (
        "python/sglang/srt/layers/attention/linear/gdn_backend.py",
        "GDNAttnBackend.init_forward_metadata",
    ): "5c6e224ae51a6bd182b8cfc99105272d7ffadca175bcf26a96391d8f33b293f8",
    (
        "python/sglang/srt/layers/layernorm.py",
        "GemmaRMSNorm.__init__",
    ): "4cc3a4f59accad28850d260f1c80fcbf2027849a7e0db29c038eb067c63b4f0c",
    (
        "python/sglang/srt/layers/layernorm.py",
        "GemmaRMSNorm.forward_cuda",
    ): "3624848c8fd860970d2950dfe9ffa651ab98440f333fdc0d9e7e39d4fcd6a847",
    (
        "python/sglang/srt/layers/layernorm.py",
        "GemmaRMSNorm.forward_with_allreduce_fusion",
    ): "222b86def0f2750ac4c754292cf552e6077e4987cc6073aadda179e092ce9437",
    (
        "python/sglang/srt/layers/layernorm.py",
        "GemmaRMSNorm.forward_with_allreduce_fusion_quant_per_group",
    ): "4e87612072942c2186ab6c90010a0764a6805e7e902e699f76938fe45351dc3a",
    (
        "python/sglang/srt/layers/layernorm.py",
        "RMSNorm.__init__",
    ): "674329fbf7620ca33b6d0edb2b415f694b1dfc8916b2d4e0f7a0875a1c7390f1",
    (
        "python/sglang/srt/layers/layernorm.py",
        "RMSNorm.forward_aiter",
    ): "a23d3a80d36cfeaedc9209bcfe40c5f4940c5688b8f57c0101bab220252cc6be",
    (
        "python/sglang/srt/layers/layernorm.py",
        "RMSNorm.forward_cuda",
    ): "2211965024630772d4b2d345f2a66db5757a4dd4b08417927641acce8bf956d8",
    (
        "python/sglang/srt/layers/layernorm.py",
        "RMSNorm.forward_hip",
    ): "f35afffce5789863b1f2134dc7fe73089b54edae05e806bcf866bda7bb8177bc",
    (
        "python/sglang/srt/layers/layernorm.py",
        "RMSNorm.forward_xpu",
    ): "bb8daaf3b8ef38de9c9a42aa93036171eb9dec2b1f45a01a08fd64455ac0a8e5",
    (
        "python/sglang/srt/layers/logits_processor.py",
        "LogitsMetadata.from_forward_batch",
    ): "eb1260d4455c01405bb46e0ae6834c29bc7479c27825a3e2342ea4d2aafacfcd",
    (
        "python/sglang/srt/layers/logits_processor.py",
        "LogitsProcessor.__init__",
    ): "da0cf525f38683e709fe26c7d2bccf229ac95ae490493b702a1dad451078f089",
    (
        "python/sglang/srt/layers/logits_processor.py",
        "LogitsProcessor._compute_lm_head",
    ): "bf3d3f43431b1e2acd7a5826b80dd8528c8533848116228a7a30124d685236d1",
    (
        "python/sglang/srt/layers/logits_processor.py",
        "LogitsProcessor._copy_logits_to_buffer",
    ): "557c5c9bf33300d7412a58c8f9e85a5ef257f17a82778d43dcc1bdba91f0f0bb",
    (
        "python/sglang/srt/layers/logits_processor.py",
        "LogitsProcessor._gather_dp_attn_hidden_states",
    ): "ddb594b6bcf0923a5c6a84096fc7a77a55a8526302d281fbf809d4b643059b15",
    (
        "python/sglang/srt/layers/logits_processor.py",
        "LogitsProcessor._get_pruned_states",
    ): "46ef77fc24bd619be4b8262c7763c26edb930e7b9543809a3c68d4f69656abfe",
    (
        "python/sglang/srt/layers/logits_processor.py",
        "LogitsProcessor.forward",
    ): "93af985593a724672956fbb6b0c8356dd77de48be51b3f9f48816f33ee7980cb",
    (
        "python/sglang/srt/layers/rotary_embedding/base.py",
        "RotaryEmbedding.__init__",
    ): "d7c9f8029d15326be8b046b289cb756100c3b833a0c84c120fef24586a5c6666",
    (
        "python/sglang/srt/layers/rotary_embedding/base.py",
        "RotaryEmbedding._compute_cos_sin_cache",
    ): "d7db0acd2b005e3cdc3d5f4e10c38b76f1e0e5c9af981f07442239ceac2bab71",
    (
        "python/sglang/srt/layers/rotary_embedding/base.py",
        "RotaryEmbedding._compute_inv_freq",
    ): "dae4bcdced5125756cd1d1e23d6538643cc74a36112866534f576c069931c921",
    (
        "python/sglang/srt/layers/rotary_embedding/base.py",
        "RotaryEmbedding._ensure_cos_sin_cache_length",
    ): "0ff8b9bb6c2966b82fbf53122abd62565472a31e85df37b9ff99397407c9a0e1",
    (
        "python/sglang/srt/layers/rotary_embedding/factory.py",
        "get_rope",
    ): "88a15e33dceb7571819475f6742265af4df56edd82c10d4553f5f273055f7cf3",
    (
        "python/sglang/srt/layers/rotary_embedding/mrope.py",
        "MRotaryEmbedding.forward_native",
    ): "0f8d5b790d2ebcb3c12081e6f4b85e04d3fe2c18f8de772cfb52c3b07ff284e2",
    (
        "python/sglang/srt/layers/rotary_embedding/mrope.py",
        "YaRNScalingMRotaryEmbedding._compute_cos_sin_cache",
    ): "0c3402920b7a75b44919db6f184696559f3f6967dac464483538450bd47f8e91",
    (
        "python/sglang/srt/layers/rotary_embedding/rope_variant.py",
        "DeepseekScalingRotaryEmbedding._compute_cos_sin_cache",
    ): "ad2eb0c43ff06847efd7397f62da2878d5ccdf1fe7f9a26de6f97f6431180676",
    (
        "python/sglang/srt/layers/rotary_embedding/rope_variant.py",
        "DeepseekScalingRotaryEmbedding._compute_inv_freq",
    ): "98de8b3bba27612eb23615225eac9e36326eaa9a70f4c55fe71d84fb10253fd5",
    (
        "python/sglang/srt/layers/rotary_embedding/rope_variant.py",
        "DynamicNTKAlphaRotaryEmbedding._compute_cos_sin_cache",
    ): "3f92b4d221ce4e47059a8c2f59adac0b4d66fde9ff7efb0bb8596736562ecbec",
    (
        "python/sglang/srt/layers/rotary_embedding/rope_variant.py",
        "DynamicNTKScalingRotaryEmbedding._compute_cos_sin_cache",
    ): "fc375cfb2901b3abfd8b91460bd4ba19d3bc29f8719bd2a4625aceba291f2c91",
    (
        "python/sglang/srt/layers/rotary_embedding/yarn.py",
        "YaRNScalingRotaryEmbedding._compute_cos_sin_cache",
    ): "0c3402920b7a75b44919db6f184696559f3f6967dac464483538450bd47f8e91",
    (
        "python/sglang/srt/layers/sampler.py",
        "Sampler.__init__",
    ): "a7bf36c6fcde4f56573598cdc6f82988fb21fc834b8dd540c6cbf4f06be18d2e",
    (
        "python/sglang/srt/layers/sampler.py",
        "Sampler._forward_ascend_backend",
    ): "2cdb0c7764425613435dfc4b1640f21bd22a363a5dfee97b498c562cdfc4ce2b",
    (
        "python/sglang/srt/layers/sampler.py",
        "Sampler._sample_from_logprobs",
    ): "472044362e83377cae02426e79abba01618976f20853c10d600a4ffc285c029d",
    (
        "python/sglang/srt/layers/sampler.py",
        "Sampler.forward",
    ): "62c779868a6f7083516f4745c46c2611140c99689aafc0ccf21b94f05098e146",
    (
        "python/sglang/srt/managers/io_struct.py",
        "GenerateReqInput.__getitem__",
    ): "2dd56f801149caf057ffc79c753107b5ab45d1e92c1fcec100c7565a4b1cb5d1",
    (
        "python/sglang/srt/managers/io_struct.py",
        "GenerateReqInput._validate_inputs",
    ): "9d32eb4555dcbf6a77a0fb49746d3f469900f701448afa47f6d5b63ef3a5ac68",
    (
        "python/sglang/srt/managers/tokenizer_manager.py",
        "ReqState",
    ): "a21390113c3816e35cc34dffb80eca2f81fab0515a67a79e8bb3ffd0b82c7078",
    (
        "python/sglang/srt/managers/tokenizer_manager.py",
        "TokenizerManager._create_tokenized_object",
    ): "73bce132ae96f2dc3757d360c3c84ac03f4f79536920aaf2d12860fcce238bd1",
    (
        "python/sglang/srt/managers/tokenizer_manager.py",
        "TokenizerManager._handle_batch_output",
    ): "4adfb1cf333c597230e8ae239b3d435261bffb856d1837d747c329179a625830",
    (
        "python/sglang/srt/managers/tokenizer_manager.py",
        "TokenizerManager._validate_one_request",
    ): "5fb935cea26fd6af4a77a886749607458e420622aaac4c284b525d44d80a3007",
    (
        "python/sglang/srt/managers/tokenizer_manager.py",
        "TokenizerManager.add_logprob_to_meta_info",
    ): "0c5c248ef548d6490aaaad5af6b9f48a8c6c4ef4dc95cbae6cec74a141442036",
    (
        "python/sglang/srt/model_executor/forward_batch_info.py",
        "ForwardBatch",
    ): "1dd60647f4bdf1d99ff98d58af4bcdcd776aac80397c1f79b17bb1ff85a299a3",
    (
        "python/sglang/srt/models/qwen2.py",
        "Qwen2MLP.__init__",
    ): "20627c1f116613e94486d8eca1fe9e0cff1ab44fea83ca0610e739ddbce4d254",
    (
        "python/sglang/srt/models/qwen2.py",
        "Qwen2Model.__init__",
    ): "42882fc3c7a878de4c901b9ab7422f8c7ddc35cc4325863c16bf0b813c632b48",
    (
        "python/sglang/srt/models/qwen2.py",
        "Qwen2Model.forward",
    ): "3fbae7476c2cc3257ff6c059854750e9b65993620a9e9d6d456676e05896cf47",
    (
        "python/sglang/srt/models/qwen2.py",
        "Qwen2Model.load_kv_cache_scales",
    ): "26fc1cd5139085e9679ffe3f201ccce852f30568a678dfaa536206b5937ba0c4",
    (
        "python/sglang/srt/models/qwen3.py",
        "Qwen3Attention.__init__",
    ): "519ab50563935a4627f9c462d5567532ca8411a62aca356608041b74e5465d8f",
    (
        "python/sglang/srt/models/qwen3.py",
        "Qwen3DecoderLayer.__init__",
    ): "6700e3b24ba8a402907339366b77cc027ddf734f6c96fc2ca38ebe360e122ec9",
    (
        "python/sglang/srt/models/qwen3_5.py",
        "Qwen3_5AttentionDecoderLayer.__init__",
    ): "0c971e445d2ed3f4037602a553b8df4bdb92704f97e439945a93102d56b0b310",
    (
        "python/sglang/srt/models/qwen3_5.py",
        "Qwen3_5AttentionDecoderLayer.forward_prepare_fused_gate",
    ): "17fa7596ac413a0431deb4eaec9d80705d50bf4d682059ef9378a52a24b12d9a",
    (
        "python/sglang/srt/models/qwen3_5.py",
        "Qwen3_5AttentionDecoderLayer.forward_prepare_native",
    ): "3a026ef682deeccc94939fe4fc150fa156546a9f39f0c8f00447a39ed33becd8",
    (
        "python/sglang/srt/models/qwen3_5.py",
        "Qwen3_5AttentionDecoderLayer.self_attention",
    ): "c080ade3b633a61610c475c3f30d4b9ad58c61a8e03761396e25b8cc28ba2907",
    (
        "python/sglang/srt/models/qwen3_5.py",
        "Qwen3_5ForCausalLM.__init__",
    ): "336cfdaab23d6f9302764277d9211a0738c952d49b7264f5cb6b280acabfb934",
    (
        "python/sglang/srt/models/qwen3_5.py",
        "Qwen3_5LinearDecoderLayer.__init__",
    ): "21b002558e1b2d6e6c6efce396176a0dcfd01dc97a013b056ae79a87eac26a7a",
}


def _repo_root() -> pathlib.Path:
    # .../python/sglang/overrides/_twin_pins.py -> repo root is 3 up from
    # "python".
    return pathlib.Path(__file__).resolve().parents[3]


def extract_source(rel_path: str, qualname: str) -> str:
    """Dedented source of ``qualname`` (e.g. ``Cls.method``) in ``rel_path``."""
    path = _repo_root() / rel_path
    tree = ast.parse(path.read_text())
    node = tree
    for part in qualname.split("."):
        for child in ast.iter_child_nodes(node):
            if (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and child.name == part
            ):
                node = child
                break
        else:
            raise LookupError(f"{qualname!r} not found in {rel_path}")
    segment = ast.get_source_segment(path.read_text(), node)
    if segment is None:
        raise LookupError(f"no source segment for {qualname!r} in {rel_path}")
    return textwrap.dedent(segment)


def source_hash(rel_path: str, qualname: str) -> str:
    return hashlib.sha256(extract_source(rel_path, qualname).encode()).hexdigest()


def main() -> None:
    for (rel_path, qualname), pinned in sorted(PINS.items()):
        current = source_hash(rel_path, qualname)
        marker = "OK  " if current == pinned else "DRIFT"
        print(
            f"{marker} {rel_path}:{qualname}\n    pinned  {pinned}\n    current {current}"
        )


if __name__ == "__main__":
    main()
