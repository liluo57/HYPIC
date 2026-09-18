import pytest
from sglang.srt.server_args import ServerArgs


def test_pic_flags_exist_with_defaults():
    args = ServerArgs(model_path="dummy")
    assert args.pic_enable is False
    assert args.pic_separator_str == "<<PIC_SEP>>"
    assert args.pic_mode == "addition"
    assert args.pic_segment_min_tokens == -1


@pytest.mark.parametrize(
    "model_arch",
    [
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    ],
)
def test_pic_whitelist_covers_qwen3_5_dense_and_moe(model_arch):
    from sglang.srt.mem_cache.registry import PIC_ALLOWED_ARCHS

    assert model_arch in PIC_ALLOWED_ARCHS


def test_pic_enable_requires_chunked_prefill_disabled():
    with pytest.raises(AssertionError, match="chunked_prefill_size"):
        ServerArgs(
            model_path="Qwen/Qwen3.5-35B-A3B",
            pic_enable=True,
            chunked_prefill_size=2048,
        )
