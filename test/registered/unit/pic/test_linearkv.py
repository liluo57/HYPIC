from sglang.srt.pic.policy import PICCompose, PICStateInit, POLICIES
from sglang.srt.pic.selector import merge_online_positions, select_epic_positions


def test_linearkv_policy_does_not_use_transition():
    policy = POLICIES["linearkv"]
    assert policy.state_init is PICStateInit.LAST_BLOCK
    assert policy.compose is PICCompose.ADDITION  # legacy field only
    assert policy.is_linearkv
    assert not policy.uses_transition
    assert policy.rope
    assert policy.selector == "epic"


def test_epic_positions_are_chunk_local_and_prompt_ordered():
    hits = [
        (20, 27, b"c2"),
        (0, 8, b"c1"),
        (40, 41, b"c3"),
    ]
    selected = select_epic_positions(hits, tokens_per_chunk=2)
    assert selected[(0, 8)] == [0, 1]
    assert selected[(20, 27)] == [20, 21]
    assert selected[(40, 41)] == [40]
    assert merge_online_positions([(8, 20), (27, 40)], selected) == [
        0,
        1,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
    ]


def test_epic_ratio_selects_chunk_prefix_and_keeps_short_chunks_nonempty():
    hits = [(0, 10, b"c1"), (20, 24, b"c2"), (40, 41, b"c3")]
    selected = select_epic_positions(hits, ratio=0.2)
    assert selected[(0, 10)] == [0, 1]
    assert selected[(20, 24)] == [20]
    assert selected[(40, 41)] == [40]


def test_epic_selector_rejects_both_budget_forms():
    try:
        select_epic_positions([(0, 8, b"c1")], tokens_per_chunk=2, ratio=0.2)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("EPIC selector accepted both budget forms")
