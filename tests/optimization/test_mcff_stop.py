import pytest

from reaxkit.optimization.models import OptimizationError
from reaxkit.optimization.worker import mcff_stop_text


@pytest.mark.parametrize("current", ["0", "1", "2"])
def test_ams_2023_stop_key_preserves_other_controls(current):
    content = (
        f"           {current} !Key: 1 - tell program to use the values below, 2 - stop\n"
        "  0.00169170 !Beta\n"
        "    NONE     !Command: one of NONE, WALK, JUMPWALK, MINIMIZE, JUMPMINI\n"
        "      3.4523 !ScaleFactor\n"
    )
    expected = content[:11] + "2" + content[12:]
    assert mcff_stop_text(content) == expected
    assert mcff_stop_text(expected) == expected


@pytest.mark.parametrize("content, expected", [
    ("0 StopKey\n", "1 StopKey\n"),
    ("StopKey 0\n", "StopKey 1\n"),
    ("1 StopKey\n", "1 StopKey\n"),
])
def test_documented_stopkey_layouts(content, expected):
    assert mcff_stop_text(content) == expected


@pytest.mark.parametrize("content", [
    "0\n", "0 !Key: unknown semantics\n", "3 StopKey\n",
    "0 StopKey\nStopKey 0\n", "10 StopKey\n",
])
def test_unknown_or_ambiguous_stop_keys_are_rejected(content):
    with pytest.raises(OptimizationError, match="refusing to guess"):
        mcff_stop_text(content)
