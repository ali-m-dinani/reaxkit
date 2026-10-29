import json

import pytest

from reaxkit.core.runtime.stack_summary import StackSummary


@pytest.mark.parametrize("header", ["Timeout (0:01:00)!", "Stack sample (elapsed 60.002 s) | timestamp"])
def test_sample_counts_recursion_and_thread_separation(tmp_path, header):
    summary = StackSummary(60, 1)
    for line in [
        header, "Thread 0x1 (most recent call first):",
        '  File "/src/work.py", line 10 in solve',
        '  File "/src/work.py", line 20 in solve',
        '  File "/src/cli.py", line 30 in main',
        "Thread 0x2 (most recent call first):",
        '  File "/lib/threading.py", line 10 in wait',
        header, "Thread 0x1 (most recent call first):",
        '  File "/src/reader.py", line 40 in read',
        '  File "/src/cli.py", line 30 in main',
    ]:
        summary.feed(line)
    summary.write(tmp_path)
    result = json.loads((tmp_path / "stack_summary.json").read_text())
    assert result["samples"] == 2
    rows = {row["function"]: row for row in result["functions"]}
    assert rows["solve"]["self_samples"] == rows["solve"]["inclusive_samples"] == 1
    assert rows["main"]["self_samples"] == 0
    assert rows["main"]["estimated_inclusive_seconds"] == 120
    assert rows["read"]["estimated_self_seconds"] == 60
    assert rows["wait"]["thread"] == "0x2"
    assert rows["solve"]["thread"] == "main"
    text = (tmp_path / "stack_summary.txt").read_text()
    assert "NOT invocation counts" in text
    assert "solve (work.py)" in text


def test_no_samples_does_not_invent_timings(tmp_path):
    summary = StackSummary(60, 1)
    summary.feed('  File "noise.py", line 1 in ignored')
    summary.write(tmp_path)
    assert summary.rows() == []
    assert "No Python stack frames sampled" in (tmp_path / "stack_summary.txt").read_text()
