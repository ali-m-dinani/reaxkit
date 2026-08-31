from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from reaxkit.core.platform.human_log import HumanReadableRunLog, current_human_log
from reaxkit.core.runtime.generator_runtime import (
    persist_generator_metadata,
    prepare_generator_output,
)


def test_human_log_writes_hierarchical_global_and_run_files(tmp_path: Path) -> None:
    logs_dir = tmp_path / "reaxkit_workspace" / "logs"
    trace = HumanReadableRunLog(
        logs_dir,
        command="python analysis.py xmolout --z-bins 20",
        run_id="run_example",
        request_name="Example analysis",
    )

    with trace:
        assert current_human_log() is trace
        trace.detail("input", tmp_path / "xmolout")
        with trace.step("Read trajectory data", source=tmp_path / "xmolout") as load_step:
            with trace.step("Parse xmolout frames") as parse_step:
                parse_step.detail("frames loaded", 12)
            load_step.detail("atoms per frame", 240)
        with trace.step("Write results") as output_step:
            output_step.result("workbook", tmp_path / "results.xlsx")
        trace.result("results directory", tmp_path / "results")

    assert current_human_log() is None

    global_log = logs_dir / "human_readable.log"
    run_log = logs_dir / "run_example.human.log"
    assert global_log.is_file()
    assert run_log.is_file()
    text = global_log.read_text(encoding="utf-8")
    assert text == run_log.read_text(encoding="utf-8")
    assert "REQUEST: Example analysis" in text
    assert "command: python analysis.py xmolout --z-bins 20" in text
    assert "run_id: run_example" in text
    assert "- Read trajectory data" in text
    assert "  substeps:\n          - Parse xmolout frames" in text
    assert "duration:" in text
    assert str((tmp_path / "results.xlsx").resolve()) in text
    assert "status: SUCCESS" in text

    machine_log = logs_dir / "machine_readable.jsonl"
    run_machine_log = logs_dir / "run_example.machine.jsonl"
    assert machine_log.is_file()
    assert run_machine_log.is_file()
    record = json.loads(machine_log.read_text(encoding="utf-8"))
    assert record["schema"] == "reaxkit.execution_trace"
    assert record["request"]["status"] == "success"
    assert record["run"]["run_id"] == "run_example"
    assert record["steps"][0]["name"] == "Read trajectory data"
    assert record["steps"][0]["steps"][0]["name"] == "Parse xmolout frames"


def test_human_log_records_failed_step_and_request(tmp_path: Path) -> None:
    trace = HumanReadableRunLog(
        tmp_path / "logs",
        command="reaxkit analyze",
        run_id="failed_example",
    )

    with pytest.raises(ValueError, match="bad input"):
        with trace:
            with trace.step("Read input"):
                raise ValueError("bad input")

    text = (tmp_path / "logs" / "human_readable.log").read_text(encoding="utf-8")
    assert text.count("status: FAILED") == 2
    assert "ValueError: bad input" in text
    machine_record = json.loads(
        (tmp_path / "logs" / "machine_readable.jsonl").read_text(encoding="utf-8")
    )
    assert machine_record["request"]["status"] == "failed"


def test_generator_runtime_adds_nested_steps_to_both_log_formats(tmp_path: Path) -> None:
    args = Namespace(
        run_id="run_generator_example",
        project_root=str(tmp_path),
        output="generated.txt",
    )
    trace = HumanReadableRunLog(
        tmp_path / "logs",
        command="reaxkit fake-generator --output generated.txt",
        run_id=args.run_id,
    )

    with trace:
        with trace.step("Execute fake-generator command"):
            output_path, layout = prepare_generator_output(
                args,
                command="fake-generator",
                output_value=args.output,
            )
            output_path.write_text("generated", encoding="utf-8")
            persist_generator_metadata(
                args,
                command="fake-generator",
                output_path=output_path,
                layout=layout,
            )

    human_text = (tmp_path / "logs" / "human_readable.log").read_text(encoding="utf-8")
    assert "- Execute fake-generator command" in human_text
    assert "- Prepare generator output" in human_text
    assert "- Save generator metadata" in human_text
    assert str(output_path.resolve()) in human_text

    machine_record = json.loads(
        (tmp_path / "logs" / "machine_readable.jsonl").read_text(encoding="utf-8")
    )
    nested_names = [step["name"] for step in machine_record["steps"][0]["steps"]]
    assert nested_names == ["Prepare generator output", "Save generator metadata"]
