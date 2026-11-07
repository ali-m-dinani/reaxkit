"""Read native text reports and rounded training progress."""

import re

from ..base import NUMBER, Reader, jax_reports
from ...models import progress


def convert(root, options=None):
    reader = Reader(root, options)
    path = reader.root / reader.options.get("progress_file", "stdout.log")
    if path.is_file():
        trial = None
        step = None
        segment = 0
        previous_trial = None
        for line_number, line in reader.lines(path):
            match = re.search(r"Trial-(\d+) is starting", line)
            if match:
                trial = int(match[1])
                if previous_trial is not None and trial <= previous_trial:
                    segment += 1
                previous_trial, step = trial, None
            match = re.search(r"^Iteration:\s*(\d+)", line)
            if match:
                step = int(match[1])
            match = re.search(r"^True total loss:\s*(" + NUMBER + ")", line, re.I)
            if match:
                reader.result.progress.append(progress(
                    reader.relative(path), line_number, match[1], native_step=step,
                    step_kind="iteration", candidate_id=f"segment:{segment}:trial:{trial}:iteration:{step}",
                    trial=trial, segment=segment, precision="native_stdout_2_decimals",
                ))
    jax_reports(reader, list(reader.root.glob("report_*.txt")) + list(reader.root.glob("valid_report_*.txt")))
    for path in sorted(reader.root.glob("*")):
        if path.name.startswith(("checkpoint_FF", "new_FF")):
            reader.artifact(path, "parameter_only")
    for path in sorted(reader.root.glob("*.json")):
        if path.name in ("machine_report.json", "evaluation.json"):
            report = reader.json(path)
            if isinstance(report, dict) and "loss" in report:
                reader.result.progress.append(progress(reader.relative(path), "evaluation", report["loss"]))
                reader.result.warnings.append(f"{path.name}: unlabeled items retained in native report; no cross-file positional join")
    return reader.result
