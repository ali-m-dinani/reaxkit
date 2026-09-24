"""Compare pre-change and current dispatch parsing without running commands.

Run from the repository with ``python tools/compare_cli_namespaces.py [revision]``.
The baseline dispatcher comes from git; command implementations and shared
helpers are held constant to isolate the help/dispatcher change.
"""

import argparse
from importlib import import_module
import json
import subprocess
import sys
from types import ModuleType


CASES = [
    ["msd", "--atom-ids", "1"],
    ["mean-square-displacement", "--atom-ids", "1", "--xmolout", "renamed.xyz"],
    ["get-dipole", "--engine", "reaxff", "--run-dir", "custom-run", "--fort7", "renamed.bo",
     "--xmolout", "renamed.xyz", "--frames", "0:20:2"],
    ["get-hbn-reference-projected-polarity", "--replication", "2", "3", "4",
     "--engine", "ams", "--input", "custom-run/reaxout.kf"],
    ["get-hbn-reference-projected-polarity", "--replication", "2", "3", "4"],
    ["get-hbn-reference-projected-polarity", "--replication", "2", "3", "4",
     "--execution", "threads", "--workers", "2", "--chunk-size", "3",
     "--project-root", "compat-workspace", "--run-id", "run", "--analysis-id", "analysis",
     "--no-input-cache", "--output-profile", "minimal"],
    ["fort7", "--workers", "2", "get", "--file", "renamed.bo", "--yaxis", "charge"],
    ["fort7", "get", "--file", "renamed.bo", "--yaxis", "charge", "--execution", "serial"],
    ["gen_template_control", "--output", "custom.control", "--parameter", "nmdit", "--value", "1000"],
    ["--timing", "--no-progress", "--no-stream", "--log-in-terminal", "get-dipole"],
]


class Captured(Exception):
    pass


def parse_without_execution(module, arguments):
    parser_cls = module._ReaxKitArgumentParser
    original_parse = parser_cls.parse_args
    original_argv = sys.argv
    result = {}

    def capture(parser, args=None, namespace=None):
        parsed = argparse.ArgumentParser.parse_args(parser, args, namespace)
        result.update({key: value for key, value in vars(parsed).items() if not callable(value)})
        raise Captured

    parser_cls.parse_args = capture
    sys.argv = ["reaxkit", *arguments]
    try:
        module.main(announce=False)
    except Captured:
        return result
    finally:
        parser_cls.parse_args = original_parse
        sys.argv = original_argv
    raise AssertionError("Dispatcher did not parse arguments")


def main():
    revision = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    source = subprocess.check_output(["git", "show", f"{revision}:src/reaxkit/cli/main.py"], text=True)
    baseline = ModuleType("reaxkit.cli._baseline")
    exec(compile(source, "baseline_cli.py", "exec"), baseline.__dict__)
    current = import_module("reaxkit.cli.main")
    for case in CASES:
        before = parse_without_execution(baseline, case)
        after = parse_without_execution(current, case)
        assert before == after, (case, before, after)
    print(json.dumps({"baseline": subprocess.check_output(["git", "rev-parse", revision], text=True).strip(),
                      "matching_namespaces": len(CASES), "cases": CASES}, indent=2))


if __name__ == "__main__":
    main()
