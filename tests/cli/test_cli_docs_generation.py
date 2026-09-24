"""Generated public help must include the live parser metadata without drift."""

from pathlib import Path
import subprocess
import sys


def test_generated_workflow_docs_are_current():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "docs/scripts/generate_workflow_cli_docs.py", "--check"],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr
