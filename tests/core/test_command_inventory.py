"""Keep the checked-in command/output inventory synchronized with production."""

import json
from pathlib import Path
import subprocess
import sys


def test_generated_command_inventory_matches_checked_in_contract(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "inventory.json"
    # A fresh interpreter prevents test-only registry entries leaking into it.
    subprocess.run([sys.executable, "-m", "reaxkit.core.runtime.command_inventory", "--output", str(output)],
                   cwd=root, check=True, capture_output=True, text=True)
    generated = json.loads(output.read_text())
    assert generated == json.loads((root / "docs/command_execution_inventory.json").read_text())
    for task in generated["analysis_tasks"]:
        assert task["result_class"] and task["capabilities"]["shape"]
        assert task["tables"] or task["public_result_fields"]
    for command in generated["cli_commands"]:
        assert command["execution"]["shape"] and command["output_resolution"]
        assert all(site.get("status") != "registered_target_missing" for site in command["workflow_output_sites"])


def test_workflow_tables_use_shared_publication():
    root = Path(__file__).resolve().parents[2] / "src/reaxkit/workflows"
    import ast
    direct = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"to_csv", "to_parquet", "DictWriter", "writer"}:
                direct.append(f"{path.name}:{node.lineno}")
    assert not direct, f"Workflow tables must use ArtifactWriter: {direct}"
