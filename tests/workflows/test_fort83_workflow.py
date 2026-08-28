import argparse
from pathlib import Path

from reaxkit.engine.reaxff.io.fort83_handler import DEFAULT_OPTIMIZED_FFIELD_NAME
from reaxkit.workflows.file_tools import fort83_workflow


def test_parser_uses_fort83_optimized_ffield_as_default_output() -> None:
    parser = fort83_workflow.build_parser(
        argparse.ArgumentParser(),
        command="get-optimized-ffield",
    )

    args = parser.parse_args([])

    assert args.fort83 == "fort.83"
    assert args.output == DEFAULT_OPTIMIZED_FFIELD_NAME


def test_run_main_writes_the_last_force_field(tmp_path: Path) -> None:
    fort83 = tmp_path / "fort.83"
    fort83.write_text(
        "Error force field: 10.0\nold field\n"
        "Error force field: 1.0\noptimized field\n",
        encoding="utf-8",
    )
    project_root = tmp_path / "workspace"
    args = argparse.Namespace(
        fort83=str(fort83),
        output=DEFAULT_OPTIMIZED_FFIELD_NAME,
        copy_to_dot=False,
        run_id="fort83-test",
        project_root=str(project_root),
        analysis_id=None,
    )

    result = fort83_workflow.run_main("get-optimized-ffield", args)

    assert result == 0
    output = project_root / "inputs" / "fort83-test" / DEFAULT_OPTIMIZED_FFIELD_NAME
    assert output.read_text(encoding="utf-8") == "optimized field\n"
