"""CSV-directory export helpers for trainset workflow results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reaxkit.cli.path import resolve_output_path


DEFAULT_TRAINSET_EXPORT_DIRECTORY = "trainset_data"


def parse_trainset_export_directory(value: str) -> str:
    """Validate that ``--export`` names a directory rather than a CSV file."""
    if Path(value).suffix.lower() == ".csv":
        raise argparse.ArgumentTypeError(
            "--export expects a directory, not a CSV filename; "
            "for example: --export trainset_data"
        )
    return value


def add_trainset_export_argument(parser: argparse.ArgumentParser) -> None:
    """Add the optional trainset export-directory argument to a CLI parser."""
    parser.add_argument(
        "--export",
        nargs="?",
        const=DEFAULT_TRAINSET_EXPORT_DIRECTORY,
        default=None,
        type=parse_trainset_export_directory,
        metavar="DIRECTORY",
        help=(
            "Write one CSV per selected trainset section to DIRECTORY. "
            f"If DIRECTORY is omitted, use '{DEFAULT_TRAINSET_EXPORT_DIRECTORY}'."
        ),
    )


def export_trainset_section_csvs(command: str, result, args) -> list[Path]:
    """Write one native-schema CSV per requested trainset section."""
    export_value = getattr(args, "export", None)
    if not export_value:
        return []

    export_dir = resolve_output_path(
        str(export_value),
        command,
        run_id=getattr(args, "run_id", None),
        project_root=getattr(args, "project_root", "."),
        analysis_id=getattr(args, "analysis_id", None),
    )
    if export_dir.exists() and not export_dir.is_dir():
        raise ValueError(f"Trainset export target is not a directory: {export_dir}")
    export_dir.mkdir(parents=True, exist_ok=True)

    section_tables = getattr(result, "section_tables", None)
    if not isinstance(section_tables, dict) or not section_tables:
        raise ValueError("Trainset result does not contain section tables to export.")

    request = getattr(result, "request", None)
    exporting_all = str(getattr(request, "section", "all")).strip().lower() == "all"
    written = 0
    for section_name, table in section_tables.items():
        if not isinstance(table, pd.DataFrame):
            raise TypeError(
                f"Trainset section {section_name!r} is not a pandas DataFrame."
            )
        if exporting_all and table.empty:
            continue
        filename = f"{str(section_name).strip().lower()}.csv"
        table.to_csv(export_dir / filename, index=False)
        written += 1

    if not written:
        raise ValueError("No trainset section rows are available to export.")

    return [export_dir]
