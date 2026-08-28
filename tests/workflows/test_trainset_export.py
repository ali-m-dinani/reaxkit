from __future__ import annotations

import argparse
from types import SimpleNamespace

import pandas as pd
import pytest

from reaxkit.workflows.file_tools.trainset_export import (
    DEFAULT_TRAINSET_EXPORT_DIRECTORY,
    add_trainset_export_argument,
    export_trainset_section_csvs,
    parse_trainset_export_directory,
)


def test_export_trainset_section_csvs_writes_one_native_csv_per_section(tmp_path) -> None:
    export_dir = tmp_path / "trainset_data"
    args = SimpleNamespace(
        export=str(export_dir),
        run_id=None,
        project_root=str(tmp_path),
        analysis_id=None,
    )
    result = SimpleNamespace(
        request=SimpleNamespace(section="all"),
        section_tables={
            "CHARGE": pd.DataFrame({"atom": [1], "charge": [-0.5]}),
            "HEATFO": pd.DataFrame(),
            "ENERGY": pd.DataFrame({"id1": ["bulk"], "lit": [-15.4]}),
        }
    )

    output_dirs = export_trainset_section_csvs("get_trainset_data", result, args)

    assert output_dirs == [export_dir]
    assert sorted(path.name for path in export_dir.glob("*.csv")) == [
        "charge.csv",
        "energy.csv",
    ]
    assert list(pd.read_csv(export_dir / "charge.csv").columns) == ["atom", "charge"]
    assert list(pd.read_csv(export_dir / "energy.csv").columns) == ["id1", "lit"]


def test_parse_trainset_export_directory_rejects_csv_filename() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="expects a directory"):
        parse_trainset_export_directory("trainset_data.csv")


def test_parse_trainset_export_directory_accepts_directory() -> None:
    assert parse_trainset_export_directory("trainset_data") == "trainset_data"


def test_trainset_export_argument_accepts_optional_directory_value() -> None:
    parser = argparse.ArgumentParser()
    add_trainset_export_argument(parser)

    assert parser.parse_args([]).export is None
    assert parser.parse_args(["--export"]).export == DEFAULT_TRAINSET_EXPORT_DIRECTORY
    assert parser.parse_args(["--export", "my_tables"]).export == "my_tables"
