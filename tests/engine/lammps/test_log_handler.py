from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reaxkit.engine.lammps.lammps_log_handler import LAMMPSLogHandler


def _example_log_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "examples_to_test" / "log.lammps"


def _artifacts_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "tests" / "artifacts" / "lammps"


def _write_log_artifact(data: dict[str, np.ndarray], out_base: Path) -> Path:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    table = {key: np.asarray(values).reshape(-1) for key, values in data.items()}
    df = pd.DataFrame(table)
    try:
        xlsx_path = out_base.with_suffix(".xlsx")
        df.to_excel(xlsx_path, index=False)
        return xlsx_path
    except Exception:
        txt_path = out_base.with_suffix(".txt")
        txt_path.write_text(df.to_string(index=False), encoding="utf-8")
        return txt_path


def test_lammps_log_handler_reads_example_file():
    log_path = _example_log_path()
    if not log_path.exists():
        pytest.skip(f"LAMMPS test input file not found: {log_path}")

    try:
        from lammps.formats import LogFile  # noqa: F401
    except Exception:
        pytest.skip("LAMMPS backend unavailable: lammps.formats.LogFile is not installed.")

    data = LAMMPSLogHandler(log_path).read()
    assert isinstance(data, dict)
    assert "runs" in data
    assert isinstance(data["runs"], list)
    assert data["runs"], "Expected at least one run in log.lammps."

    thermo = data.get("thermo")
    assert isinstance(thermo, dict)
    assert "Density" in thermo, "Expected selected thermo table to include Density."
    assert "Step" in thermo
    assert isinstance(thermo["Step"], np.ndarray)
    assert thermo["Step"].size > 0

    artifact_path = _write_log_artifact(thermo, _artifacts_dir() / "log_thermo")
    assert artifact_path.exists()
