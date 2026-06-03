from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reaxkit.engine.lammps.dump_handler import LAMMPSDumpHandler


def _example_dump_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "examples_to_test" / "dump.xyz"


def _artifacts_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "tests" / "artifacts" / "lammps"


def _write_frame_artifact(frame_df: pd.DataFrame, out_base: Path) -> Path:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    try:
        xlsx_path = out_base.with_suffix(".xlsx")
        frame_df.to_excel(xlsx_path, index=False)
        return xlsx_path
    except Exception:
        txt_path = out_base.with_suffix(".txt")
        txt_path.write_text(frame_df.to_string(index=False), encoding="utf-8")
        return txt_path


def _write_metadata_artifact(metadata: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}: {v}" for k, v in sorted(metadata.items(), key=lambda kv: str(kv[0]))]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def test_lammps_dump_handler_reads_example_dump_and_writes_artifacts():
    dump_path = _example_dump_path()
    if not dump_path.exists():
        pytest.skip(f"LAMMPS test input file not found: {dump_path}")

    handler = LAMMPSDumpHandler(dump_path)
    sim_df = handler.dataframe()
    metadata = handler.metadata()

    assert not sim_df.empty
    assert "iter" in sim_df.columns
    assert int(metadata.get("n_frames", 0)) > 0

    frame0 = handler.frame(0)
    frame0_df = pd.DataFrame(
        {
            "atom_type": frame0["atom_types"],
            "x": frame0["coords"][:, 0],
            "y": frame0["coords"][:, 1],
            "z": frame0["coords"][:, 2],
        }
    )
    frame_artifact = _write_frame_artifact(frame0_df, _artifacts_dir() / "dump_frame_0000")
    metadata_artifact = _write_metadata_artifact(metadata, _artifacts_dir() / "dump_metadata.txt")

    assert frame_artifact.exists()
    assert metadata_artifact.exists()
