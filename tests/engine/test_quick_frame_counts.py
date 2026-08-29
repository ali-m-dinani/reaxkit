from __future__ import annotations

from pathlib import Path

from reaxkit.engine.ams.adapter import AMSAdapter
from reaxkit.engine.lammps.adapter import LAMMPSAdapter
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_reaxff_quick_count_uses_xmolout_frame_start_lines(tmp_path: Path) -> None:
    xmolout = _write(
        tmp_path / "xmolout",
        "2\nsim 0 -1 10 10 10 90 90 90\nC 0 0 0\nH 1 0 0\n"
        "2\nsim 1 -1 10 10 10 90 90 90\nC 0 0 0\nH 1 0 0\n"
        "2\nsim 2 -1 10 10 10 90 90 90\nC 0 0 0\nH 1 0 0\n",
    )
    control = _write(tmp_path / "control", "100 nmdit\n10 iout2\n")

    assert ReaxFFAdapter.quick_n_frames({"xmolout": str(xmolout), "control": str(control)}) == 3


def test_lammps_quick_count_supports_xyz_like_dump(tmp_path: Path) -> None:
    dump = _write(
        tmp_path / "dump.xyz",
        "2\nLAMMPS timestep: 0\nAl 0 0 0\nN 1 0 0\n"
        "2\nLAMMPS timestep: 10\nAl 0.1 0 0\nN 1.1 0 0\n",
    )

    assert LAMMPSAdapter().quick_n_frames({"dump": str(dump)}) == 2


def test_lammps_quick_count_supports_native_item_dump(tmp_path: Path) -> None:
    dump = _write(
        tmp_path / "dump.lammpstrj",
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n"
        "0 1\n0 1\n0 1\nITEM: ATOMS id element x y z\n1 C 0 0 0\n"
        "ITEM: TIMESTEP\n10\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n"
        "0 1\n0 1\n0 1\nITEM: ATOMS id element x y z\n1 C 0.1 0 0\n",
    )

    assert LAMMPSAdapter().quick_n_frames({"dump": str(dump)}) == 2


def test_ams_quick_count_reads_only_history_metadata(monkeypatch) -> None:
    class MetadataOnlyKF:
        def read(self, section: str, variable: str):
            if (section, variable) == ("MDHistory", "nEntries"):
                return 12
            raise AssertionError(f"Unexpected RKF read: {section}%{variable}")

    adapter = AMSAdapter()
    monkeypatch.setattr(adapter, "load_kf", lambda _args: MetadataOnlyKF())

    assert adapter.quick_n_frames({"rkf": "trajectory.rkf"}) == 12
