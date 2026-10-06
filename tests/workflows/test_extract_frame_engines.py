from io import StringIO
import importlib
import sys

import numpy as np
import pytest
from ase.io import read

from reaxkit.engine.ams.adapter import BOHR_TO_ANG
from reaxkit.engine.ams.rkf_handler import RKFHandler
from reaxkit.engine.common.generators.trajectory_frame import extract_frame


def dump(rows="2 2 0.5 0.5 0.5\n1 1 0 0 0\n", columns="id type xs ys zs", bounds="1 14 2\n2 24 1\n3 33 2\n"):
    return "ITEM: TIMESTEP\n100\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS xy xz yz pp pp pp\n" + bounds + "ITEM: ATOMS " + columns + "\n" + rows


def export(tmp_path, contents, **kwargs):
    source = tmp_path / "dump.lammpstrj"
    source.write_text(contents)
    xyz, geo = extract_frame(source, engine="lammps", xyz_file=tmp_path / "frame.xyz", geo_file=tmp_path / "frame.geo", **kwargs)
    atoms = read(StringIO(xyz.read_text()), format="xyz")
    lines = geo.read_text().splitlines()
    cellpar = [float(value) for value in next(line for line in lines if line.startswith("CRYSTX")).split()[1:]]
    return atoms, cellpar


def test_lammps_triclinic_scaled_preserves_geometry(tmp_path):
    atoms, cellpar = export(tmp_path, dump(), type_map=["Al", "N"])
    expected_cell = np.array([[10, 0, 0], [2, 20, 0], [1, 2, 30]])
    from ase.cell import Cell

    np.testing.assert_allclose(cellpar, Cell(expected_cell).cellpar(), atol=1e-5)
    assert atoms.get_chemical_symbols() == ["N", "Al"]
    assert atoms.get_distance(0, 1) == pytest.approx(np.linalg.norm([6.5, 11, 15]))
    np.testing.assert_allclose(atoms.positions[1], 0, atol=1e-12)


def test_lammps_cartesian_origin_and_element_column(tmp_path):
    atoms, _ = export(tmp_path, dump(rows="2 N 7.5 13 18\n1 Al 1 2 3\n", columns="id element x y z"))
    assert atoms.get_distance(0, 1) == pytest.approx(np.linalg.norm([6.5, 11, 15]))
    np.testing.assert_allclose(atoms.positions[1], 0, atol=1e-12)


@pytest.mark.parametrize("tail", ["ITEM: TIMESTEP\n", dump(rows="2 2 0.5 0.5 0.5\n"), dump(rows="2 2 0.5 0.5 0.5\n1 1 0 0 1e-")])
def test_lammps_incomplete_tail(tmp_path, tail):
    atoms, _ = export(tmp_path, dump() + tail, type_map=["Al", "N"])
    assert len(atoms) == 2
    with pytest.raises(ValueError, match="No complete frame"):
        export(tmp_path, dump() + tail, frame=1, type_map=["Al", "N"])


def test_lammps_numeric_types_require_mapping(tmp_path):
    with pytest.raises(ValueError, match="type-map"):
        export(tmp_path, dump())
    assert not (tmp_path / "frame.xyz").exists()
    with pytest.raises(ValueError, match="missing from"):
        export(tmp_path, dump(), type_map=["Al"])


def test_xyz_requires_cell(tmp_path):
    contents = "2\nTimestep: 100\nAl 0 0 0\nN 1 2 3\n"
    with pytest.raises(ValueError, match="--cell"):
        export(tmp_path, contents)
    atoms, cellpar = export(tmp_path, contents, cell=[10, 20, 30, 90, 90, 90])
    np.testing.assert_allclose(atoms.positions[1], [1, 2, 3], atol=1e-12)
    np.testing.assert_allclose(cellpar, [10, 20, 30, 90, 90, 90])


def test_extended_xyz_lattice(tmp_path):
    contents = '1\nLattice="10 0 0 2 20 0 1 2 30" Properties=species:S:1:pos:R:3\nAl 1 2 3\n'
    atoms, cellpar = export(tmp_path, contents)
    assert len(atoms) == 1
    assert cellpar[0] == 10
    assert cellpar[-1] < 90


def test_mapping_takes_precedence_over_mass_guessing(tmp_path):
    atoms, _ = export(tmp_path, dump(rows="2 2 1 0.5 0.5 0.5\n1 1 1 0 0 0\n", columns="id type mass xs ys zs"), type_map=["Al", "N"])
    assert atoms.get_chemical_symbols() == ["N", "Al"]


class FakeKF:
    def __init__(self, values):
        self.values = values

    def read(self, section, variable):
        return self.values[(section, variable)]

    def __getitem__(self, key):
        return self.read(*key.split("%"))


@pytest.mark.parametrize("frame,expected_length", [(0, 10), (1, 20), ("last", 20)])
def test_ams_selected_cell(tmp_path, monkeypatch, frame, expected_length):
    values = {("History", "nEntries"): 2, ("Molecule", "AtomSymbols"): "Al",
              ("History", "Coords(1)"): [0, 0, 0], ("History", "Coords(2)"): [1, 2, 3],
              ("History", "LatticeVectors(1)"): [10, 0, 0, 0, 10, 0, 0, 0, 10],
              ("History", "LatticeVectors(2)"): [20, 0, 0, 0, 20, 0, 0, 0, 20]}
    monkeypatch.setattr(RKFHandler, "kf", lambda self: FakeKF(values))
    _, geo = extract_frame(tmp_path / "ams.rkf", engine="ams", frame=frame,
                           xyz_file=tmp_path / "out.xyz", geo_file=tmp_path / "out.geo")
    crystx = next(line for line in geo.read_text().splitlines() if line.startswith("CRYSTX"))
    assert float(crystx.split()[1]) == pytest.approx(expected_length * BOHR_TO_ANG, abs=1e-5)


def test_ams_incomplete_dynamic_cell_does_not_use_initial_cell(tmp_path, monkeypatch):
    values = {("History", "nEntries"): 2, ("Molecule", "AtomSymbols"): "Al",
              ("History", "Coords(1)"): [0, 0, 0], ("History", "Coords(2)"): [1, 2, 3],
              ("History", "LatticeVectors(1)"): [10, 0, 0, 0, 10, 0, 0, 0, 10],
              ("Molecule", "LatticeVectors"): [99, 0, 0, 0, 99, 0, 0, 0, 99]}
    monkeypatch.setattr(RKFHandler, "kf", lambda self: FakeKF(values))
    xyz, _ = extract_frame(tmp_path / "ams.rkf", engine="ams",
                          xyz_file=tmp_path / "out.xyz", geo_file=tmp_path / "out.geo")
    assert "ams frame 0;" in xyz.read_text()
    with pytest.raises(ValueError, match="No complete frame"):
        extract_frame(tmp_path / "ams.rkf", engine="ams", frame=1,
                      xyz_file=tmp_path / "bad.xyz", geo_file=tmp_path / "bad.geo")


@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("frame", [0, "last"])
def test_ams_units_lattice_and_incomplete_tail(tmp_path, monkeypatch, standalone, frame):
    coordinate_name = "Coordinates 100" if standalone else "Coords(1)"
    lattice_name = "Unit cell axes 100" if standalone else "LatticeVectors(1)"
    factor = 1.0 if standalone else BOHR_TO_ANG
    values = {
        ("History", "nEntries"): 2,
        ("General", "Step numbers"): [100, 200],
        ("Molecule", "AtomSymbols"): "Al N",
        ("History", coordinate_name): [0, 0, 0, 1, 2, 3],
        ("History", lattice_name): [10, 0, 0, 2, 20, 0, 1, 2, 30],
    }
    monkeypatch.setattr(RKFHandler, "kf", lambda self: FakeKF(values))
    source = tmp_path / "ams.rkf"
    source.touch()
    xyz, geo = extract_frame(source, engine="ams", frame=frame, xyz_file=tmp_path / "ams.xyz", geo_file=tmp_path / "ams.geo")
    atoms = read(xyz, format="xyz")
    assert atoms.get_distance(0, 1) == pytest.approx(np.sqrt(14) * factor)
    crystx = next(line for line in geo.read_text().splitlines() if line.startswith("CRYSTX"))
    assert float(crystx.split()[1]) == pytest.approx(10 * factor, abs=1e-5)
    with pytest.raises(ValueError, match="No complete frame"):
        extract_frame(source, engine="ams", frame=1, xyz_file=tmp_path / "bad.xyz", geo_file=tmp_path / "bad.geo")
    assert not (tmp_path / "bad.xyz").exists()


@pytest.mark.parametrize("engine", ["reaxff", "ams", "lammps"])
def test_cli_long_alias_all_engines(tmp_path, monkeypatch, engine):
    monkeypatch.chdir(tmp_path)
    if engine == "lammps":
        (tmp_path / "dump.xyz").write_text(dump(rows="2 N 7 8 9\n1 Al 1 2 3\n", columns="id element x y z"))
    elif engine == "reaxff":
        (tmp_path / "xmolout").write_text("1\nslab 0 -1 10 10 10 90 90 90\nAl 1 2 3\n")
    else:
        (tmp_path / "ams.rkf").touch()
        values = {("History", "nEntries"): 1, ("Molecule", "AtomSymbols"): "Al",
                  ("History", "Coords(1)"): [1, 2, 3],
                  ("Molecule", "LatticeVectors"): [10, 0, 0, 0, 10, 0, 0, 0, 10]}
        monkeypatch.setattr(RKFHandler, "kf", lambda self: FakeKF(values))
    monkeypatch.setattr(sys, "argv", ["reaxkit", "extract_trajectory_frame", "--engine", engine, "--output", "restart"])
    assert importlib.import_module("reaxkit.cli.main").main() == 0
    assert (tmp_path / "restart.xyz").exists()
    assert (tmp_path / "restart.geo").exists()
