import json
from pathlib import Path
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import cellpar_to_cell
from ase.io import read, write

import reaxkit.cli as cli
from reaxkit.engine.reaxff.generators.geo_generator import vasp2reax


def read_geo(path):
    lines = path.read_text().splitlines()
    parameters = np.array(next(line.split()[1:] for line in lines if line.startswith("CRYSTX")), dtype=float)
    cell = cellpar_to_cell(parameters[[2, 1, 0, 5, 4, 3]])[::-1, ::-1]
    records = [line.split() for line in lines if line.startswith("HETATM")]
    return Atoms([record[2] for record in records], positions=np.array([record[3:6] for record in records], dtype=float), cell=cell, pbc=True)


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("selective", [True, False])
def test_vasp_modes_and_species_preserve_geometry(tmp_path, direct, selective):
    atoms = Atoms("MgTeOHNC", scaled_positions=np.arange(18).reshape(6, 3) / 19,
                  cell=cellpar_to_cell([9, 7, 12, 73, 82, 107]), pbc=True)
    if selective:
        atoms.set_constraint(FixAtoms(indices=[0]))
    source = tmp_path / "POSCAR"
    write(source, atoms, format="vasp", direct=direct)
    geo, xyz = vasp2reax(source, tmp_path / "converted.geo")
    exported = read_geo(geo)
    assert exported.get_chemical_symbols() == atoms.get_chemical_symbols()
    np.testing.assert_allclose(exported.get_all_distances(mic=True), atoms.get_all_distances(mic=True), atol=3e-5)
    np.testing.assert_allclose(exported.positions, read(xyz).positions, atol=6e-6)
    assert xyz.name == "converted_coordinates.xyz"


def test_cif_conversion(tmp_path):
    source = Path(__file__).parent / "fixtures" / "mg_te_o_768893.cif"
    geo, xyz = vasp2reax(source, tmp_path / "reaxff.geo")
    np.testing.assert_allclose(read_geo(geo).get_all_distances(mic=True), read(source).get_all_distances(mic=True), atol=3e-5)
    assert xyz.is_file()


@pytest.mark.parametrize("scale", ["2.0", "-192.0"])
def test_vasp_scale_handling(tmp_path, scale):
    source = tmp_path / "structure.dat"
    source.write_text(f"MgO\n{scale}\n2 0 0\n0 3 0\n0 0 4\nMg O\n1 1\nDirect\n0 0 0\n0.25 0.25 0.25\n")
    geo, _ = vasp2reax(source, tmp_path / "scaled.geo", format="vasp")
    np.testing.assert_allclose(read_geo(geo).cell.lengths(), [4, 6, 8], atol=1e-5)


def test_output_cannot_overwrite_input(tmp_path):
    source = tmp_path / "POSCAR"
    source.write_text("original")
    with pytest.raises(ValueError, match="different files"):
        vasp2reax(source, source)
    assert source.read_text() == "original"


def test_cli_help_describes_outputs(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["reaxkit", "vasp2reax", "-h"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for text in ("fort.15", "fort.51", "reaxff.geo", "reaxff_coordinates.xyz", "Examples:", "[NOTE]"):
        assert text in help_text


def test_cli_writes_and_copies_both_outputs(tmp_path, monkeypatch):
    source = Path(__file__).parent / "fixtures" / "mg_te_o_768893.cif"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "reaxkit", "vasp2reax", "--file", str(source.resolve()),
        "--project-root", str(tmp_path / "workspace"), "--run-id", "conversion", "--copy-to-dot",
    ])
    assert cli.main() == 0
    geo = tmp_path / "reaxff.geo"
    xyz = tmp_path / "reaxff_coordinates.xyz"
    assert geo.is_file() and xyz.is_file()
    settings = next((tmp_path / "workspace").rglob("settings.json"))
    payload = json.loads(settings.read_text())
    assert payload["command"] == "vasp2reax"
    assert Path(payload["args"]["coordinates_output"]).read_bytes() == xyz.read_bytes()
    assert Path(payload["output"]["path"]).read_bytes() == geo.read_bytes()
