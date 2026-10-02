"""Geometry regressions using the reported mp-768893 CIF and vasp2reax reference."""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.io import read
from pymatgen.core import Structure

from reaxkit.engine.reaxff.generators.geo_generator import orient_structure_for_reaxff
from reaxkit.engine.reaxff.generators.trainset_heatfo import _write_structure_triplet
from reaxkit.engine.reaxff.generators.trainset_mp import _convert_structure_setting


FIXTURES = Path(__file__).parent / "fixtures"


def read_reaxff_geo(path):
    lines = path.read_text().splitlines()
    parameters = np.array(next(line.split()[1:] for line in lines if line.startswith("CRYSTX")), dtype=float)
    reverse_parameters = parameters[[2, 1, 0, 5, 4, 3]]
    cell = cellpar_to_cell(reverse_parameters)[::-1, ::-1]
    records = [line.split() for line in lines if line.startswith("HETATM")]
    return Atoms(
        [record[2] for record in records],
        positions=np.array([record[3:6] for record in records], dtype=float),
        cell=cell,
        pbc=True,
    )


@pytest.mark.parametrize("angles", [(90, 90, 90), (105, 90, 90), (90, 123, 90), (90, 90, 120), (73, 82, 107)])
def test_reaxff_orientation_preserves_periodic_geometry(angles):
    atoms = Atoms(
        "MgTeO2", cell=cellpar_to_cell([9, 7, 12, *angles]), pbc=True,
        scaled_positions=[[0.02, 0.1, 0.3], [0.98, 0.9, 0.7], [0.4, 0.6, 0.8], [1.2, -0.1, 0.5]],
    )
    atoms.rotate(37, [1, 2, 3], rotate_cell=True)
    original = atoms.copy()
    oriented = orient_structure_for_reaxff(atoms)

    np.testing.assert_allclose(oriented.get_all_distances(mic=True), atoms.get_all_distances(mic=True), atol=1e-12)
    np.testing.assert_allclose(oriented.cell.cellpar(), atoms.cell.cellpar(), atol=1e-12)
    np.testing.assert_allclose(oriented.get_scaled_positions(wrap=False), atoms.get_scaled_positions(wrap=False), atol=1e-12)
    np.testing.assert_allclose(oriented.cell.array[2, :2], 0, atol=1e-12)
    assert oriented.cell.array[1, 0] == 0
    np.testing.assert_array_equal(atoms.positions, original.positions)
    np.testing.assert_array_equal(atoms.cell.array, original.cell.array)


def test_reaxff_orientation_rejects_missing_cell():
    with pytest.raises(ValueError, match="non-degenerate"):
        orient_structure_for_reaxff(Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.2]]))


def test_orientation_matches_supplied_vasp2reax_structure():
    atoms = read(FIXTURES / "mg_te_o_768893.cif")
    atoms.set_cell(atoms.cell.array[[0, 2, 1]], scale_atoms=False)
    oriented = orient_structure_for_reaxff(atoms)
    reference = read_reaxff_geo(FIXTURES / "mg_te_o_768893_vasp2reax.geo")

    assert oriented.get_chemical_symbols() == reference.get_chemical_symbols()
    np.testing.assert_allclose(oriented.positions, reference.positions, atol=1e-5, rtol=0)
    np.testing.assert_allclose(oriented.get_all_distances(mic=True), reference.get_all_distances(mic=True), atol=2e-5, rtol=0)


@pytest.mark.parametrize("conversion", ["to-conventional", "to-primitive"])
def test_setting_conversion_preserves_pymatgen_axis_order(conversion):
    structure = Structure.from_file(FIXTURES / "mg_te_o_768893.cif")
    expected = structure.to_conventional()
    if conversion == "to-primitive":
        expected = expected.get_primitive_structure()
    converted = _convert_structure_setting(structure, conversion)
    np.testing.assert_allclose(converted.lattice.matrix, expected.lattice.matrix)
    np.testing.assert_allclose(converted.frac_coords, expected.frac_coords)


@pytest.mark.parametrize("conversion", ["to-conventional", "to-primitive"])
def test_heatfo_triplet_preserves_cif_geometry(tmp_path, conversion):
    directories = {name: tmp_path / name for name in ("cif", "xyz", "geo")}
    for directory in directories.values():
        directory.mkdir()
    result = _write_structure_triplet(
        doc_obj={
            "material_id": "mp-768893", "formula_pretty": "Mg2(TeO3)3",
            "symmetry": {"crystal_system": "monoclinic"},
            "structure": Structure.from_file(FIXTURES / "mg_te_o_768893.cif"),
        },
        cif_dir=directories["cif"], xyz_dir=directories["xyz"], geo_dir=directories["geo"],
        crystallographic_setting_conversion=conversion,
    )
    source = read(directories["cif"] / f"{result.identifier}.cif")
    exported = read_reaxff_geo(result.geo_path)
    xyz = read(directories["xyz"] / f"{result.identifier}.xyz")
    assert len(exported) == len(source) == result.total_atoms
    assert exported.get_chemical_symbols() == source.get_chemical_symbols()
    np.testing.assert_allclose(exported.get_all_distances(mic=True), source.get_all_distances(mic=True), atol=3e-5, rtol=0)
    np.testing.assert_allclose(exported.positions, xyz.positions, atol=6e-6, rtol=0)
    if conversion == "to-conventional":
        np.testing.assert_allclose(exported.cell.angles(), [90, 123.001, 90], atol=1e-4)
