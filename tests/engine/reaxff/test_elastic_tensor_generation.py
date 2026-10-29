from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.io import read, write

from reaxkit.engine.reaxff.generators.elastic_tensor import (
    strain_matrix, validate_tensor, tensor_warnings,
)
from reaxkit.engine.reaxff.generators.trainset_elastic_energy import (
    BulkEnergySpec, CellSpec, ElasticEnergySpec, ENERGY_CONVERSION_FACTOR,
    _generate_elastic_data, _generate_trainset_energy,
)
from reaxkit.engine.reaxff.generators.trainset_elastic_geometry import (
    StrainedGeometrySpec, _deformation_matrix, _generate_strained_geometries,
    _write_strained_geometries,
)
from reaxkit.engine.reaxff.generators.trainset_yaml import (
    _generate_trainset_from_yaml, _write_trainset_settings_yaml,
    _concat_geo_strained, _merge_successful_elastic_trainsets,
)


def stiffness():
    generator = np.random.default_rng(672)
    basis = generator.normal(size=(6, 6))
    return basis.T @ basis * 10 + np.eye(6) * 80


def legacy_constants():
    return dict(c11=180, c22=190, c33=220, c12=40, c13=50, c23=60, c44=70, c55=80, c66=90)


def orthorhombic_tensor():
    tensor = np.zeros((6, 6))
    for label, value in legacy_constants().items():
        first, second = int(label[1]) - 1, int(label[2]) - 1
        tensor[first, second] = tensor[second, first] = value
    return tensor


def geometry_spec(tmp_path, angles=(90, 90, 90), maximum=0.01):
    cell = CellSpec(3, 4, 5, *angles)
    atoms = Atoms("MgO", cell=cellpar_to_cell(list(cell.as_dict().values())),
                  scaled_positions=[[0.12, 0.23, 0.34], [0.61, 0.57, 0.48]], pbc=True)
    xyz = tmp_path / "base.xyz"
    write(xyz, atoms, format="xyz")
    return StrainedGeometrySpec(xyz, None, cell, cell, maximum, 0.005,
                                (1.06)**(1/3)-1, 0.004, tensor_mode=True)


def test_full_tensor_reproduces_all_nine_legacy_energy_curves():
    cell = CellSpec(3, 4, 5, 90, 90, 90)
    legacy = ElasticEnergySpec(legacy_constants(), 3, cell)
    expected = _generate_elastic_data(legacy)
    actual = _generate_elastic_data(replace(legacy, tensor_gpa=orthorhombic_tensor().tolist()))
    for mode in expected:
        np.testing.assert_allclose(actual[mode][0], expected[mode][0], atol=1e-14)
        assert actual[mode][1] == expected[mode][1]


def test_twenty_one_modes_recover_every_tensor_entry():
    tensor = stiffness()
    cell = CellSpec(3, 4, 5, 72, 83, 107)
    volume = abs(np.linalg.det(cellpar_to_cell(list(cell.as_dict().values()))))
    result = _generate_elastic_data(ElasticEnergySpec({}, 1, cell, tensor_gpa=tensor.tolist()))
    design, curvatures = [], []
    for mode, (rows, _) in result.items():
        derivative = (_deformation_matrix(mode, 1e-6) - _deformation_matrix(mode, -1e-6)) / 2e-6
        strain = (derivative + derivative.T) / 2
        voigt = np.array([strain[0, 0], strain[1, 1], strain[2, 2], 2*strain[1, 2], 2*strain[0, 2], 2*strain[0, 1]])
        design.append([voigt[first]*voigt[second]*(1 if first == second else 2)
                       for first in range(6) for second in range(first, 6)])
        delta, energy = rows[-1]
        curvatures.append(2 * energy * ENERGY_CONVERSION_FACTOR / volume / delta**2)
    assert np.linalg.matrix_rank(design) == 21
    recovered = np.linalg.solve(design, curvatures)
    np.testing.assert_allclose(recovered, [tensor[first, second] for first in range(6) for second in range(first, 6)], rtol=1e-8)


@pytest.mark.parametrize("angles", [(90, 90, 90), (90, 90, 120), (90, 104, 90), (72, 83, 107)])
@pytest.mark.parametrize("tensor_mode", [False, True])
def test_cell_and_atomic_deformation_in_cartesian_axes(tmp_path, angles, tensor_mode):
    spec = replace(geometry_spec(tmp_path, angles), tensor_mode=tensor_mode)
    result = _generate_strained_geometries(spec)
    for mode, records in result.records_by_mode.items():
        reference = next(record.atoms for record in records if record.title.endswith("_0"))
        delta = spec.dstrain_bulk_linear if mode == "bulk" else spec.dstrain_elastic
        step = next(record.atoms for record in records if record.title.endswith("e0001"))
        expected = _deformation_matrix(mode, delta)
        np.testing.assert_allclose(step.cell.array, reference.cell.array @ expected.T, atol=1e-12)
        np.testing.assert_allclose(step.positions, reference.positions @ expected.T, atol=1e-12)
        np.testing.assert_allclose(step.get_scaled_positions(), reference.get_scaled_positions(), atol=1e-12)


@pytest.mark.parametrize("tensor_mode", [False, True])
def test_generated_energy_labels_have_geometries_on_rounded_grid(tmp_path, tensor_mode):
    spec = replace(geometry_spec(tmp_path, maximum=0.012), tensor_mode=tensor_mode)
    geometries = _generate_strained_geometries(spec)
    energy = _generate_trainset_energy(BulkEnergySpec(100, 1.5, 6, spec.bulk_cell),
                                      ElasticEnergySpec(legacy_constants(), 1.2, spec.elastic_cell,
                                                        tensor_gpa=stiffness().tolist() if tensor_mode else None))
    labels = {record.title for records in geometries.records_by_mode.values() for record in records}
    for line in energy.trainset_text.splitlines():
        if " /1 " in line:
            assert line.split()[2] in labels
            assert line.split()[5] in labels
    assert len(geometries.records_by_mode["bulk"]) == len(energy.bulk_table)


@pytest.mark.parametrize("tensor_mode", [False, True])
def test_reaxff_export_preserves_periodic_geometry(tmp_path, tensor_mode):
    spec = replace(geometry_spec(tmp_path, (72, 83, 107)), tensor_mode=tensor_mode)
    result = _generate_strained_geometries(spec)
    _write_strained_geometries(result, out_dir=tmp_path / "export")
    for mode in (("c44", "c15", "c46") if tensor_mode else ("c44", "c55", "c66")):
        record = result.records_by_mode[mode][-1]
        exported = read(tmp_path / "export/xyz_strained" / record.xyz_filename, format="xyz")
        lengths = record.box_lengths
        cosine_alpha, cosine_beta, cosine_gamma = np.cos(np.radians(record.box_angles))
        sine_alpha = np.sqrt(1-cosine_alpha**2)
        axis_y = lengths[0] * (cosine_gamma-cosine_beta*cosine_alpha)/sine_alpha
        axis_z = lengths[0] * cosine_beta
        cell = np.array([[np.sqrt(lengths[0]**2-axis_y**2-axis_z**2), axis_y, axis_z],
                         [0, lengths[1]*sine_alpha, lengths[1]*cosine_alpha], [0, 0, lengths[2]]])
        exported.set_cell(cell)
        exported.pbc = True
        np.testing.assert_allclose(exported.get_scaled_positions(), record.atoms.get_scaled_positions(), atol=1e-9)
        np.testing.assert_allclose(exported.get_all_distances(mic=True), record.atoms.get_all_distances(mic=True), atol=1e-8)


@pytest.mark.parametrize("tensor", [np.eye(3), np.full((6, 6), np.nan), np.triu(np.ones((6, 6)))])
def test_invalid_tensors_fail(tensor):
    with pytest.raises(ValueError):
        validate_tensor(tensor)


def test_negative_couplings_are_not_a_stability_test():
    stable = np.eye(6) * 100
    stable[0, 3] = stable[3, 0] = -5
    assert not tensor_warnings(stable)
    stable[3, 3] = -10
    assert "not positive definite" in tensor_warnings(stable)[0]


@pytest.mark.parametrize("cell", [CellSpec(-1, 4, 5, 90, 90, 90), CellSpec(3, 4, 5, 10, 10, 170), CellSpec(3, 4, 5, 90, 0, 90)])
def test_invalid_cell_metric_is_rejected(cell):
    with pytest.raises(ValueError, match="Cell"):
        _generate_elastic_data(ElasticEnergySpec({}, 1, cell, tensor_gpa=stiffness().tolist()))


def test_explicit_tensor_frame_required(tmp_path):
    import yaml

    settings = tmp_path / "settings.yaml"
    _write_trainset_settings_yaml(out_path=str(settings), geo_enable=False, tensor_gpa=stiffness().tolist())
    config = yaml.safe_load(settings.read_text())
    config["elastic"]["tensor_frame"] = "ieee"
    settings.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="tensor_frame"):
        _generate_trainset_from_yaml(str(settings), str(tmp_path / "output"))


def test_raw_tensor_requires_original_frame():
    pytest.importorskip("mp_api")
    from pymatgen.core import Structure
    from reaxkit.engine.reaxff.generators.trainset_mp import _tensor_in_structure_frame

    structure = Structure(cellpar_to_cell([3, 4, 5, 72, 83, 107]), ["Mg", "O"], [[0, 0, 0], [0.12, 0.23, 0.34]])
    source = SimpleNamespace(raw=stiffness().tolist())
    with pytest.raises(ValueError, match="original structure"):
        _tensor_in_structure_frame(source, structure)
    np.testing.assert_allclose(_tensor_in_structure_frame(source, structure, structure), stiffness(), atol=1e-9)


def test_cli_includes_nonorthogonal_cells_by_default():
    import argparse
    from reaxkit.workflows.file_tools.trainset_workflow import build_parser

    parser = build_parser(argparse.ArgumentParser(), command="gen_elastic_trainset")
    assert not parser.parse_args(["--api-key", "test-key"]).skip_not_orthogonal
    assert parser.parse_args(["--api-key", "test-key", "--skip-not-orthogonal"]).skip_not_orthogonal


def test_yaml_batch_names_and_merged_linkage(tmp_path):
    for material in ("mp-1", "mp-2"):
        output = tmp_path / "successful" / material
        output.mkdir(parents=True)
        spec = geometry_spec(output, (90, 104, 90))
        settings = output / "settings.yaml"
        _write_trainset_settings_yaml(out_path=str(settings), mp_id=material,
                                     elastic_cell=spec.elastic_cell.as_dict(), bulk_cell=spec.bulk_cell.as_dict(),
                                     elastic_xyz=str(spec.elastic_xyz), tensor_gpa=stiffness().tolist())
        _generate_trainset_from_yaml(str(settings), str(output))
        _concat_geo_strained(output)
    _merge_successful_elastic_trainsets(tmp_path / "successful", tmp_path)
    descriptors = [line.split()[1] for line in (tmp_path / "geo").read_text().splitlines() if line.startswith("DESCRP")]
    assert len(set(descriptors)) == len(descriptors)
    for line in (tmp_path / "trainset_elastic.in").read_text().splitlines():
        if " /1 " in line:
            assert line.split()[2] in descriptors
            assert line.split()[5] in descriptors
    assert "c15_e1_mp_1" in descriptors
    assert "c15_e1_mp_2" in descriptors


def test_nine_constant_yaml_matches_tensor_geometries_and_vasp2reax(tmp_path):
    from reaxkit.engine.reaxff.generators.geo_generator import vasp2reax

    spec = geometry_spec(tmp_path, maximum=0.012)
    for tensor_mode in (False, True):
        destination = tmp_path / ("tensor" if tensor_mode else "nine_constant")
        settings = destination / "settings.yaml"
        _write_trainset_settings_yaml(
            out_path=str(settings), elastic_cell=spec.elastic_cell.as_dict(),
            bulk_cell=spec.bulk_cell.as_dict(), elastic_xyz=str(spec.elastic_xyz),
            elastic_max_strain_percent=1.2, cij_gpa=legacy_constants(),
            tensor_gpa=orthorhombic_tensor().tolist() if tensor_mode else None,
        )
        _generate_trainset_from_yaml(str(settings), str(destination))
    old_output = tmp_path / "nine_constant"
    tensor_output = tmp_path / "tensor"
    for table in (old_output / "volume_energy_data").glob("*.dat"):
        assert table.read_bytes() == (tensor_output / "volume_energy_data" / table.name).read_bytes()
    geometries = list((old_output / "structures/geo_strained").glob("*"))
    descriptors = {path.stem for path in geometries}
    for line in (old_output / "trainset_elastic.in").read_text().splitlines():
        if " /1 " in line:
            assert line.split()[2] in descriptors
            assert line.split()[5] in descriptors
    for geometry in geometries:
        assert geometry.read_bytes() == (tensor_output / "structures/geo_strained" / geometry.name).read_bytes()
    strained = _generate_strained_geometries(replace(spec, tensor_mode=False))
    for mode in ("c44", "c55", "c66"):
        record = strained.records_by_mode[mode][-1]
        poscar = tmp_path / f"POSCAR_{mode}"
        write(poscar, record.atoms, format="vasp")
        reference_geo, _ = vasp2reax(poscar, tmp_path / f"reference_{mode}.geo", format="vasp")
        actual_geo = old_output / "structures/geo_strained" / record.geo_filename
        reference_lines = [line for line in reference_geo.read_text().splitlines() if line.startswith(("CRYSTX", "HETATM"))]
        actual_lines = [line for line in actual_geo.read_text().splitlines() if line.startswith(("CRYSTX", "HETATM"))]
        assert actual_lines == reference_lines


def test_incomplete_nonorthogonal_yaml_requires_full_tensor(tmp_path):
    settings = tmp_path / "settings.yaml"
    cell = CellSpec(3, 4, 5, 90, 104, 90).as_dict()
    _write_trainset_settings_yaml(out_path=str(settings), elastic_cell=cell, geo_enable=False)
    with pytest.raises(ValueError, match="full tensor"):
        _generate_trainset_from_yaml(str(settings), str(tmp_path / "output"))
    assert _generate_trainset_from_yaml(str(settings), str(tmp_path / "skip"), skip_no_orthogonal=True) is False


@pytest.mark.parametrize("angles", [(90, 90, 90), (90, 90, 120), (90, 104, 90), (72, 83, 107)])
def test_ieee_tensor_rotation_round_trip(angles):
    pytest.importorskip("mp_api")
    from pymatgen.core import Structure
    from pymatgen.core.tensors import Tensor
    from pymatgen.analysis.elasticity.elastic import ElasticTensor
    from reaxkit.engine.reaxff.generators.trainset_mp import _tensor_in_structure_frame

    structure = Structure(cellpar_to_cell([3, 4, 5, *angles]), ["Mg", "O"], [[0, 0, 0], [0.12, 0.23, 0.34]])
    tensor = ElasticTensor.from_voigt(stiffness())
    rotation = Tensor.get_ieee_rotation(structure)
    ieee = tensor.rotate(rotation).voigt
    actual = _tensor_in_structure_frame(SimpleNamespace(ieee_format=ieee), structure)
    np.testing.assert_allclose(actual, tensor.voigt, atol=1e-10)
    strain = np.array([0.003, -0.002, 0.001, 0.004, -0.005, 0.002])
    expected = np.einsum("ijkl,ij,kl", tensor, strain_matrix(strain), strain_matrix(strain))
    assert strain @ actual @ strain == pytest.approx(expected)
