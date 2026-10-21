import numpy as np
import pandas as pd
import pytest
from ase.io import read

from reaxkit.analysis.ferroelectrics.hbn_reference.al_neighborhoods import (
    build_al_neighborhoods,
    neighborhood_mapping,
    neighborhood_hull_volumes,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization import (
    HBNReferenceLocalPolarizationRequest,
    calculate_hbn_reference_local_polarization,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import (
    ELECTRON_CHARGE_SIGN,
    REFERENCE_STRUCTURE_PATH,
    prepare_hbn_reference,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.projected_polarity import (
    HBNReferenceProjectedPolarityRequest,
    HBNReferenceProjectedPolarityTask,
    calculate_hbn_reference_projected_polarity,
)
from reaxkit.domain.data_models import ChargeData, ElectrostaticsData, SimulationData, TrajectoryData
from reaxkit.engine.common.generators.structure_transformers import orthogonalize_hexagonal_cell


def trajectory(*, orthogonal=True, shifts=(0.1,), translated=False):
    atoms = read(REFERENCE_STRUCTURE_PATH)
    if orthogonal:
        atoms = orthogonalize_hexagonal_cell(atoms)
        fractional = atoms.get_scaled_positions()
        atoms.set_cell(np.diag(atoms.cell.lengths()), scale_atoms=False)
        atoms.set_scaled_positions(fractional)
    atoms = atoms.repeat((2, 1, 1))
    labels = np.asarray(atoms.get_chemical_symbols())
    positions = np.repeat(atoms.positions[None, :, :], len(shifts), axis=0)
    for frame, shift in enumerate(shifts):
        positions[frame, labels == "Al", 2] += shift
        if translated:
            positions[frame] += [0.17 * frame, -0.13 * frame, 0.21 * frame]
    atom_ids = np.arange(1, len(atoms) + 1)
    iterations = np.arange(len(shifts)) * 10
    simulation = SimulationData(
        atom_ids=atom_ids, elements=labels.tolist(), iterations=iterations,
        cell_lengths=np.repeat(atoms.cell.lengths()[None, :], len(shifts), axis=0),
        cell_angles=np.repeat(atoms.cell.angles()[None, :], len(shifts), axis=0),
    )
    return TrajectoryData(positions=positions, elements=labels.tolist(), atom_ids=atom_ids,
                          iterations=iterations, simulation=simulation)


def with_charges(data, charges):
    return ElectrostaticsData(trajectory=data, charges=ChargeData(
        charges=np.broadcast_to(charges, (len(data.positions), len(data.atom_ids))).copy(),
        iterations=data.iterations, simulation=data.simulation,
    ))


@pytest.mark.parametrize("method", ["partition", "neutral-center"])
@pytest.mark.parametrize("charge_source", ["formal", "reaxff"])
def test_local_models_conserve_or_neutralize_and_use_al_volumes(method, charge_source):
    data = trajectory(shifts=(0.1, -0.08))
    labels = np.asarray(data.elements)
    charges = np.where(labels == "Al", 2.0, -1.8) + np.linspace(-0.3, 0.3, len(labels))
    result = calculate_hbn_reference_local_polarization(with_charges(data, charges),
        HBNReferenceLocalPolarizationRequest(
            reference_path=REFERENCE_STRUCTURE_PATH, replication=(2, 1, 1),
            charge_source=charge_source, local_charge_method=method, volume_method="cell",
        ))
    count = np.count_nonzero(labels == "Al")
    assert len(result.table) == 2 * count
    edges = neighborhood_mapping(result.reference_result)
    np.testing.assert_allclose(edges.groupby("n_atom_id").partition_weight.sum(), 1.0)
    assert edges.groupby("al_atom_id").size().eq(3).all()
    for frame, table in result.table.groupby("frame_index"):
        global_row = result.reference_result.table.set_index("frame_index").loc[frame]
        np.testing.assert_allclose(table["local_volume (angstrom^3)"], global_row["volume (angstrom^3)"] / count)
        atomic = result.reference_result.displacements.query("frame_index == @frame").set_index("atom_id")
        for axis in "xyzc":
            assert table[f"raw_dipole_{axis} (e*angstrom)"].sum() == pytest.approx(global_row[f"dipole_{axis} (e*angstrom)"], abs=1e-10)
            if method == "partition":
                assert table[f"dipole_{axis} (e*angstrom)"].sum() == pytest.approx(global_row[f"dipole_{axis} (e*angstrom)"], abs=1e-10)
                assert table[f"P_{axis} (uC/cm^2)"].mean() == pytest.approx(global_row[f"P_{axis} (uC/cm^2)"], abs=1e-10)
            else:
                for row in table.to_dict("records"):
                    atom_id = row["representative_atom_id"]
                    neighbors = edges.loc[edges.al_atom_id == atom_id, "n_atom_id"]
                    relative = atomic.loc[atom_id, f"displacement_{axis} (angstrom)"] - atomic.loc[neighbors, f"displacement_{axis} (angstrom)"].mean()
                    expected = ELECTRON_CHARGE_SIGN * atomic.loc[atom_id, "charge (e)"] * relative
                    assert row[f"dipole_{axis} (e*angstrom)"] == pytest.approx(expected)
        if method == "neutral-center":
            assert table["effective_net_charge (e)"].eq(0).all()
        else:
            assert table["effective_net_charge (e)"].sum() == pytest.approx(atomic["charge (e)"].sum())


def test_neighborhood_ids_join_local_projected_and_shared_neighbor_mapping():
    data = trajectory(shifts=(0.1, 0.2))
    options = dict(reference_path=REFERENCE_STRUCTURE_PATH, replication=(2, 1, 1))
    local = calculate_hbn_reference_local_polarization(data, HBNReferenceLocalPolarizationRequest(**options))
    projected = calculate_hbn_reference_projected_polarity(data, HBNReferenceProjectedPolarityRequest(**options))
    edges = neighborhood_mapping(local.reference_result)
    id_column = "local_neighborhood_id"
    assert edges.groupby(id_column).size().eq(3).all()
    assert edges.groupby(id_column).al_atom_id.nunique().eq(1).all()
    assert edges.groupby("n_atom_id")[id_column].nunique().gt(1).any()
    expected = edges.groupby(id_column).al_atom_id.first().sort_index()
    for table in (local.table, projected.centers):
        assert not {"local_al_id", "local_layer_id", "local_cell_id", "local_group_id"} & set(table.columns)
        for _, frame in table.groupby("frame_index"):
            actual = frame.set_index(id_column).representative_atom_id.sort_index()
            np.testing.assert_array_equal(actual.index, expected.index)
            np.testing.assert_array_equal(actual, expected)


def test_neutral_al_ignores_n_charge_but_partition_does_not():
    data = trajectory()
    labels = np.asarray(data.elements)
    data.positions[0, labels == "N", 2] += np.linspace(-0.03, 0.04, np.count_nonzero(labels == "N"))
    charges = np.where(labels == "Al", 2.0, -2.0)
    changed = charges.copy()
    changed[labels == "N"] += np.linspace(-1, 1, np.count_nonzero(labels == "N"))
    request = HBNReferenceLocalPolarizationRequest(reference_path=REFERENCE_STRUCTURE_PATH,
        replication=(2, 1, 1), charge_source="reaxff", local_charge_method="neutral-center")
    first = calculate_hbn_reference_local_polarization(with_charges(data, charges), request)
    second = calculate_hbn_reference_local_polarization(with_charges(data, changed), request)
    np.testing.assert_allclose(first.table["dipole_c (e*angstrom)"], second.table["dipole_c (e*angstrom)"])
    assert not np.allclose(first.table["raw_dipole_c (e*angstrom)"], second.table["raw_dipole_c (e*angstrom)"])


@pytest.mark.parametrize("orthogonal", [True, False])
def test_reference_neighbors_are_coplanar_nearest_images_including_shared_periodic_sites(orthogonal):
    data = trajectory(orthogonal=orthogonal)
    request = HBNReferenceLocalPolarizationRequest(reference_path=REFERENCE_STRUCTURE_PATH, replication=(2, 1, 1))
    prepared = prepare_hbn_reference(data, request)
    neighbors = build_al_neighborhoods(prepared)
    assert prepared.orthogonalized == orthogonal
    np.testing.assert_allclose(neighbors.reference_vectors[:, :, 2], 0.0, atol=1e-8)
    distances = np.linalg.norm(neighbors.reference_vectors, axis=2)
    assert distances.max() / distances.min() < 1.001
    assert distances.max() < 2.0
    sharing = np.bincount(neighbors.nitrogen_indices.ravel(), weights=neighbors.nitrogen_weights.ravel(), minlength=len(data.atom_ids))
    np.testing.assert_allclose(sharing[np.asarray(data.elements) == "N"], 1.0)
    assert any(len(set(row)) < 3 for row in neighbors.nitrogen_indices)


def test_missing_reference_neighbors_fail_instead_of_using_distant_nitrogen():
    data = trajectory()
    request = HBNReferenceLocalPolarizationRequest(reference_path=REFERENCE_STRUCTURE_PATH,
        replication=(2, 1, 1), periodic=(False, False, False))
    with pytest.raises(ValueError, match="three unambiguous"):
        calculate_hbn_reference_local_polarization(data, request)


def test_geometric_polarity_is_charge_independent_and_tracks_positive_negative_zero():
    data = trajectory(shifts=(0.0, 0.1, -0.1, 1e-8), translated=True)
    request = HBNReferenceProjectedPolarityRequest(reference_path=REFERENCE_STRUCTURE_PATH,
        replication=(2, 1, 1), charge_source="reaxff", projection_bins=(1, 1))
    task = HBNReferenceProjectedPolarityTask()
    assert task.required_data_for(request) is TrajectoryData
    assert task.required_data_fields_for(request, {}) == ("trajectory",)
    geometry_only = task.run(data, request)
    count = np.count_nonzero(np.asarray(data.elements) == "Al")
    summary = geometry_only.whole_slab_summary
    np.testing.assert_allclose(summary.mean_polarity, [0, 1, -1, 0])
    assert summary.defined_group_count.eq(count).all()
    assert not any("charge" in name or "dipole" in name for name in geometry_only.centers.columns)
    for charges in (np.zeros(len(data.atom_ids)), np.arange(len(data.atom_ids)) - 4.5):
        weighted = calculate_hbn_reference_projected_polarity(with_charges(data, charges), request)
        pd.testing.assert_frame_equal(weighted.centers, geometry_only.centers)
        pd.testing.assert_frame_equal(weighted.projected_bins, geometry_only.projected_bins)
    local = calculate_hbn_reference_local_polarization(data, HBNReferenceLocalPolarizationRequest(
        reference_path=REFERENCE_STRUCTURE_PATH, replication=(2, 1, 1), charge_source="formal"))
    np.testing.assert_array_equal(local.table.geometric_polarity_c, geometry_only.centers.polarity)


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_geometric_tolerance_rejects_invalid_values(tolerance):
    request = HBNReferenceProjectedPolarityRequest(reference_path=REFERENCE_STRUCTURE_PATH,
        replication=(2, 1, 1), geometric_zero_tolerance=tolerance)
    with pytest.raises(ValueError, match="finite and non-negative"):
        calculate_hbn_reference_projected_polarity(trajectory(), request)


def test_tetrahedral_hull_volume_and_coplanar_degeneracy():
    from scipy.spatial import ConvexHull
    vectors = np.asarray([np.eye(3), [[1, 0, 0], [0, 1, 0], [-1, -1, 0]],
                          [[1.2, 0.1, -0.2], [-0.4, 1.4, -0.3], [-0.5, -0.8, -0.1]]])
    volumes = neighborhood_hull_volumes(vectors)
    assert volumes[0] == pytest.approx(1.0 / 6.0)
    assert volumes[1] == 0.0
    expected = ConvexHull(np.vstack((np.zeros(3), vectors[2]))).volume
    assert volumes[2] == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["local-hull", "explicit-hull"])
def test_neighborhood_hulls_are_independent_and_degenerate_polarization_is_undefined(mode):
    from scipy.spatial import ConvexHull
    from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import _frame_cell
    data = trajectory(shifts=(0.0, 0.1, 0.2))
    request = HBNReferenceLocalPolarizationRequest(reference_path=REFERENCE_STRUCTURE_PATH,
        replication=(2, 1, 1), charge_source="formal",
        volume_method="local-hull" if mode == "local-hull" else "bbox",
        local_volume_method="equal" if mode == "local-hull" else "hull")
    result = calculate_hbn_reference_local_polarization(data, request)
    planar = result.table.query("frame_index == 0")
    assert planar["local_volume (angstrom^3)"].eq(0).all()
    assert planar["local_volume_status"].eq("degenerate_hull").all()
    assert planar["P_c (uC/cm^2)"].isna().all()
    assert not planar.has_valid_local_volume.any()
    first = result.table.query("frame_index == 1")
    second = result.table.query("frame_index == 2")
    assert first.has_valid_local_volume.all()
    np.testing.assert_allclose(second["local_volume (angstrom^3)"], 2 * first["local_volume (angstrom^3)"], rtol=1e-10)
    assert first["local_volume (angstrom^3)"].sum() < first["frame_volume (angstrom^3)"].iloc[0] / 10
    assert result.summary.iloc[0]["invalid_local_volume_count"] == len(planar)
    neighbors = build_al_neighborhoods(result.reference_result.reference)
    cell = _frame_cell(data, request, 1)
    raw = data.positions[1]
    for group, center in enumerate(neighbors.al_indices):
        delta = raw[neighbors.nitrogen_indices[group]] - raw[center]
        fractional = delta @ np.linalg.inv(cell)
        expected_reference = neighbors.reference_vectors[group] @ np.linalg.inv(cell)
        fractional -= np.round(fractional - expected_reference)
        points = np.vstack((np.zeros(3), fractional @ cell))
        expected = ConvexHull(points).volume
        assert first.iloc[group]["local_volume (angstrom^3)"] == pytest.approx(expected)


@pytest.mark.parametrize("orthogonal", [True, False])
@pytest.mark.parametrize("method", ["partition", "neutral-center"])
@pytest.mark.parametrize("volume_method", ["bbox", "local-hull"])
def test_individual_periodic_image_wraps_preserve_local_values_and_projected_polarity(orthogonal, method, volume_method):
    from copy import deepcopy
    from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import _frame_cell
    data = trajectory(orthogonal=orthogonal, shifts=(0.1, -0.15))
    wrapped = deepcopy(data)
    request = HBNReferenceLocalPolarizationRequest(reference_path=REFERENCE_STRUCTURE_PATH,
        replication=(2, 1, 1), charge_source="formal", volume_method=volume_method,
        local_charge_method=method, periodic=(True, True, False))
    cell = _frame_cell(data, request, 1)
    images = np.random.default_rng(7).integers(-3, 4, size=(len(data.atom_ids), 3))
    images[:, 2] = 0
    wrapped.positions[1] += images @ cell
    original = calculate_hbn_reference_local_polarization(data, request)
    shifted = calculate_hbn_reference_local_polarization(wrapped, request)
    columns = ["local_volume (angstrom^3)"] + [f"{prefix}_{axis} ({unit})" for axis in "xyzc"
        for prefix, unit in [("dipole", "e*angstrom"), ("P", "uC/cm^2"), ("geometric_displacement", "angstrom")]]
    np.testing.assert_allclose(original.table[columns], shifted.table[columns], rtol=1e-10, atol=1e-9)
    projection_request = HBNReferenceProjectedPolarityRequest(reference_path=REFERENCE_STRUCTURE_PATH,
        replication=(2, 1, 1), periodic=(True, True, False), projection_bins=(2, 2))
    original_projection = calculate_hbn_reference_projected_polarity(data, projection_request)
    wrapped_projection = calculate_hbn_reference_projected_polarity(wrapped, projection_request)
    pd.testing.assert_frame_equal(original_projection.projected_bins, wrapped_projection.projected_bins)


def test_local_hull_rejects_conflicting_deformation_selection():
    request = HBNReferenceLocalPolarizationRequest(reference_path=REFERENCE_STRUCTURE_PATH,
        replication=(2, 1, 1), volume_method="local-hull", local_volume_method="deformation")
    with pytest.raises(ValueError, match="cannot be combined"):
        calculate_hbn_reference_local_polarization(trajectory(), request)


def test_hull_wrapping_in_a_changing_triclinic_cell():
    from copy import deepcopy
    from ase.cell import Cell
    from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import _frame_cell
    data = trajectory(orthogonal=False, shifts=(0.1, 0.15))
    request = HBNReferenceLocalPolarizationRequest(reference_path=REFERENCE_STRUCTURE_PATH,
        replication=(2, 1, 1), volume_method="local-hull", periodic=(True, True, False))
    original_cell = _frame_cell(data, request, 0)
    current_cell = original_cell @ np.diag([1.04, 0.97, 1.02])
    data.positions[1] = data.positions[1] @ np.linalg.inv(original_cell) @ current_cell
    data.simulation.cell_lengths[1] = Cell(current_cell).lengths()
    data.simulation.cell_angles[1] = Cell(current_cell).angles()
    wrapped = deepcopy(data)
    wrapped.positions[1, ::2] += current_cell[0] - 2 * current_cell[1]
    wrapped.positions[1, 1::2] -= 2 * current_cell[0] + current_cell[1]
    original = calculate_hbn_reference_local_polarization(data, request)
    shifted = calculate_hbn_reference_local_polarization(wrapped, request)
    np.testing.assert_allclose(original.table["local_volume (angstrom^3)"], shifted.table["local_volume (angstrom^3)"], rtol=1e-10)
    np.testing.assert_allclose(original.table["geometric_displacement_c (angstrom)"], shifted.table["geometric_displacement_c (angstrom)"], atol=1e-10)
