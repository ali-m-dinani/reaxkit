from dataclasses import replace

import numpy as np
import pytest
from ase.io import read

from tests.analysis.test_hbn_al_neighborhoods import trajectory
from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import REFERENCE_STRUCTURE_PATH
from reaxkit.analysis.ferroelectrics.hbn_reference.projected_polarity import (
    HBNReferenceProjectedPolarityRequest,
    HBNReferenceProjectedPolarityTask,
    calculate_hbn_reference_projected_polarity,
)
from reaxkit.core.runtime.result_store import ResultStore


@pytest.mark.parametrize("component", ["x", "y", "z", "c"])
def test_geometric_extxyz_matches_centers_without_electrical_properties(tmp_path, component):
    data = trajectory(shifts=(0.0, 0.1))
    output = tmp_path / "geometric.extxyz"
    request = HBNReferenceProjectedPolarityRequest(
        reference_path=REFERENCE_STRUCTURE_PATH, replication=(2, 1, 1), component=component, write_extxyz=True,
        _output_path=str(output), extxyz_precision=12, workers=2,
    )
    result = calculate_hbn_reference_projected_polarity(data, request)
    frames = read(output, index=":")
    assert len(frames) == 2
    for index, atoms in enumerate(frames):
        assert set(atoms.arrays) == {
            "numbers", "positions", "atom_number", "local_neighborhood_id", "is_local_center",
            "geometric_displacement", "geometric_displacement_c", "geometric_polarity",
            "geometric_polarity_c", "polarity",
        }
        np.testing.assert_allclose(atoms.positions, data.positions[index], atol=1e-10)
        np.testing.assert_array_equal(atoms.arrays["atom_number"], data.atom_ids)
        centers = atoms.arrays["is_local_center"].astype(bool)
        np.testing.assert_array_equal(centers, np.asarray(data.elements) == "Al")
        assert (atoms.arrays["local_neighborhood_id"][~centers] == -1).all()
        assert np.isnan(atoms.arrays["polarity"][~centers]).all()
        rows = result.centers.loc[result.centers.frame_index == index].set_index("representative_atom_id")
        rows = rows.loc[atoms.arrays["atom_number"][centers]]
        np.testing.assert_array_equal(atoms.arrays["local_neighborhood_id"][centers], rows.local_neighborhood_id)
        np.testing.assert_allclose(atoms.arrays["polarity"][centers], rows.polarity)
        displacement = atoms.arrays["geometric_displacement_c"] if component == "c" else atoms.arrays["geometric_displacement"][:, "xyz".index(component)]
        np.testing.assert_allclose(displacement[centers], rows[f"geometric_displacement_{component} (angstrom)"], atol=1e-10)
        assert atoms.info["frame"] == index
        assert atoms.info["iter"] == data.iterations[index]
        assert atoms.info["polarity_component"] == component
    without_centers = replace(request, include_centers=False, _output_path=str(tmp_path / "no_centers.extxyz"))
    result = calculate_hbn_reference_projected_polarity(data, without_centers)
    assert result.centers.empty
    assert output.read_bytes() == (tmp_path / "no_centers.extxyz").read_bytes()


def test_projected_extxyz_resume_and_completed_reexport(tmp_path):
    frames = [trajectory(shifts=(shift,)) for shift in (0.0, 0.1, -0.1)]
    for source, data in zip((0, 5, 10), frames):
        data.source_frame_indices = [source]
        data.iterations = np.asarray([source * 20])
    output = tmp_path / "expected.extxyz"
    request = HBNReferenceProjectedPolarityRequest(
        reference_path=REFERENCE_STRUCTURE_PATH, replication=(2, 1, 1), include_centers=False, frames=(0, 5, 10),
        write_extxyz=True, _output_path=str(output), workers=1, chunk_size=1,
    )
    task = HBNReferenceProjectedPolarityTask()
    task.run_stream(iter(frames), request)
    expected = output.read_bytes()
    actual_path = tmp_path / "actual.extxyz"
    request = replace(request, _output_path=str(actual_path))

    def broken():
        yield frames[0]
        yield frames[1]
        raise OSError("interrupted")

    with pytest.raises(OSError, match="interrupted"):
        with ResultStore(tmp_path / "checkpoint", {}, batch_frames=1) as store:
            request._result_store = store
            task.run_stream(broken(), request)
    with ResultStore(tmp_path / "checkpoint", {}, resume=True) as store:
        store.transition("running")
        request._result_store = store
        result = task.run_stream(iter(frames), request)
        np.testing.assert_array_equal(result.frame_indices, [0, 5, 10])
    assert actual_path.read_bytes() == expected
    assert [atoms.info["frame"] for atoms in read(actual_path, index=":")] == [0, 5, 10]
    actual_path.unlink()
    with ResultStore(tmp_path / "checkpoint", {}, resume=True) as store:
        request._result_store = store
        task.run_stream(iter(()), request)
    assert actual_path.read_bytes() == expected


def test_projected_extxyz_requires_path_and_positive_precision(tmp_path):
    data = trajectory()
    request = HBNReferenceProjectedPolarityRequest(reference_path=REFERENCE_STRUCTURE_PATH, replication=(2, 1, 1), write_extxyz=True)
    with pytest.raises(ValueError, match="output path"):
        calculate_hbn_reference_projected_polarity(data, request)
    request = replace(request, _output_path=str(tmp_path / "invalid.extxyz"), extxyz_precision=0)
    with pytest.raises(ValueError, match="precision"):
        calculate_hbn_reference_projected_polarity(data, request)
