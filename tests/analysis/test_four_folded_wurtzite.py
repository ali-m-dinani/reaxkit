from __future__ import annotations

import numpy as np

from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    WurtziteNeighborRequest,
    extract_wurtzite_neighbors,
)
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.polarity import (
    WurtzitePolarityRequest,
    calculate_wurtzite_polarity,
)
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.plotting import (
    plot_site_resolved_polarity,
)
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.trajectory import (
    PolarityExtendedXYZRequest,
    PolarityExtendedXYZTask,
)
from reaxkit.domain.data_models import SimulationData, TrajectoryData
from reaxkit.workflows.ferroelectrics.four_folded_wurtzite.artifacts import (
    write_polarity_tables,
)


def _trajectory() -> TrajectoryData:
    elements = ["Al", "N", "N", "N", "N", "H"]
    frame_zero = np.asarray([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.8],
        [1.7, 0.0, -0.5],
        [-0.85, 1.472, -0.5],
        [-0.85, -1.472, -0.5],
        [0.0, 0.0, 1.0],
    ])
    frame_one = frame_zero.copy()
    frame_one[2:5, 2] = -0.4
    positions = np.stack([frame_zero, frame_one])
    simulation = SimulationData(
        atom_ids=list(range(1, 7)),
        elements=elements,
        iterations=np.asarray([0, 10]),
        cell_lengths=np.asarray([[20.0, 20.0, 20.0], [20.0, 20.0, 20.0]]),
        cell_angles=np.asarray([[90.0, 90.0, 90.0], [90.0, 90.0, 90.0]]),
    )
    return TrajectoryData(
        positions=positions,
        elements=elements,
        atom_ids=list(range(1, 7)),
        simulation=simulation,
        iterations=np.asarray([0, 10]),
    )


def test_neighbor_module_returns_normalized_coordinates_and_formal_charges() -> None:
    request = WurtziteNeighborRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0, "H": 1.0},
    )
    result = extract_wurtzite_neighbors(_trajectory(), request)

    assert len(result.centers) == 2
    assert len(result.neighbors) == 8
    assert result.centers["site_element"].unique().tolist() == ["Al"]
    assert result.centers["site_charge (e)"].tolist() == [3.0, 3.0]
    assert result.neighbors["neighbor_charge (e)"].eq(-3.0).all()
    assert result.neighbors.groupby("frame_index")["neighbor_rank"].apply(list).tolist() == [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ]
    assert result.neighbors.groupby("frame_index")["neighbor_role"].apply(list).apply(
        lambda roles: roles.count("apical") == 1 and roles.count("basal") == 3
    ).all()
    assert result.centers["has_proton_within_cutoff"].all()


def test_neighbor_csv_tables_are_compact_nonduplicate_and_time_is_after_iter() -> None:
    request = WurtziteNeighborRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
    )
    result = extract_wurtzite_neighbors(_trajectory(), request)
    result.centers["time"] = [0.0, 0.01]
    result.neighbors["time"] = [0.0] * 4 + [0.01] * 4

    assert set(result.csv_tables) == {"centers", "neighbors"}
    centers = result.csv_tables["centers"]
    neighbors = result.csv_tables["neighbors"]
    assert centers.columns[centers.columns.get_loc("iter") + 1] == "time"
    assert neighbors.columns[neighbors.columns.get_loc("iter") + 1] == "time"
    assert not {
                   "proton_cutoff (angstrom)",
                   "charge_source",
                   "neighbor_cutoff (angstrom)",
                   "c_axis_x",
                   "c_axis_y",
                   "c_axis_z",
                   "periodic_a",
                   "periodic_b",
                   "periodic_c",
               } & set(centers.columns)
    assert "bond_c (angstrom)" in neighbors.columns
    assert "neighbor_role" in neighbors.columns


def test_bond_c_is_not_generally_the_cartesian_bond_z_component() -> None:
    request = WurtziteNeighborRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
        c_axis=(1.0, 0.0, 0.0),
    )
    neighbors = extract_wurtzite_neighbors(_trajectory(), request).neighbors

    assert np.allclose(neighbors["bond_c (angstrom)"], neighbors["bond_x (angstrom)"])
    assert not np.allclose(neighbors["bond_c (angstrom)"], neighbors["bond_z (angstrom)"])


def test_polarity_module_marks_apical_and_basal_and_tracks_frame_zero_change() -> None:
    request = WurtzitePolarityRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
    )
    neighbor_result = extract_wurtzite_neighbors(_trajectory(), request)
    result = calculate_wurtzite_polarity(neighbor_result, request)

    assert result.table["polarity"].tolist() == [1, 1]
    assert result.apical_neighbors.groupby("frame_index").size().tolist() == [1, 1]
    assert result.basal_neighbors.groupby("frame_index").size().tolist() == [3, 3]
    assert result.table.filter(regex=r"neighbor_[1-4]_role").iloc[0].tolist().count("apical") == 1
    assert np.isclose(
        result.table["mean_basal_bond_c_change_from_frame_0 (angstrom)"].iloc[1],
        0.1,
    )
    assert result.summary["n_polarity_up"].tolist() == [1, 1]
    assert set(result.proton_summary["proton_group"]) == {"with_proton", "without_proton"}


def test_polarity_artifacts_include_a_complete_variable_guide(tmp_path) -> None:
    request = WurtzitePolarityRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
    )
    result = calculate_wurtzite_polarity(
        extract_wurtzite_neighbors(_trajectory(), request), request
    )
    for frame in (
            result.centers,
            result.neighbor_geometry,
            result.table,
            result.summary,
            result.proton_summary,
    ):
        frame["time"] = np.arange(len(frame), dtype=float)

    paths = write_polarity_tables(result, tmp_path)
    guide = paths["variables"].read_text(encoding="utf-8")

    assert paths["centers"] == tmp_path / "other_helpful_data" / "centers.csv"
    assert paths["neighbors"] == tmp_path / "other_helpful_data" / "neighbors.csv"
    assert paths["centers"].is_file()
    assert paths["neighbors"].is_file()
    assert not (tmp_path / "centers.csv").exists()
    assert not (tmp_path / "centers_and_neighbors.csv").exists()
    assert not (tmp_path / "apical_neighbors.csv").exists()
    assert not (tmp_path / "basal_neighbors.csv").exists()
    assert "delta (angstrom)\n  Geometric polar displacement" in guide
    assert "eta_c (e*angstrom)\n  Charge-weighted local polar order" in guide
    assert "neighbor_role\n  Geometric classification" in guide

    emitted_csvs = [paths[key] for key in ("centers", "neighbors", "polarity", "summary", "proton")]
    for path in emitted_csvs:
        columns = path.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert columns[columns.index("iter") + 1] == "time"
        for column in columns:
            assert f"\n{column}\n  " in guide

    neighbor_columns = paths["neighbors"].read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "neighbor_role" in neighbor_columns
    polarity_columns = paths["polarity"].read_text(encoding="utf-8").splitlines()[0].split(",")
    assert not {
                   "site_x (angstrom)",
                   "site_y (angstrom)",
                   "site_z (angstrom)",
                   "site_charge (e)",
                   "proton_cutoff (angstrom)",
                   "neighbor_cutoff (angstrom)",
                   "charge_source",
               } & set(polarity_columns)
    assert not any(column.startswith("neighbor_") for column in polarity_columns)


def test_polarity_extxyz_preserves_species_and_adds_columns(tmp_path) -> None:
    output = tmp_path / "polarity.extxyz"
    request = PolarityExtendedXYZRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0, "H": 1.0},
        _output_path=str(output),
    )
    result = PolarityExtendedXYZTask().run(_trajectory(), request)
    lines = output.read_text(encoding="utf-8").splitlines()

    assert result.frame_indices.tolist() == [0, 1]
    assert lines[1].startswith(
        "Properties=species:S:1:pos:R:3:atom_number:I:1:charge:R:1:polarity:I:1"
    )
    assert "frame=0 iter=0" in lines[1]
    assert [line.split()[0] for line in lines[2:8]] == ["Al", "N", "N", "N", "N", "H"]
    assert lines[2].split()[6] == "1"


def test_polarity_plotting_supports_3d_and_2d_slices(tmp_path) -> None:
    request = WurtzitePolarityRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
    )
    result = calculate_wurtzite_polarity(
        extract_wurtzite_neighbors(_trajectory(), request), request
    )

    plots_3d = plot_site_resolved_polarity(result.table, tmp_path / "3d", value="eta")
    plots_2d = plot_site_resolved_polarity(
        result.table,
        tmp_path / "2d",
        plane="xz",
        value="basal-difference",
        slice_range=(-1.0, 1.0),
        color_scale="frame",
    )
    assert len(plots_3d) == len(plots_2d) == 2
    assert all(path.is_file() for path in [*plots_3d, *plots_2d])
