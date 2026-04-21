"""Parity checks against bundled TRACT reference outputs (same trajectory/frame)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from ase.io import read as ase_read
from reaxkit.analysis.active_sites import ActiveSiteStructuralRequest, ActiveSiteStructuralTask
from reaxkit.analysis.active_sites.structural import (
    _compute_has_hetero_bond,
    _compute_psi6,
    _compute_soap,
    _compute_strain_roughness,
    _detect_grains,
)
from reaxkit.domain.data_models import ConnectivityData, ConnectivityTrajectoryData, SimulationData, TrajectoryData

ROOT = Path(__file__).resolve().parents[2]
TRACT_SCRIPT = ROOT / "full_sim_examples" / "Anirban codes" / "TRACT_package" / "scripts" / "structural_analyzer.py"
TRACT_FRAME = ROOT / "full_sim_examples" / "Anirban codes" / "TRACT_package" / "example_run" / "30_1073_frame0.extxyz"
TRACT_STRUCTURAL_CSV = ROOT / "full_sim_examples" / "Anirban codes" / "TRACT_package" / "example_run" / "outputs" / "30_1073_frame0_structural.csv"
TRACT_SOAP_NPY = ROOT / "full_sim_examples" / "Anirban codes" / "TRACT_package" / "example_run" / "outputs" / "30_1073_frame0_soap.npy"


def _load_tract_module():
    spec = importlib.util.spec_from_file_location("tract_structural_analyzer", str(TRACT_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load TRACT structural_analyzer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_reference_graph():
    if not TRACT_SCRIPT.exists() or not TRACT_FRAME.exists():
        pytest.skip("TRACT reference script/frame is not available in workspace.")
    tract = _load_tract_module()
    structure = tract.read_structure(str(TRACT_FRAME), frame_index=0)
    graph, neighbors = tract.build_bond_graph(structure, scale=1.20, extra=0.0)
    xyz = np.asarray(structure.coords, dtype=float)
    elements = [str(e) for e in structure.elements]
    cell = np.asarray(structure.cell, dtype=float) if structure.cell is not None else None
    return graph, neighbors, xyz, elements, cell


def test_structural_descriptors_match_tract_reference_frame():
    if not TRACT_STRUCTURAL_CSV.exists():
        pytest.skip("TRACT reference structural CSV is not available in workspace.")

    graph, neighbors, xyz, elements, cell = _load_reference_graph()
    ref = pd.read_csv(TRACT_STRUCTURAL_CSV).sort_values("atom_id").reset_index(drop=True)

    bond_strain, angle_strain, local_roughness = _compute_strain_roughness(graph, neighbors, xyz, cell=cell)
    psi6_re, psi6_im, psi6_mag, psi6_ang = _compute_psi6(xyz, neighbors, cell=cell)
    grain_id = _detect_grains(neighbors, psi6_mag, psi6_ang)
    has_hetero_bond = _compute_has_hetero_bond(elements, neighbors, carbon_element="C")

    calc = pd.DataFrame(
        {
            "atom_id": np.arange(1, len(elements) + 1, dtype=int),
            "bond_strain_calc": bond_strain,
            "angle_strain_calc": angle_strain,
            "local_roughness_calc": local_roughness,
            "psi6_re_calc": psi6_re,
            "psi6_im_calc": psi6_im,
            "psi6_mag_calc": psi6_mag,
            "psi6_ang_calc": psi6_ang,
            "grain_id_calc": grain_id,
            "has_hetero_bond_calc": has_hetero_bond,
        }
    )
    merged = ref.merge(calc, on="atom_id", how="inner")
    assert len(merged) == len(ref)

    atol = 1.0e-10
    for ref_col, calc_col in [
        ("bond_strain", "bond_strain_calc"),
        ("angle_strain", "angle_strain_calc"),
        ("local_roughness", "local_roughness_calc"),
        ("psi6_re", "psi6_re_calc"),
        ("psi6_im", "psi6_im_calc"),
        ("psi6_mag", "psi6_mag_calc"),
        ("psi6_ang", "psi6_ang_calc"),
    ]:
        diff = np.abs(pd.to_numeric(merged[ref_col], errors="coerce") - pd.to_numeric(merged[calc_col], errors="coerce"))
        assert float(np.nanmax(diff.to_numpy(dtype=float))) <= atol

    assert np.array_equal(merged["grain_id"].to_numpy(dtype=int), merged["grain_id_calc"].to_numpy(dtype=int))
    assert np.array_equal(
        merged["has_hetero_bond"].to_numpy(dtype=bool),
        merged["has_hetero_bond_calc"].to_numpy(dtype=bool),
    )


def test_soap_matches_tract_reference_frame():
    pytest.importorskip("dscribe")
    if not TRACT_SOAP_NPY.exists() or not TRACT_STRUCTURAL_CSV.exists():
        pytest.skip("TRACT reference SOAP outputs are not available in workspace.")

    graph, neighbors, xyz, elements, cell = _load_reference_graph()
    _ = graph
    _ = neighbors
    ref = pd.read_csv(TRACT_STRUCTURAL_CSV).sort_values("atom_id").reset_index(drop=True)

    c_idx = np.where(np.asarray(elements, dtype=object) == "C")[0]
    descriptors, pca_scores, soap_score = _compute_soap(
        xyz=xyz,
        elements=elements,
        c_idx=c_idx,
        cell=cell,
        r_cut=5.0,
        n_max=9,
        l_max=9,
        soap_ref=None,
        zeta=2,
    )
    assert soap_score is None

    ref_descriptors = np.load(TRACT_SOAP_NPY)
    assert descriptors.shape == ref_descriptors.shape
    assert np.allclose(descriptors, ref_descriptors, atol=1.0e-10)

    ref_c = ref[ref["element"] == "C"].reset_index(drop=True)
    ref_pc = ref_c[["soap_pc1", "soap_pc2", "soap_pc3"]].to_numpy(dtype=float)
    assert pca_scores.shape == ref_pc.shape
    # PCA component signs can flip; compare by absolute correlation.
    for i in range(3):
        corr = np.corrcoef(pca_scores[:, i], ref_pc[:, i])[0, 1]
        assert float(abs(corr)) > 0.999999


def test_distance_bond_mode_task_matches_reference_statistics_without_bond_orders():
    if not TRACT_STRUCTURAL_CSV.exists() or not TRACT_FRAME.exists():
        pytest.skip("TRACT reference frame/csv is not available in workspace.")

    atoms = ase_read(str(TRACT_FRAME), format="extxyz")
    xyz = np.asarray(atoms.get_positions(), dtype=float)
    elements = [str(e) for e in atoms.get_chemical_symbols()]
    n_atoms = len(elements)
    atom_ids = list(range(1, n_atoms + 1))

    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
    angles = np.asarray(atoms.get_cell().angles(), dtype=float)
    sim = SimulationData(
        atom_ids=atom_ids,
        iterations=np.asarray([0], dtype=int),
        cell_lengths=np.asarray([lengths], dtype=float),
        cell_angles=np.asarray([angles], dtype=float),
    )
    traj = TrajectoryData(
        positions=np.asarray([xyz], dtype=float),
        elements=elements,
        atom_ids=atom_ids,
        iterations=np.asarray([0], dtype=int),
        simulation=sim,
    )
    # Intentionally omit bond_orders to verify distance mode does not depend on ConnectivityData.bond_orders.
    conn = ConnectivityData(
        atom_ids=atom_ids,
        elements=elements,
        iterations=np.asarray([0], dtype=int),
        simulation=sim,
    )
    data = ConnectivityTrajectoryData(connectivity=conn, trajectory=traj)

    task = ActiveSiteStructuralTask()
    req = ActiveSiteStructuralRequest(
        frame=0,
        bond_mode="distance",
        bond_scale=1.2,
        include_noncarbon=True,
    )
    out = task.run(data, req).table.sort_values("atom_id").reset_index(drop=True)
    ref = pd.read_csv(TRACT_STRUCTURAL_CSV).sort_values("atom_id").reset_index(drop=True)

    assert len(out) == len(ref) == n_atoms
    assert np.array_equal(out["atom_id"].to_numpy(dtype=int), ref["atom_id"].to_numpy(dtype=int))
    assert np.array_equal(out["n_bonds"].to_numpy(dtype=int), ref["n_bonds"].to_numpy(dtype=int))
    assert np.array_equal(out["has_hetero_bond"].to_numpy(dtype=bool), ref["has_hetero_bond"].to_numpy(dtype=bool))
    assert np.array_equal(out["is_undercoord"].to_numpy(dtype=bool), ref["is_undercoord"].to_numpy(dtype=bool))
    assert np.array_equal(out["ring_size_min"].to_numpy(dtype=int), ref["ring_size_min"].to_numpy(dtype=int))
    assert np.array_equal(out["ring_size_max"].to_numpy(dtype=int), ref["ring_size_max"].to_numpy(dtype=int))
    assert np.array_equal(out["label"].to_numpy(dtype=object), ref["label"].to_numpy(dtype=object))
    assert np.array_equal(out["seg_id"].to_numpy(dtype=int), ref["seg_id"].to_numpy(dtype=int))

    d_all_out = pd.to_numeric(out["d_pyr"], errors="coerce").to_numpy(dtype=float)
    d_all_ref = pd.to_numeric(ref["d_pyr"], errors="coerce").to_numpy(dtype=float)
    assert np.allclose(np.abs(d_all_out), np.abs(d_all_ref), atol=1.0e-10, equal_nan=True)

    for col in ["bond_strain", "angle_strain", "local_roughness", "psi6_re", "psi6_im", "psi6_mag", "psi6_ang"]:
        diff = np.abs(pd.to_numeric(out[col], errors="coerce") - pd.to_numeric(ref[col], errors="coerce"))
        assert float(np.nanmax(diff.to_numpy(dtype=float))) <= 1.0e-10

    assert np.array_equal(out["grain_id"].to_numpy(dtype=int), ref["grain_id"].to_numpy(dtype=int))

    # d_pyr parity target: match TRACT trend very closely across carbon atoms.
    is_c = ref["element"].to_numpy(dtype=object) == "C"
    d_out = pd.to_numeric(out.loc[is_c, "d_pyr"], errors="coerce").to_numpy(dtype=float)
    d_ref = pd.to_numeric(ref.loc[is_c, "d_pyr"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(d_out) & np.isfinite(d_ref)
    corr = float(np.corrcoef(d_out[mask], d_ref[mask])[0, 1])
    mae = float(np.mean(np.abs(d_out[mask] - d_ref[mask])))
    assert corr > 0.99
    assert mae < 0.005


def test_bo_mode_still_requires_bond_orders():
    atoms = ase_read(str(TRACT_FRAME), format="extxyz")
    xyz = np.asarray(atoms.get_positions(), dtype=float)
    elements = [str(e) for e in atoms.get_chemical_symbols()]
    n_atoms = len(elements)
    atom_ids = list(range(1, n_atoms + 1))
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
    angles = np.asarray(atoms.get_cell().angles(), dtype=float)
    sim = SimulationData(
        atom_ids=atom_ids,
        iterations=np.asarray([0], dtype=int),
        cell_lengths=np.asarray([lengths], dtype=float),
        cell_angles=np.asarray([angles], dtype=float),
    )
    traj = TrajectoryData(
        positions=np.asarray([xyz], dtype=float),
        elements=elements,
        atom_ids=atom_ids,
        iterations=np.asarray([0], dtype=int),
        simulation=sim,
    )
    conn = ConnectivityData(
        atom_ids=atom_ids,
        elements=elements,
        iterations=np.asarray([0], dtype=int),
        simulation=sim,
    )
    data = ConnectivityTrajectoryData(connectivity=conn, trajectory=traj)
    task = ActiveSiteStructuralTask()
    req = ActiveSiteStructuralRequest(frame=0, bond_mode="bo", include_noncarbon=True)
    with pytest.raises(ValueError, match="requires ConnectivityData.bond_orders"):
        task.run(data, req)


def test_required_data_resolution_switches_by_bond_mode():
    task = ActiveSiteStructuralTask()
    req_dist = ActiveSiteStructuralRequest(frame=0, bond_mode="distance")
    req_bo = ActiveSiteStructuralRequest(frame=0, bond_mode="bo")

    assert task.required_data_for(req_dist, {"bond_mode": "distance"}) is TrajectoryData
    assert task.required_data_for(req_bo, {"bond_mode": "bo"}) is ConnectivityTrajectoryData

    required_for_validation = task.required_data_for(req_dist, None)
    assert isinstance(required_for_validation, tuple)
    assert TrajectoryData in required_for_validation
    assert ConnectivityTrajectoryData in required_for_validation
