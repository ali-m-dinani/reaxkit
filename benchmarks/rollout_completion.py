"""Scientific and isolated-memory gates for the completion migrations."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter, process_time

import numpy as np
import pandas as pd

from reaxkit.core.runtime.benchmark import peak_rss_bytes
from reaxkit.core.runtime.execution_contracts import resolve_execution_policy
from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline
from reaxkit.domain.data_models import SimulationData, TrajectoryData
from reaxkit.analysis.stress_strain.z_binned_strain_using_top_bottom_atoms import ZBinnedTopBottomStrainRequest, ZBinnedTopBottomStrainTask
from reaxkit.analysis.stress_strain.z_binned_deformation_gradient_strain import ZBinnedDeformationGradientStrainRequest, ZBinnedDeformationGradientStrainTask


def trajectory_source(positions, iterations, lengths, elements):
    for index, xyz in enumerate(positions):
        sim = SimulationData(atom_ids=list(range(1, len(xyz)+1)), iterations=iterations[index:index+1],
                             cell_lengths=np.asarray(lengths[index:index+1]), cell_angles=np.full((1, 3), 90.))
        yield TrajectoryData(positions=xyz[None], atom_ids=sim.atom_ids, elements=elements,
                             iterations=sim.iterations, simulation=sim, source_frame_indices=np.array([index]))


def memory_case(frames, atoms=8192):
    """Only the current coordinates and four strain bins survive each frame."""
    import psutil
    base = np.random.default_rng(417).uniform(.1, 19.9, (atoms, 3))
    task, request = ZBinnedTopBottomStrainTask(), ZBinnedTopBottomStrainRequest(z_bins=4, unwrap=False)
    pipeline = BoundedFramePipeline(resolve_execution_policy(task, request, {"workers": 4}))
    reads = 0
    def source():
        nonlocal reads
        for index in range(frames):
            reads += 1
            sim = SimulationData(atom_ids=list(range(1, atoms+1)), iterations=np.array([index]),
                                 cell_lengths=np.full((1, 3), 20.), cell_angles=np.full((1, 3), 90.))
            yield TrajectoryData(positions=base[None].copy(), atom_ids=sim.atom_ids, elements=["Al"]*atoms,
                                 iterations=sim.iterations, simulation=sim, source_frame_indices=np.array([index]))
    process = psutil.Process()
    before = process.io_counters()
    wall, cpu = perf_counter(), process_time()
    result = task.run_stream(source(), request, pipeline=pipeline)
    wall, cpu = perf_counter()-wall, process_time()-cpu
    after = process.io_counters()
    assert reads == frames
    return {"frames": frames, "atoms": atoms, "rows": len(result.table), "source_frames_read": reads,
            "wall_seconds": wall, "cpu_seconds": cpu, "peak_rss_bytes": peak_rss_bytes(),
            "cpu_efficiency": cpu / (wall * pipeline.policy.allocated_cpus),
            "frames_per_second": frames / wall, "bytes_read": after.read_bytes-before.read_bytes,
            "bytes_written": after.write_bytes-before.write_bytes,
            "core_csv_bytes": len(result.table.to_csv(index=False).encode()), "policy": pipeline.policy.as_dict()}


def scientific_cases():
    from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarity import WurtzitePolarityTask as Three, WurtzitePolarityRequest as ThreeRequest
    from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.polarity import WurtzitePolarityTask as Four, WurtzitePolarityRequest as FourRequest
    from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.dipole import BasalPlaneDipoleTask, BasalPlaneDipoleRequest
    from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.polarization import BasalPlanePolarizationTask, BasalPlanePolarizationRequest
    from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.local_polarization import BasalPlaneLocalPolarizationTask, BasalPlaneLocalPolarizationRequest
    from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarization import BinnedPolarizationTask, BinnedPolarizationRequest
    from reaxkit.analysis.trajectory.diffusivity import DiffusivityTask, DiffusivityRequest
    frames = 24
    points = np.array([[0, 0, 0], [1.7, 0, -.5], [-.85, 1.472, -.5], [-.85, -1.472, -.5], [0, 0, 1.8]])
    points = np.concatenate([points + [i*5, j*5, k*5] for i in range(3) for j in range(3) for k in range(2)]) + 2
    xyz = np.stack([points + [0, 0, i*.003] for i in range(frames)])
    labels = ["Al", "N", "N", "N", "N"] * 18
    lengths = np.full((frames, 3), 20.)
    iterations = np.arange(frames)*10
    first = next(trajectory_source(xyz, iterations, lengths, labels))
    data = replace(first, positions=xyz, iterations=iterations, source_frame_indices=None,
                   simulation=replace(first.simulation, iterations=iterations, cell_lengths=lengths, cell_angles=np.full((frames, 3), 90.)))
    settings = dict(reference_frame=3, charge_source="formal", formal_charges={"Al": 3, "N": -3})
    cases = [(Three(), ThreeRequest(**settings)), (Four(), FourRequest(**settings)),
             (BasalPlaneDipoleTask(), BasalPlaneDipoleRequest(**settings)),
             (BasalPlanePolarizationTask(), BasalPlanePolarizationRequest(**settings, volume_method="cell")),
             (BasalPlaneLocalPolarizationTask(), BasalPlaneLocalPolarizationRequest(**settings)),
             (BinnedPolarizationTask(), BinnedPolarizationRequest(**settings, volume_method="cell")),
             (ZBinnedTopBottomStrainTask(), ZBinnedTopBottomStrainRequest(z_bins=3)),
             (ZBinnedDeformationGradientStrainTask(), ZBinnedDeformationGradientStrainRequest(z_bins=3)),
             (DiffusivityTask(), DiffusivityRequest(atom_ids=[1, 6, 11]))]
    records = []
    for task, request in cases:
        start = perf_counter()
        expected = task.run(data, request).table
        baseline = perf_counter() - start
        for workers in (1, 2, 4):
            policy = resolve_execution_policy(task, request, {"workers": workers}, environ={"SLURM_CPUS_PER_TASK": "4"})
            pipeline = BoundedFramePipeline(policy)
            wall, cpu = perf_counter(), process_time()
            runner = getattr(task, "run_stream", None) or task.run_blocks
            result = runner(trajectory_source(xyz, iterations, lengths, labels), request, pipeline=pipeline)
            elapsed, cpu = perf_counter()-wall, process_time()-cpu
            actual_table, expected_table = result.table, expected
            if "frame_index" in expected:
                # Legacy preparation sometimes emitted an included late reference
                # before frame zero. The runtime promises source-frame order.
                actual_table = actual_table.sort_values("frame_index", kind="stable").reset_index(drop=True)
                expected_table = expected.sort_values("frame_index", kind="stable").reset_index(drop=True)
                assert result.table.frame_index.is_monotonic_increasing
            pd.testing.assert_frame_equal(actual_table, expected_table, atol=1e-10, rtol=1e-10)
            if hasattr(result, "_trajectory_spool"):
                result._trajectory_spool.close()
            records.append({"task": task.__class__.__module__ + "." + task.__class__.__name__,
                "requested_workers": workers, "workers": policy.workers, "frames": frames, "atoms": len(points),
                "wall_seconds": elapsed, "cpu_seconds": cpu, "materialized_seconds": baseline,
                "equivalent": True, "rows": len(expected), "peak_in_flight": pipeline.metrics.peak_in_flight})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-child", type=int)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/rollout_completion.json"))
    args = parser.parse_args()
    if args.memory_child:
        print(json.dumps(memory_case(args.memory_child)))
    else:
        cases = scientific_cases()
        memory = []
        for frames in (16, 512):
            process = subprocess.run([sys.executable, __file__, "--memory-child", str(frames)], check=True, capture_output=True, text=True)
            memory.append(json.loads(process.stdout))
        assert memory[1]["peak_rss_bytes"] <= memory[0]["peak_rss_bytes"] + 32 * 1024 * 1024
        report = {"schema_version": 1, "scientific_cases": cases, "isolated_memory": memory,
                  "validation": {"equivalent": True, "bounded_memory": True, "single_source_pass": True}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Passed {len(cases)} scientific and {len(memory)} isolated memory cases")
