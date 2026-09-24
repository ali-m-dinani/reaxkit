"""Compare scientific frame kernels with their frozen materialized reference."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import pandas as pd

from reaxkit.analysis.timeseries.timeseries import (
    TrajectoryCoordinateSeriesTask, TrajectoryCoordinateSeriesRequest,
    TrajectoryDisplacementSeriesTask, TrajectoryDisplacementSeriesRequest,
)
from reaxkit.analysis.trajectory.dihedral import DihedralTask, DihedralRequest
from reaxkit.analysis.trajectory.voronoi import VoronoiScipyTask, VoronoiGeometryScipyTask, VoronoiRequest
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import WurtziteNeighborTask as FourNeighbors, WurtziteNeighborRequest as FourRequest
from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.neighbors import WurtziteNeighborTask as ThreeNeighbors, WurtziteNeighborRequest as ThreeRequest
from reaxkit.core.runtime.execution_contracts import resolve_execution_policy
from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline
from reaxkit.domain.data_models import SimulationData, TrajectoryData


def run(frames=32, atoms=128):
    random = np.random.default_rng(739)
    sim = SimulationData(atom_ids=list(range(1, atoms + 1)), iterations=np.arange(frames),
                         cell_lengths=np.full((frames, 3), 20.), cell_angles=np.full((frames, 3), 90.))
    data = TrajectoryData(positions=random.random((frames, atoms, 3))*20,
                          atom_ids=sim.atom_ids, elements=["B" if index % 2 else "N" for index in range(atoms)], iterations=sim.iterations, simulation=sim)
    def source():
        for index in range(frames):
            yield replace(data, positions=data.positions[index:index+1], iterations=np.array([index]),
                          source_frame_indices=np.array([index]),
                          simulation=replace(sim, iterations=np.array([index]), cell_lengths=sim.cell_lengths[index:index+1],
                                             cell_angles=sim.cell_angles[index:index+1]))
    cases = []
    for task, request in (
        (TrajectoryCoordinateSeriesTask(), TrajectoryCoordinateSeriesRequest()),
        (TrajectoryDisplacementSeriesTask(), TrajectoryDisplacementSeriesRequest()),
        (DihedralTask(), DihedralRequest(atom_ids=[1, 2, 3, 4])),
        (VoronoiScipyTask(), VoronoiRequest()),
        (VoronoiGeometryScipyTask(), VoronoiRequest()),
        (FourNeighbors(), FourRequest(centers=("B",), neighbors=("N",), charge_source="formal", formal_charges={"B": 0.0, "N": 0.0}, periodic=(False, False, False))),
        (ThreeNeighbors(), ThreeRequest(centers=("B",), neighbors=("N",), charge_source="formal", formal_charges={"B": 0.0, "N": 0.0}, periodic=(False, False, False))),
    ):
        start = perf_counter()
        expected = task.run(data, request).table
        materialized_seconds = perf_counter() - start
        for workers in (1, 2, 4):
            policy = resolve_execution_policy(task, request, {"workers": workers}, environ={"SLURM_CPUS_PER_TASK": "4"})
            runtime = BoundedFramePipeline(policy)
            wall, cpu = perf_counter(), process_time()
            result = task.run_stream(source(), request, pipeline=runtime).table
            elapsed, cpu = perf_counter()-wall, process_time()-cpu
            pd.testing.assert_frame_equal(result, expected, atol=1e-12, rtol=1e-12)
            cases.append({"task": f"{task.__class__.__module__}.{task.__class__.__name__}", "workers": workers, "frames": frames, "atoms": atoms,
                          "wall_seconds": elapsed, "cpu_seconds": cpu, "rows": len(result),
                          "materialized_seconds": materialized_seconds, "equivalent": True,
                          "peak_in_flight": runtime.metrics.peak_in_flight,
                          "peak_payload_bytes": runtime.metrics.peak_in_flight_bytes})
    return {"schema_version": 1, "seed": 739, "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--atoms", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/scientific_frame_maps.json"))
    args = parser.parse_args()
    report = run(args.frames, args.atoms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Passed {len(report['cases'])} scientific equivalence cases")
