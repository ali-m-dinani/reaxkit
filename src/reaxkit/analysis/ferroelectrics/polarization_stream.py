"""Bounded fixed-reference polarization with shared prepared geometry."""

from contextlib import closing
from dataclasses import replace
import numpy as np
from reaxkit.core.runtime.execution_contracts import FrameEnvelope, resolve_execution_policy
from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline
from reaxkit.core.runtime.reference_frames import reference_frames
from reaxkit.core.runtime.result_stream import source_result, combine_results
from reaxkit.core.runtime.trajectory_spool import TrajectorySpool
from reaxkit.core.runtime.artifacts import TableSpool, TableChunks
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import _trajectory_and_charges


def stream_polarization(task, frames, request, kind, reporter=None, pipeline=None):
    pipeline = pipeline or BoundedFramePipeline(resolve_execution_policy(task, request))
    local = replace(request, frames=[0], every=1, reference_frame=0)
    wanted = None if request.frames is None else set(list(request.frames)[::request.every])
    spool = TrajectorySpool() if kind in {"hbn_local", "basal_local"} else None
    profile = getattr(request, "_output_profile", None)
    detail_enabled = (profile not in {None, "minimal"} and kind.startswith("hbn")
                      and (profile in {"full", "legacy"} or kind == "hbn" and request.include_displacements
                           or bool(getattr(request, "_write_displacements", False))))
    details = TableSpool() if detail_enabled else None
    succeeded = False
    try:
        with reference_frames(frames, request.reference_frame) as (reference, source):
            reference_trajectory, _ = _trajectory_and_charges(reference)

            def prepare():
                if kind.startswith("hbn"):
                    from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import prepare_hbn_reference
                    return prepare_hbn_reference(reference_trajectory, local)
                from reaxkit.analysis.ferroelectrics.three_folded_wurtzite import polarity as pol
                trajectory, charges = pol._charges_for_request(reference, local)
                neighbors = pol.extract_wurtzite_neighbors(trajectory, local, charges=charges)
                initial = pol.calculate_wurtzite_polarity(neighbors, local)
                return {"basal": dict(zip(initial.table.site_atom_id.astype(int), initial.table["mean_basal_bond_c (angstrom)"].astype(float))),
                        "coordinates": np.asarray(trajectory.positions[0]).copy()}

            def kernel(data, state):
                if kind == "hbn":
                    from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import calculate_hbn_reference_polarization
                    return calculate_hbn_reference_polarization(data, local, prepared=state)
                if kind == "hbn_local":
                    from reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization import calculate_hbn_reference_local_polarization
                    return calculate_hbn_reference_local_polarization(data, local, prepared=state)
                from reaxkit.analysis.ferroelectrics.three_folded_wurtzite import polarity as pol
                trajectory, charges = pol._charges_for_request(data, local)
                neighbors = pol.extract_wurtzite_neighbors(trajectory, local, charges=charges)
                polarity = pol.calculate_wurtzite_polarity(neighbors, local, reference_basal=state["basal"])
                if kind == "three_binned":
                    from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarization import calculate_binned_polarization
                    return calculate_binned_polarization(data, local, polarity=polarity, reference_positions=state["coordinates"])
                from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.dipole import calculate_basal_plane_dipoles
                dipoles = calculate_basal_plane_dipoles(data, local, polarity=polarity)
                if kind == "basal_dipole":
                    return dipoles
                if kind == "basal_local":
                    from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.local_polarization import calculate_basal_plane_local_polarization
                    return calculate_basal_plane_local_polarization(data, local, dipoles=dipoles)
                from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.polarization import calculate_basal_plane_polarization
                return calculate_basal_plane_polarization(data, local, dipoles=dipoles, reference_positions=state["coordinates"])

            def selected():
                seen = set()
                for index, (frame, data) in enumerate(source):
                    if frame in wanted if wanted is not None else frame % request.every == 0:
                        seen.add(frame)
                        yield FrameEnvelope(index, frame, data)
                if wanted is not None and wanted-seen:
                    raise ValueError(f"Requested frames not found: {sorted(wanted-seen)}")

            results = []
            with closing(pipeline.map_reference(selected(), prepare, kernel)) as completed:
                for count, item in enumerate(completed, 1):
                    result = source_result(item.value, item.envelope.source_frame)
                    if profile is not None and not kind.startswith("hbn"):
                        # Neighbor geometry is required within a frame to derive
                        # dipoles/volumes, but is not a public artifact of these
                        # polarization commands. Keep their declared results only.
                        dipoles = getattr(result, "dipole_result", result)
                        polarity = getattr(dipoles, "polarity_result", None)
                        if polarity is not None:
                            for name in ("table", "centers", "neighbors", "neighbor_geometry",
                                         "apical_neighbors", "basal_neighbors", "ignored_neighbors"):
                                table = getattr(polarity, name, None)
                                if table is not None:
                                    setattr(polarity, name, table.iloc[:0])
                    if profile is not None and kind.startswith("hbn"):
                        target = result.reference_result if kind == "hbn_local" else result
                        if details is not None:
                            details.append(target.displacements)
                        target.displacements = target.displacements.iloc[:0]
                    if spool is not None:
                        trajectory, _ = _trajectory_and_charges(item.envelope.payload)
                        spool.append(item.envelope.source_frame, trajectory)
                        result.trajectory = None
                    results.append(result)
                    if reporter:
                        reporter("stream", count, 0, "Calculating polarization frames")
            result = combine_results(results, request)
            if details is not None:
                target = result.reference_result if kind == "hbn_local" else result
                target.table_chunks = {"hbn_reference_displacements": TableChunks(details)}
                result.skip_result_cache = True
            if spool is not None:
                result.trajectory = spool.finish()
                result._trajectory_spool = spool
                result.skip_result_cache = True
            succeeded = True
            return result
    finally:
        if spool is not None and not succeeded:
            spool.close()
        if details is not None and not succeeded:
            details.close()
