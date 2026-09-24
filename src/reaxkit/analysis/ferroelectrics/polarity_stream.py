"""Shared reference preparation for three- and four-fold polarity."""

from contextlib import closing
from dataclasses import replace
import importlib
from reaxkit.core.runtime.execution_contracts import resolve_execution_policy, FrameEnvelope
from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline
from reaxkit.core.runtime.reference_frames import reference_frames
from reaxkit.core.runtime.artifacts import TableSpool, TableChunks


def stream_polarity(task, frames, request, reporter=None, pipeline=None, on_frame=None):
    module = importlib.import_module(task.__class__.__module__)
    pipeline = pipeline or BoundedFramePipeline(resolve_execution_policy(task, request))
    module._validate_request(request)
    local = replace(request, frames=[0], every=1)
    wanted = None if request.frames is None else set(list(request.frames)[::request.every])
    profile = getattr(request, "_output_profile", None)
    details = {name: TableSpool() for name in ("centers", "neighbors")} if profile in {"full", "legacy"} else {}

    def neighbors(data):
        trajectory, charges = module._charges_for_request(data, request)
        return module.extract_wurtzite_neighbors(trajectory, local, charges=charges,
                    frame_indices=[0], preserve_source_frame_indices=True)

    with reference_frames(frames, request.reference_frame) as (reference, source):
        def prepare():
            raw = module.calculate_wurtzite_polarity(neighbors(reference), request)
            return dict(zip(raw.table.site_atom_id.astype(int), raw.table["mean_basal_bond_c (angstrom)"].astype(float)))

        def selected():
            seen = set()
            for index, (source_frame, data) in enumerate(source):
                if source_frame in wanted if wanted is not None else source_frame % request.every == 0:
                    seen.add(source_frame)
                    yield FrameEnvelope(index, source_frame, data)
            if wanted is not None and wanted - seen:
                raise ValueError(f"Requested frames not found: {sorted(wanted-seen)}")

        def kernel(data, state):
            return module.calculate_wurtzite_polarity(neighbors(data), request, reference_basal=state)

        results = []
        with closing(pipeline.map_reference(selected(), prepare, kernel)) as completed:
            for count, item in enumerate(completed, 1):
                if on_frame is not None:
                    on_frame(item.envelope.payload, item.envelope.source_frame, item.value)
                if profile is not None:
                    if details:
                        from importlib import import_module
                        formatter = import_module(task.__class__.__module__.rsplit(".", 1)[0] + ".neighbors")
                        details["centers"].append(formatter.centers_csv_table(item.value.centers))
                        details["neighbors"].append(formatter.neighbors_csv_table(item.value.neighbor_geometry))
                    for name in ("centers", "neighbors", "neighbor_geometry", "apical_neighbors", "basal_neighbors", "ignored_neighbors"):
                        table = getattr(item.value, name, None)
                        if table is not None:
                            setattr(item.value, name, table.iloc[:0])
                results.append(item.value)
                if reporter:
                    reporter("stream", count, 0, "Calculating referenced polarity")
        result = module.combine_wurtzite_polarity_results(results, request)
        if details:
            result.detail_chunks = {name: TableChunks(spool) for name, spool in details.items()}
        if profile is not None:
            result.skip_result_cache = True
        return result


def stream_polarity_trajectory(task, frames, request, reporter=None, pipeline=None):
    """Write each completed reference-dependent frame before releasing it."""
    from reaxkit.engine.common.generators.extended_xyz_generator import ExtendedXYZWriter
    from pathlib import Path
    from reaxkit.core.runtime.artifacts import ArtifactSpec, ArtifactWriter
    module = importlib.import_module(task.__class__.__module__)
    polarity = importlib.import_module(task.__class__.__module__.rsplit(".", 1)[0] + ".polarity")
    module._validate_export_request(request)
    indices, iterations, atoms = [], [], []
    result = None
    path = Path(request._output_path)
    def produce(temporary):
        nonlocal result
        with ExtendedXYZWriter(temporary, precision=request.precision) as writer:
            def publish(data, source, frame_result):
                extended = module.polarity_extended_xyz_frame(data, request, frame_result.table, output_frame_index=source)
                writer.write_frame(extended)
                indices.append(source)
                iterations.append(int(extended.iteration))
                atoms.append(len(extended.species))
            result = stream_polarity(polarity.WurtzitePolarityTask(), frames, request,
                                     reporter=reporter, pipeline=pipeline, on_frame=publish)
    with ArtifactWriter(path.parent, [ArtifactSpec("trajectory", path.name, "core", True, "extxyz", True)],
                        overwrite=True, manifest_name=path.name + ".artifacts.json",
                        metadata={"command": getattr(task, "_reaxkit_task_name", task.__class__.__name__), "source_frames": indices}) as artifacts:
        artifacts.write_file("trajectory", produce)
    output = module._result(request, indices, iterations, atoms, result)
    output.skip_result_cache = True
    return output
