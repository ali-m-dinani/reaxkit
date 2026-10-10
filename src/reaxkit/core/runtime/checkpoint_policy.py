"""Conservative opt-in capability gate for durable streaming results."""

from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from reaxkit.core.runtime.checkpoint_results import encode
from reaxkit.core.runtime.result_store import ResultStore, source_identity, _json_write


AUDITED_TASKS = {"HBNReferencePolarizationTask", "HBNReferenceLocalPolarizationTask", "PotentialElectricFieldTask",
                 "HBNReferenceProjectedPolarityTask", "BinnedDynamicChargeTask", "ChargeFieldTask"}
AUDITED_TASKS.update({"ChargeExtendedXYZTask", "PotentialElectricFieldTrajectoryTask"})
AUDITED_TASKS.update({"RDFTask", "RDFPropertyTask", "DihedralTask"})
AUDITED_TASKS.add("ConnectionStatsTask")
AUDITED_TASKS.add("BondEventsTask")
AUDITED_TASKS.update({"ConnectionListTask", "CoordinationStatusTask", "HybridizationStatusTask",
                     "TrajectoryCoordinateSeriesTask", "ChargeSeriesTask", "VoronoiScipyTask",
                     "VoronoiPyvoroTask", "VoronoiGeometryScipyTask", "VoronoiGeometryPyvoroTask"})


def checkpoint_enabled(task, args):
    supported = type(task).__name__ in AUDITED_TASKS
    explicit = args.get("checkpoint") is True or bool(args.get("resume"))
    if explicit and not supported:
        raise ValueError(f"Result checkpointing is not yet audited for {type(task).__name__}.")
    if args.get("resume") and args.get("checkpoint") is False:
        raise ValueError("--resume cannot be combined with --no-checkpoint.")
    return supported and (explicit or args.get("checkpoint") is not False and "output_profile" in args)


def open_checkpoint(task, request, args, source, directory):
    paths = [item["path"] for item in source["sources"]]
    for name, value in asdict(request).items():
        if (name.endswith("path") or name.endswith("file")) and isinstance(value, (str, Path)) and Path(value).is_file():
            paths.append(value)
    normalized_request = encode(request)
    normalized_request["fields"].pop("_output_path", None)
    sources = source_identity(paths)
    raw_root = Path(args.get("project_root") or ".").resolve() / "data" / "raw"
    for item in sources:
        if Path(item["path"]).is_relative_to(raw_root):
            item["path"] = f"snapshot:{Path(item['path']).name}"
    sources.sort(key=lambda item: item["path"])
    identity = {"task": f"{type(task).__module__}.{type(task).__name__}",
                "algorithm_version": str(getattr(task, "VERSION", "1")), "checkpoint_algorithm": 1,
                "request": normalized_request, "sources": sources,
                "engine": source["adapter"],
                "artifacts": {name: args.get(name) for name in ("output_profile", "write_displacements", "write_extxyz", "detail_format")}}
    destination = Path(args["resume"]) if args.get("resume") else Path(directory) / f"recovery-{uuid4().hex[:12]}" / "checkpoint"
    store = ResultStore(destination, identity, resume=bool(args.get("resume")),
                        buffer_bytes=int(float(args.get("checkpoint_buffer_mb", 32)) * 1024 * 1024),
                        interval_seconds=float(args.get("checkpoint_interval_seconds", 30)))
    try:
        if not args.get("resume") and args.get("_invocation_argv"):
            import os
            import shlex
            import subprocess
            command = ["reaxkit", *args["_invocation_argv"], "--resume", str(store.directory)]
            store.manifest["resume_command"] = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
            _json_write(store.manifest_path, store.manifest)
        if store.manifest["state"] not in {"analysis_complete", "complete"}:
            store.transition("running")
    except BaseException:
        store.close()
        raise
    print(f"Result checkpoint: {store.directory}")
    return store
