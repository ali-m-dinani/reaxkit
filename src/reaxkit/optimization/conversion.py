"""Publish consistent, immutable normalized snapshots for offline/live readers."""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

from .config import validate_conversion
from .models import OptimizationError, identity, finite, progress
from .registry import converter, specification
from .storage import atomic_text, digest, lock, read_json, write_json


def collect_native(optimizer, native, output, options=None, metadata=None):
    native, output = Path(native).resolve(), Path(output).resolve()
    if output == native or output in native.parents:
        raise OptimizationError("Normalized output must not replace the native directory or its parents")
    validate_conversion(options or {})
    with lock(output, ".conversion.lock"):
        result = converter(optimizer)(native, options)
        events = native / "optimization-events.jsonl"
        if not result.progress and events.is_file():
            payload = events.read_bytes()
            result.sources[events.name] = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for number, line in enumerate(payload.splitlines(keepends=True), 1):
                if not line.endswith(b"\n"):
                    break
                try:
                    event = json.loads(line)
                    result.progress.append(progress(
                        events.name, number, event["score"],
                        native_step=event["completed"], step_kind="checkpoint",
                        elapsed_seconds=event["elapsed_seconds"],
                        record_kind="live_checkpoint", comparison_status="unverified",
                    ))
                except (ValueError, KeyError, TypeError):
                    result.warnings.append(f"{events.name}:{number}: invalid live checkpoint")
            result.warnings.append("Live checkpoints are completed optimizer units, not guaranteed engine evaluation counts")
        timing = None
        timing_file = native / "optimization-timing.json"
        if timing_file.is_file():
            try:
                timing = read_json(timing_file)
            except (OSError, ValueError):
                result.warnings.append("Timing file incomplete; retry collection")
        for artifact in result.artifacts:
            if artifact.get("restart") in (None, "neither"):
                continue
            source = (native / artifact["path"]).resolve()
            if native not in source.parents:
                raise OptimizationError("Checkpoint artifact escapes native directory")
            destination = output / "checkpoints" / artifact["sha256"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                temporary = destination.with_suffix(".tmp")
                try:
                    shutil.copyfile(source, temporary)
                    if digest(temporary) != artifact["sha256"]:
                        result.warnings.append(f"{artifact['path']}: changed while preserving checkpoint; retry")
                        continue
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
            artifact["preserved_path"] = str(destination.relative_to(output)).replace("\\", "/")
        previous_path = output / "conversion.json"
        previous = read_json(previous_path) if previous_path.is_file() else {}
        transitions = []
        for name, details in result.sources.items():
            old = previous.get("sources", {}).get(name)
            if old and old != details:
                change = "truncated" if details["size"] < old["size"] else "replaced_or_modified"
                if details["size"] > old["size"]:
                    checksum = hashlib.sha256()
                    remaining = old["size"]
                    try:
                        with (native / name).open("rb") as stream:
                            while remaining:
                                chunk = stream.read(min(remaining, 1024 * 1024))
                                if not chunk:
                                    break
                                checksum.update(chunk)
                                remaining -= len(chunk)
                        if remaining == 0 and checksum.hexdigest() == old["sha256"]:
                            change = "appended"
                    except OSError:
                        change = "changed_during_collection"
                transitions.append({
                    "source": name,
                    "change": change,
                    "previous_sha256": old["sha256"],
                })
        for name in set(previous.get("sources", {})) - set(result.sources):
            transitions.append({"source": name, "change": "removed"})
        if not result.progress:
            result.warnings.append("No supported progress records found")
        spec = specification(optimizer)
        metadata = metadata or {}
        for record in result.progress + result.predictions:
            source_profile = record.get("profile_source") is not None
            record.update({
                "optimizer": optimizer,
                "run_id": metadata.get("run_id"),
                "replica": metadata.get("replica"),
                "dataset_fingerprint": metadata.get("dataset_fingerprint"),
                "comparison_fingerprint": metadata.get("comparison_fingerprint"),
                "objective_definition": record.get("objective_definition", metadata.get("objective", spec["objective"])),
                "engine_identity": record.get("engine_identity", metadata.get("engine", spec["engine"])),
                "optimizer_version": record.get("optimizer_version", metadata.get("optimizer_version")),
                "engine_version": metadata.get("engine_version"),
                "comparison_status": "unverified" if source_profile else metadata.get("comparison_status", "unverified"),
                "dataset_scope": record.get("dataset_scope", metadata.get("dataset_scope", "native_input_semantics")),
                "seed_forwarded": metadata.get("seed_forwarded"),
            })
            if source_profile:
                record["comparison_fingerprint"] = identity(
                    metadata.get("comparison_fingerprint"), record["engine_identity"],
                    record["objective_definition"], record["dataset_scope"], record["optimizer_version"],
                )
        payload = {
            "progress": result.progress, "predictions": result.predictions,
            "artifacts": result.artifacts, "warnings": result.warnings,
            "sources": result.sources,
            "timing": timing,
        }
        snapshot_id = identity(payload)
        snapshot = output / "snapshots" / snapshot_id
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in ("progress", "predictions"):
            content = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in payload[name])
            atomic_text(snapshot / (name + ".jsonl"), content)
        write_json(snapshot / "artifacts.json", result.artifacts)
        report = {
            "schema_version": 1, "parser_version": 1, "optimizer": optimizer,
            "snapshot": "snapshots/" + snapshot_id, "sources": result.sources,
            "warnings": result.warnings, "progress_count": len(result.progress),
            "finite_progress_count": sum(
                finite(record.get("score")) is not None
                or finite(record.get("best_score")) is not None
                for record in result.progress
            ),
            "prediction_count": len(result.predictions), "artifact_count": len(result.artifacts),
            "coverage": "partial" if result.warnings else "available",
            "timing": timing,
        }
        write_json(snapshot / "conversion.json", report)
        if transitions:
            write_json(snapshot / "source_changes.json", transitions)
        write_json(output / "conversion.json", report)
        return report
