"""Pipeline snapshot serialization helpers."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any
from datetime import datetime, timezone
import csv
import shutil
from copy import deepcopy
from uuid import uuid4
from reaxkit.webui.backend.artifact_tables import is_table

from reaxkit.webui.backend.tabular_payload import extract_tabular_rows


def save_snapshot(snapshot: dict[str, Any], path: str, *, tables=None) -> str:
    """Persist a pipeline snapshot as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot = deepcopy(snapshot)
    copied = {}
    if tables is not None:
        snapshot['format_version'] = 2
        for artifact in snapshot.get('artifacts', {}).values():
            for value in artifact.get('payload', {}).values():
                if is_table(value):
                    source = tables.path(value)
                    if str(source) in copied:
                        value['file'] = copied[str(source)]
                        continue
                    folder = target.parent / (target.name + '.tables')
                    folder.mkdir(parents=True, exist_ok=True)
                    destination = folder / (uuid4().hex + '.parquet')
                    shutil.copy2(source, destination)
                    value['file'] = destination.relative_to(target.parent).as_posix()
                    copied[str(source)] = value['file']
    temporary = target.with_name(target.name + '.' + uuid4().hex + '.tmp')
    with temporary.open("w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2, sort_keys=True)
    temporary.replace(target)
    return str(target)


def load_snapshot(path: str, *, tables=None) -> dict[str, Any]:
    """Load a pipeline snapshot JSON file."""
    target = Path(path)
    with target.open("r", encoding="utf-8") as fh:
        snapshot = json.load(fh)
    if int(snapshot.get('format_version', 1)) > 2:
        raise ValueError('Unsupported pipeline snapshot format')
    if tables is not None:
        for artifact in snapshot.get('artifacts', {}).values():
            for value in artifact.get('payload', {}).values():
                if is_table(value):
                    source = (target.parent / value['file']).resolve()
                    if target.parent.resolve() not in source.parents or source.suffix != '.parquet':
                        raise ValueError('Snapshot contains an invalid table reference')
                    destination = tables.write_dir / (uuid4().hex + '.parquet')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    value['file'] = destination.relative_to(tables.root).as_posix()
    return snapshot


def export_bundle(
    *,
    snapshot: dict[str, Any],
    output_dir: str,
    selected_node_id: str | None = None,
    selected_artifact: dict[str, Any] | None = None,
    tables=None,
) -> dict[str, str]:
    """Export reproducibility bundle with snapshot and optional selected result."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    snapshot_path = root / "pipeline.snapshot.json"
    save_snapshot(snapshot, str(snapshot_path), tables=tables)

    manifest: dict[str, Any] = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_id": snapshot.get("id"),
        "selected_node_id": selected_node_id,
        "files": {"snapshot": str(snapshot_path)},
    }

    if selected_artifact:
        artifact_json = root / "selected_result.json"
        portable_artifact = load_snapshot(str(snapshot_path)).get('artifacts', {}).get(selected_artifact.get('id'), selected_artifact)
        with artifact_json.open("w", encoding="utf-8") as fh:
            json.dump(portable_artifact, fh, indent=2, sort_keys=True)
        manifest["files"]["selected_result_json"] = str(artifact_json)

        payload = selected_artifact.get("payload", {})
        if tables is not None:
            from reaxkit.webui.backend.artifact_tables import table_descriptor
            descriptor = table_descriptor(payload)
            if descriptor:
                tables.export_csv(descriptor, root / 'selected_result.csv')
                manifest['files']['selected_result_csv'] = str(root / 'selected_result.csv')
        rows = extract_tabular_rows(payload if isinstance(payload, dict) else None)
        if tables is not None and descriptor:
            rows = []
        if rows:
            csv_path = root / "selected_result.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            manifest["files"]["selected_result_csv"] = str(csv_path)

    manifest_path = root / "bundle.manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    return {
        "bundle_dir": str(root),
        "snapshot": str(snapshot_path),
        "manifest": str(manifest_path),
    }
