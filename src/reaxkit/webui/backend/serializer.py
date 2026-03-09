"""Pipeline snapshot serialization helpers."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any
from datetime import datetime, timezone
import csv


def save_snapshot(snapshot: dict[str, Any], path: str) -> str:
    """Persist a pipeline snapshot as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2, sort_keys=True)
    return str(target)


def load_snapshot(path: str) -> dict[str, Any]:
    """Load a pipeline snapshot JSON file."""
    target = Path(path)
    with target.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def export_bundle(
    *,
    snapshot: dict[str, Any],
    output_dir: str,
    selected_node_id: str | None = None,
    selected_artifact: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Export reproducibility bundle with snapshot and optional selected result."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    snapshot_path = root / "pipeline.snapshot.json"
    with snapshot_path.open("w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2, sort_keys=True)

    manifest: dict[str, Any] = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_id": snapshot.get("id"),
        "selected_node_id": selected_node_id,
        "files": {"snapshot": str(snapshot_path)},
    }

    if selected_artifact:
        artifact_json = root / "selected_result.json"
        with artifact_json.open("w", encoding="utf-8") as fh:
            json.dump(selected_artifact, fh, indent=2, sort_keys=True)
        manifest["files"]["selected_result_json"] = str(artifact_json)

        payload = selected_artifact.get("payload", {})
        rows = None
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    rows = value
                    break
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
