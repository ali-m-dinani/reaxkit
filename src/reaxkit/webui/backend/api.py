"""API surface for ReaxKit Web UI backend."""

from __future__ import annotations

import logging
from typing import Any

from reaxkit.webui.backend.node_runtime import PipelineRuntime
from reaxkit.webui.backend.pipeline_store import PipelineStore
from reaxkit.webui.backend.serializer import export_bundle, load_snapshot, save_snapshot

logger = logging.getLogger(__name__)


def _payload_keys(payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(payload, dict):
        return []
    return sorted(str(k) for k in payload.keys())


def _artifact_shallow(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return lightweight artifact payload for UI transport (without heavy data table)."""
    out: dict[str, Any] = {
        "id": str(artifact.get("id") or ""),
        "node_id": str(artifact.get("node_id") or ""),
        "metadata": dict(artifact.get("metadata") or {}),
        "recommended_views": list(artifact.get("recommended_views") or []),
        "created_at": str(artifact.get("created_at") or ""),
    }
    return out


def _snapshot_shallow(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Trim heavy artifact payloads before returning snapshot to Dash stores."""
    if not isinstance(snapshot, dict):
        return {}
    out = dict(snapshot)
    artifacts = snapshot.get("artifacts", {})
    if isinstance(artifacts, dict):
        out["artifacts"] = {
            str(artifact_id): _artifact_shallow(artifact)
            for artifact_id, artifact in artifacts.items()
            if isinstance(artifact, dict)
        }
    return out


class WebUIApiService:
    """Framework-neutral backend service used by the Web UI."""

    def __init__(self) -> None:
        self.store = PipelineStore()
        self.runtime = PipelineRuntime(self.store)

    def create_pipeline(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        logger.debug("create_pipeline payload_keys=%s", _payload_keys(payload))
        out = self.runtime.create_pipeline(name=str(payload.get("name") or "Untitled Pipeline"))
        logger.debug("create_pipeline -> keys=%s", _payload_keys(out))
        return out

    def get_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        logger.debug("get_pipeline pipeline_id=%s", pipeline_id)
        out = _snapshot_shallow(self.runtime.get_pipeline(pipeline_id))
        logger.debug("get_pipeline -> keys=%s", _payload_keys(out))
        return out

    def load_dataset(self, pipeline_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        logger.debug(
            "load_dataset pipeline_id=%s run_dir=%s engine=%s payload_keys=%s",
            pipeline_id,
            payload.get("run_dir"),
            payload.get("engine"),
            _payload_keys(payload),
        )
        out = self.runtime.load_dataset(
            pipeline_id,
            run_dir=str(payload.get("run_dir") or "."),
            engine=payload.get("engine"),
            sources=payload.get("sources") or {},
            project_root=str(payload.get("project_root") or "") or None,
        )
        logger.debug("load_dataset -> keys=%s", _payload_keys(out))
        return out

    def update_dataset_sources(self, pipeline_id: str, node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        logger.debug(
            "update_dataset_sources pipeline_id=%s node_id=%s source_keys=%s",
            pipeline_id,
            node_id,
            sorted((payload.get("sources") or {}).keys()),
        )
        out = self.runtime.update_dataset_sources(pipeline_id, node_id, sources=dict(payload.get("sources") or {}))
        logger.debug("update_dataset_sources -> keys=%s", _payload_keys(out))
        return out

    def add_node(self, pipeline_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        logger.debug(
            "add_node pipeline_id=%s parent_id=%s kind=%s name=%s request_keys=%s metadata_keys=%s",
            pipeline_id,
            payload.get("parent_id"),
            payload.get("kind"),
            payload.get("name"),
            _payload_keys(payload.get("request") if isinstance(payload.get("request"), dict) else {}),
            _payload_keys(payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}),
        )
        out = self.runtime.add_node(
            pipeline_id,
            parent_id=str(payload["parent_id"]),
            kind=str(payload["kind"]),
            name=str(payload["name"]),
            request=payload.get("request") or {},
            metadata=payload.get("metadata") or {},
        )
        logger.debug("add_node -> node_id=%s kind=%s status=%s", out.get("id"), out.get("kind"), out.get("status"))
        return out

    def update_node(self, pipeline_id: str, node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        logger.debug(
            "update_node pipeline_id=%s node_id=%s request_keys=%s metadata_keys=%s",
            pipeline_id,
            node_id,
            _payload_keys(payload.get("request") if isinstance(payload.get("request"), dict) else {}),
            _payload_keys(payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}),
        )
        out = self.runtime.update_node(
            pipeline_id,
            node_id,
            request=payload.get("request"),
            metadata=payload.get("metadata"),
        )
        logger.debug("update_node -> status=%s", out.get("status"))
        return out

    def delete_node(self, pipeline_id: str, node_id: str) -> dict[str, Any]:
        logger.debug("delete_node pipeline_id=%s node_id=%s", pipeline_id, node_id)
        out = self.runtime.delete_node(pipeline_id, node_id)
        logger.debug(
            "delete_node -> deleted_nodes=%s deleted_artifacts=%s",
            len(out.get("deleted_node_ids", [])),
            len(out.get("deleted_artifact_ids", [])),
        )
        return out

    def apply_node(self, pipeline_id: str, node_id: str) -> dict[str, Any]:
        logger.debug("apply_node pipeline_id=%s node_id=%s", pipeline_id, node_id)
        out = self.runtime.apply_node(pipeline_id, node_id)
        artifact = out.get("artifact") if isinstance(out, dict) else None
        artifact_keys = _payload_keys(artifact if isinstance(artifact, dict) else {})
        logger.debug("apply_node -> keys=%s artifact_keys=%s", _payload_keys(out), artifact_keys)
        return out

    def get_node_result(self, pipeline_id: str, node_id: str) -> dict[str, Any]:
        logger.debug("get_node_result pipeline_id=%s node_id=%s", pipeline_id, node_id)
        out = self.runtime.get_result(pipeline_id, node_id)
        payload = out.get("payload") if isinstance(out, dict) else None
        logger.debug(
            "get_node_result -> artifact_keys=%s payload_keys=%s",
            _payload_keys(out),
            _payload_keys(payload if isinstance(payload, dict) else {}),
        )
        return out

    def get_catalog(self) -> dict[str, Any]:
        out = self.runtime.get_catalog()
        logger.debug(
            "get_catalog -> tasks=%s schemas=%s utilities=%s",
            len(out.get("analysis_tasks", [])),
            len((out.get("analysis_schemas") or {}).keys()),
            len(out.get("utility_nodes", [])),
        )
        return out

    def export_pipeline(self, pipeline_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        out_path = str(payload.get("path") or f"./{pipeline_id}.pipeline.json")
        logger.debug("export_pipeline pipeline_id=%s path=%s", pipeline_id, out_path)
        snapshot = self.store.snapshot(pipeline_id)
        written = save_snapshot(snapshot, out_path)
        logger.debug("export_pipeline -> path=%s", written)
        return {"path": written}

    def load_pipeline_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = str(payload.get("path") or "")
        if not path:
            raise ValueError("Snapshot path is required")
        logger.debug("load_pipeline_snapshot path=%s", path)
        snapshot = load_snapshot(path)
        pipeline = self.store.load_snapshot(snapshot)
        logger.debug("load_pipeline_snapshot -> nodes=%s", len(pipeline.nodes))
        return _snapshot_shallow(pipeline.to_dict())

    def export_pipeline_bundle(self, pipeline_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        out_dir = str(payload.get("path") or f"./{pipeline_id}.bundle")
        selected_node_id = payload.get("selected_node_id")
        logger.debug(
            "export_pipeline_bundle pipeline_id=%s out_dir=%s selected_node_id=%s",
            pipeline_id,
            out_dir,
            selected_node_id,
        )
        snapshot = self.store.snapshot(pipeline_id)
        selected_artifact = None
        if selected_node_id:
            try:
                node = self.store.get_node(pipeline_id, str(selected_node_id))
                artifact_id = node.result_ref or node.metadata.get("last_artifact_id")
                if artifact_id:
                    selected_artifact = self.store.get_artifact(pipeline_id, str(artifact_id)).__dict__
            except Exception:
                selected_artifact = None
        out = export_bundle(
            snapshot=snapshot,
            output_dir=out_dir,
            selected_node_id=str(selected_node_id) if selected_node_id else None,
            selected_artifact=selected_artifact,
        )
        logger.debug("export_pipeline_bundle -> files=%s", sorted((out.get("files") or {}).keys()))
        return out


def create_fastapi_app():
    """Create FastAPI app for the Web UI backend.

    Raises:
        RuntimeError: If FastAPI is unavailable.
    """
    try:
        from fastapi import FastAPI, HTTPException
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "FastAPI is not installed. Install it to run ReaxKit Web UI HTTP endpoints."
        ) from exc

    service = WebUIApiService()
    app = FastAPI(title="ReaxKit Web UI API", version="0.1.0")

    @app.post("/api/pipelines")
    def create_pipeline(payload: dict[str, Any] | None = None):
        return service.create_pipeline(payload)

    @app.get("/api/pipelines/{pipeline_id}")
    def get_pipeline(pipeline_id: str):
        try:
            return service.get_pipeline(pipeline_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/datasets/load")
    def load_dataset(payload: dict[str, Any]):
        try:
            pipeline_id = str(payload["pipeline_id"])
            return service.load_dataset(pipeline_id, payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch("/api/datasets/{pipeline_id}/{node_id}/sources")
    def update_sources(pipeline_id: str, node_id: str, payload: dict[str, Any]):
        try:
            return service.update_dataset_sources(pipeline_id, node_id, payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/pipelines/{pipeline_id}/nodes")
    def add_node(pipeline_id: str, payload: dict[str, Any]):
        try:
            return service.add_node(pipeline_id, payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch("/api/pipelines/{pipeline_id}/nodes/{node_id}")
    def update_node(pipeline_id: str, node_id: str, payload: dict[str, Any]):
        try:
            return service.update_node(pipeline_id, node_id, payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/pipelines/{pipeline_id}/nodes/{node_id}")
    def delete_node(pipeline_id: str, node_id: str):
        try:
            return service.delete_node(pipeline_id, node_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/pipelines/{pipeline_id}/nodes/{node_id}/apply")
    def apply_node(pipeline_id: str, node_id: str):
        try:
            return service.apply_node(pipeline_id, node_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/pipelines/{pipeline_id}/nodes/{node_id}/result")
    def get_result(pipeline_id: str, node_id: str):
        try:
            return service.get_node_result(pipeline_id, node_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/catalog")
    def catalog():
        return service.get_catalog()

    @app.post("/api/pipelines/{pipeline_id}/export")
    def export_pipeline(pipeline_id: str, payload: dict[str, Any] | None = None):
        try:
            return service.export_pipeline(pipeline_id, payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/pipelines/load")
    def load_pipeline_snapshot(payload: dict[str, Any]):
        try:
            return service.load_pipeline_snapshot(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/pipelines/{pipeline_id}/export_bundle")
    def export_bundle_route(pipeline_id: str, payload: dict[str, Any] | None = None):
        try:
            return service.export_pipeline_bundle(pipeline_id, payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
