"""API surface for ReaxKit Web UI backend."""

from __future__ import annotations

from typing import Any

from reaxkit.webui.backend.node_runtime import PipelineRuntime
from reaxkit.webui.backend.pipeline_store import PipelineStore
from reaxkit.webui.backend.serializer import export_bundle, load_snapshot, save_snapshot


class WebUIApiService:
    """Framework-neutral backend service used by the Web UI."""

    def __init__(self) -> None:
        self.store = PipelineStore()
        self.runtime = PipelineRuntime(self.store)

    def create_pipeline(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        return self.runtime.create_pipeline(name=str(payload.get("name") or "Untitled Pipeline"))

    def get_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        return self.runtime.get_pipeline(pipeline_id)

    def load_dataset(self, pipeline_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.runtime.load_dataset(
            pipeline_id,
            run_dir=str(payload.get("run_dir") or "."),
            engine=payload.get("engine"),
            sources=payload.get("sources") or {},
            project_root=str(payload.get("project_root") or "") or None,
        )

    def update_dataset_sources(self, pipeline_id: str, node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.runtime.update_dataset_sources(pipeline_id, node_id, sources=dict(payload.get("sources") or {}))

    def add_node(self, pipeline_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.runtime.add_node(
            pipeline_id,
            parent_id=str(payload["parent_id"]),
            kind=str(payload["kind"]),
            name=str(payload["name"]),
            request=payload.get("request") or {},
            metadata=payload.get("metadata") or {},
        )

    def update_node(self, pipeline_id: str, node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.runtime.update_node(
            pipeline_id,
            node_id,
            request=payload.get("request"),
            metadata=payload.get("metadata"),
        )

    def delete_node(self, pipeline_id: str, node_id: str) -> dict[str, Any]:
        return self.runtime.delete_node(pipeline_id, node_id)

    def apply_node(self, pipeline_id: str, node_id: str) -> dict[str, Any]:
        return self.runtime.apply_node(pipeline_id, node_id)

    def get_node_result(self, pipeline_id: str, node_id: str) -> dict[str, Any]:
        return self.runtime.get_result(pipeline_id, node_id)

    def get_catalog(self) -> dict[str, Any]:
        return self.runtime.get_catalog()

    def export_pipeline(self, pipeline_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        out_path = str(payload.get("path") or f"./{pipeline_id}.pipeline.json")
        snapshot = self.store.snapshot(pipeline_id)
        written = save_snapshot(snapshot, out_path)
        return {"path": written}

    def load_pipeline_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = str(payload.get("path") or "")
        if not path:
            raise ValueError("Snapshot path is required")
        snapshot = load_snapshot(path)
        pipeline = self.store.load_snapshot(snapshot)
        return pipeline.to_dict()

    def export_pipeline_bundle(self, pipeline_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        out_dir = str(payload.get("path") or f"./{pipeline_id}.bundle")
        selected_node_id = payload.get("selected_node_id")
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
        return export_bundle(
            snapshot=snapshot,
            output_dir=out_dir,
            selected_node_id=str(selected_node_id) if selected_node_id else None,
            selected_artifact=selected_artifact,
        )


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
