"""Pipeline runtime orchestration for Web UI operations."""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from typing import Any
import math

from reaxkit.webui.backend.adapters.engine_probe import detect_engine
from reaxkit.webui.backend.adapters.executor_adapter import run_analysis_task
from reaxkit.webui.backend.adapters.result_normalizer import normalize_result, recommend_views
from reaxkit.webui.backend.pipeline_store import PipelineStore
from reaxkit.webui.backend.schemas import DatasetInfo, PipelineNode, ResultArtifact, make_id
from reaxkit.presentation.specs import serialize_presentation_specs


class PipelineRuntime:
    """High-level operations over pipeline store and ReaxKit runtime."""

    def __init__(self, store: PipelineStore) -> None:
        self.store = store

    def create_pipeline(self, name: str = "Untitled Pipeline") -> dict[str, Any]:
        pipeline = self.store.create_pipeline(name=name)
        return pipeline.to_dict()

    def get_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        return self.store.snapshot(pipeline_id)

    def load_dataset(
        self,
        pipeline_id: str,
        *,
        run_dir: str,
        engine: str | None = None,
        sources: dict[str, str] | None = None,
        project_root: str | None = None,
    ) -> dict[str, Any]:
        engine_detected = detect_engine(run_dir, engine_override=engine)
        source_map = dict(sources or {})
        frames, atoms = self._probe_dataset_dimensions(
            run_dir=run_dir,
            engine_name=(engine or engine_detected),
            sources=source_map,
        )
        info = DatasetInfo(
            engine_detected=engine_detected,
            engine_override=engine,
            sources=source_map,
            frames=frames,
            atoms=atoms,
            fields=sorted(source_map.keys()),
        )
        pipeline = self.store.get_pipeline(pipeline_id)
        dataset_nodes = [n for n in pipeline.nodes.values() if n.kind == "dataset"]
        if dataset_nodes:
            dataset_nodes.sort(key=lambda n: str(n.updated_at))
            latest = dataset_nodes[-1]
            updated = self.store.update_node(
                pipeline_id,
                latest.id,
                request={},
                metadata={
                    "dataset": info.__dict__,
                    "run_dir": run_dir,
                    "project_root": str(project_root) if project_root else None,
                },
                status="done",
                propagate_dirty=True,
            )
            return updated.__dict__

        node = PipelineNode(
            id=make_id("node"),
            kind="dataset",
            name="Dataset",
            parent_id=None,
            status="done",
            metadata={
                "dataset": info.__dict__,
                "run_dir": run_dir,
                "project_root": str(project_root) if project_root else None,
            },
        )
        self.store.upsert_node(pipeline_id, node)
        return node.__dict__

    @staticmethod
    def _probe_dataset_dimensions(
        *,
        run_dir: str,
        engine_name: str | None,
        sources: dict[str, str],
    ) -> tuple[int | None, int | None]:
        """Best-effort probe for frames/atoms from trajectory data."""
        try:
            from reaxkit.core.engine_registry import resolve_engine
            from reaxkit.domain.data_models import TrajectoryData
            import reaxkit.engine  # noqa: F401

            adapter = resolve_engine(run_dir, engine=engine_name)
            args = {
                "run_dir": run_dir,
                "engine": engine_name,
                "xmolout": sources.get("trajectory"),
                "fort7": sources.get("bonds"),
                "summary": sources.get("summary"),
            }
            quick_probe = getattr(adapter, "quick_n_frames", None)
            if callable(quick_probe):
                try:
                    quick_frames = quick_probe(args)
                except Exception:
                    quick_frames = None
                if quick_frames is not None:
                    return (int(quick_frames), None)
            traj = adapter.load(TrajectoryData, args, reporter=None)
            atoms = len(getattr(traj, "atom_ids", []) or [])
            frames = None
            sim = getattr(traj, "simulation", None)
            if sim is not None:
                iterations = getattr(sim, "iterations", None)
                if iterations is not None:
                    frames = len(iterations)
            if frames is None:
                pos = getattr(traj, "positions", None)
                if pos is not None and hasattr(pos, "shape") and len(pos.shape) >= 1:
                    frames = int(pos.shape[0])
            return (int(frames) if frames is not None else None, int(atoms) if atoms else None)
        except Exception:
            return (None, None)

    def update_dataset_sources(self, pipeline_id: str, node_id: str, sources: dict[str, str]) -> dict[str, Any]:
        node = self.store.get_node(pipeline_id, node_id)
        if node.kind != "dataset":
            raise ValueError("Source overrides are only valid for dataset nodes")

        dataset = dict(node.metadata.get("dataset", {}))
        dataset["sources"] = dict(sources)
        dataset["fields"] = sorted(sources.keys())
        updated = self.store.update_node(
            pipeline_id,
            node_id,
            metadata={"dataset": dataset},
            status="done",
        )
        return updated.__dict__

    def add_node(
        self,
        pipeline_id: str,
        *,
        parent_id: str,
        kind: str,
        name: str,
        request: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        node = PipelineNode(
            id=make_id("node"),
            kind=kind,  # type: ignore[arg-type]
            name=name,
            parent_id=parent_id,
            request=dict(request or {}),
            metadata=dict(metadata or {}),
        )
        self.store.upsert_node(pipeline_id, node)
        return node.__dict__

    def update_node(
        self,
        pipeline_id: str,
        node_id: str,
        *,
        request: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        updated = self.store.update_node(
            pipeline_id,
            node_id,
            request=request,
            metadata=metadata,
        )
        return updated.__dict__

    def delete_node(self, pipeline_id: str, node_id: str) -> dict[str, Any]:
        """Delete a node (and descendants) from the pipeline."""
        return self.store.delete_node(pipeline_id, node_id)

    def apply_node(self, pipeline_id: str, node_id: str) -> dict[str, Any]:
        node = self.store.get_node(pipeline_id, node_id)
        self.store.update_node(pipeline_id, node_id, status="running", propagate_dirty=False)
        try:
            if node.kind == "analysis":
                artifact = self._run_analysis_node(pipeline_id, node)
                result = {
                    "node": self.store.get_node(pipeline_id, node_id).__dict__,
                    "artifact": artifact.__dict__,
                }
            elif node.kind == "utility":
                artifact = self._run_utility_node(pipeline_id, node)
                result = {
                    "node": self.store.get_node(pipeline_id, node_id).__dict__,
                    "artifact": artifact.__dict__,
                }
            elif node.kind == "visualization":
                artifact = self._run_visualization_node(pipeline_id, node)
                result = {
                    "node": self.store.get_node(pipeline_id, node_id).__dict__,
                    "artifact": artifact.__dict__,
                }
            else:
                updated = self.store.update_node(pipeline_id, node_id, status="done", propagate_dirty=False)
                result = {"node": updated.__dict__, "artifact": None}
            return result
        except Exception:
            self.store.update_node(pipeline_id, node_id, status="error", propagate_dirty=False)
            raise

    def _run_analysis_node(self, pipeline_id: str, node: PipelineNode) -> ResultArtifact:
        dataset_node = self._resolve_dataset_node(pipeline_id, str(node.parent_id))
        dataset_meta = dataset_node.metadata.get("dataset", {})
        run_dir = dataset_node.metadata.get("run_dir", ".")
        sources = dataset_meta.get("sources", {}) if isinstance(dataset_meta, dict) else {}
        runtime_args = {
            "run_dir": run_dir,
            "engine": dataset_meta.get("engine_override") or dataset_meta.get("engine_detected"),
            "xmolout": sources.get("trajectory"),
            "fort7": sources.get("bonds"),
            "summary": sources.get("summary"),
            "project_root": node.metadata.get("project_root") or dataset_node.metadata.get("project_root"),
        }
        task_name = str(node.metadata.get("task_name") or node.name).strip().lower().replace(" ", "_")
        result, task_cls = run_analysis_task(task_name, node.request, runtime_args)
        payload = normalize_result(result)
        recommended = self._recommend_views_for_task(task_cls, result, payload)
        artifact = ResultArtifact(
            id=make_id("artifact"),
            node_id=node.id,
            payload=payload,
            metadata={"task_name": task_name},
            recommended_views=recommended,
        )
        self.store.store_artifact(pipeline_id, artifact)
        node.result_ref = artifact.id
        self.store.upsert_node(pipeline_id, node)
        self.store.update_node(
            pipeline_id,
            node.id,
            status="done",
            metadata={"last_artifact_id": artifact.id},
            propagate_dirty=False,
        )
        # Promote latest utility artifact to the ancestor analysis node so
        # visualizations under that analysis consume transformed data.
        try:
            cursor = self.store.get_node(pipeline_id, str(node.parent_id))
            visited: set[str] = set()
            while True:
                if cursor.id in visited:
                    break
                visited.add(cursor.id)
                if cursor.kind == "analysis":
                    cursor.result_ref = artifact.id
                    self.store.upsert_node(pipeline_id, cursor)
                    self.store.update_node(
                        pipeline_id,
                        cursor.id,
                        status="done",
                        metadata={"last_artifact_id": artifact.id},
                        propagate_dirty=False,
                    )
                    break
                if not cursor.parent_id:
                    break
                cursor = self.store.get_node(pipeline_id, str(cursor.parent_id))
        except Exception:
            pass
        return artifact

    @staticmethod
    def _recommend_views_for_task(task_cls: type, result: object, payload: dict[str, Any]) -> list[dict[str, Any]]:
        hook = getattr(task_cls, "recommended_presentations", None)
        if callable(hook):
            try:
                views = hook(result, payload)
                if isinstance(views, list):
                    serialized = serialize_presentation_specs(views)
                    if serialized:
                        return serialized
            except Exception:
                pass
        return recommend_views(payload)

    def _run_utility_node(self, pipeline_id: str, node: PipelineNode) -> ResultArtifact:
        parent = self.store.get_node(pipeline_id, str(node.parent_id))
        artifact_id = parent.result_ref or parent.metadata.get("last_artifact_id")
        if not artifact_id:
            # If direct parent has no artifact yet, walk up to nearest ancestor
            # with materialized results (commonly the analysis node).
            cursor = parent
            visited: set[str] = set()
            while not artifact_id and cursor.parent_id:
                if cursor.id in visited:
                    break
                visited.add(cursor.id)
                cursor = self.store.get_node(pipeline_id, str(cursor.parent_id))
                artifact_id = cursor.result_ref or cursor.metadata.get("last_artifact_id")
        if not artifact_id:
            raise ValueError("Parent node has no result artifact to transform")
        parent_artifact = self.store.get_artifact(pipeline_id, str(artifact_id))
        base_rows = self._extract_rows(parent_artifact.payload)
        if not base_rows:
            raise ValueError("Parent artifact has no tabular payload")

        util_name = str(node.metadata.get("utility_name") or node.name).strip().lower()
        util_req = node.request if isinstance(node.request, dict) else {}

        if util_name in {"filter_rows", "filter_atoms"}:
            column = str(util_req.get("column") or "atom_id")
            values = util_req.get("values")
            keep = set()
            if isinstance(values, str):
                keep = {v.strip() for v in values.split(",") if v.strip()}
            elif isinstance(values, list):
                keep = {str(v) for v in values}
            out_rows = [r for r in base_rows if str(r.get(column)) in keep] if keep else list(base_rows)
        elif util_name == "frame_range":
            fmin = util_req.get("min_frame")
            fmax = util_req.get("max_frame")
            out_rows = []
            for r in base_rows:
                fv = r.get("frame_index")
                if fv is None:
                    continue
                try:
                    fvi = int(fv)
                except Exception:
                    continue
                if fmin is not None and fvi < int(fmin):
                    continue
                if fmax is not None and fvi > int(fmax):
                    continue
                out_rows.append(r)
        elif util_name in {"denoise_ema", "denoise_sma"}:
            column = str(util_req.get("column") or "msd")
            group_col = str(util_req.get("group_by") or "atom_id")
            x_col = str(util_req.get("x_col") or ("iter" if "iter" in base_rows[0] else "frame_index"))
            groups: dict[str, list[dict[str, Any]]] = {}
            for r in base_rows:
                groups.setdefault(str(r.get(group_col, "all")), []).append(dict(r))
            out_rows = []
            for _, rows in groups.items():
                rows.sort(key=lambda rr: float(rr.get(x_col, 0)))
                if util_name == "denoise_ema":
                    alpha = float(util_req.get("alpha") or 0.3)
                    ema = None
                    for rr in rows:
                        try:
                            val = float(rr.get(column))
                        except Exception:
                            out_rows.append(rr)
                            continue
                        ema = val if ema is None else (alpha * val + (1.0 - alpha) * ema)
                        rr[f"{column}_ema"] = ema
                        out_rows.append(rr)
                else:
                    window = max(1, int(util_req.get("window") or 5))
                    q: list[float] = []
                    for rr in rows:
                        try:
                            val = float(rr.get(column))
                        except Exception:
                            out_rows.append(rr)
                            continue
                        q.append(val)
                        if len(q) > window:
                            q.pop(0)
                        rr[f"{column}_sma"] = sum(q) / float(len(q))
                        out_rows.append(rr)
        elif util_name == "column_transform":
            src = str(util_req.get("source") or "msd")
            dst = str(util_req.get("new_column") or f"{src}_transformed")
            scale = float(util_req.get("scale") or 1.0)
            offset = float(util_req.get("offset") or 0.0)
            out_rows = []
            for r in base_rows:
                rr = dict(r)
                try:
                    rr[dst] = float(rr.get(src)) * scale + offset
                except Exception:
                    rr[dst] = math.nan
                out_rows.append(rr)
        else:
            raise ValueError(f"Unsupported utility node '{util_name}'")

        payload = {"table": out_rows}
        artifact = ResultArtifact(
            id=make_id("artifact"),
            node_id=node.id,
            payload=payload,
            metadata={"utility_name": util_name},
            recommended_views=recommend_views(payload),
        )
        self.store.store_artifact(pipeline_id, artifact)
        node.result_ref = artifact.id
        self.store.upsert_node(pipeline_id, node)
        self.store.update_node(
            pipeline_id,
            node.id,
            status="done",
            metadata={"last_artifact_id": artifact.id},
            propagate_dirty=False,
        )
        # Promote utility output to ancestor analysis and invalidate stale
        # visualization artifacts so table/plot controls read transformed columns.
        try:
            cursor = self.store.get_node(pipeline_id, str(node.parent_id))
            visited: set[str] = set()
            while True:
                if cursor.id in visited:
                    break
                visited.add(cursor.id)
                if cursor.kind == "analysis":
                    cursor.result_ref = artifact.id
                    self.store.upsert_node(pipeline_id, cursor)
                    self.store.update_node(
                        pipeline_id,
                        cursor.id,
                        status="done",
                        metadata={"last_artifact_id": artifact.id},
                        propagate_dirty=False,
                    )
                    self._invalidate_visualization_results(pipeline_id, cursor.id)
                    break
                if not cursor.parent_id:
                    break
                cursor = self.store.get_node(pipeline_id, str(cursor.parent_id))
        except Exception:
            pass
        return artifact

    def _invalidate_visualization_results(self, pipeline_id: str, root_node_id: str) -> None:
        pipeline = self.store.get_pipeline(pipeline_id)
        queue = list(pipeline.children.get(root_node_id, []))
        while queue:
            child_id = queue.pop(0)
            child = pipeline.nodes.get(child_id)
            if child is None:
                continue
            if child.kind == "visualization":
                child.result_ref = None
                child.metadata.pop("last_artifact_id", None)
                child.status = "dirty"
                child.touch()
            queue.extend(pipeline.children.get(child_id, []))

    def _run_visualization_node(self, pipeline_id: str, node: PipelineNode) -> ResultArtifact:
        parent = self.store.get_node(pipeline_id, str(node.parent_id))
        artifact_id = parent.result_ref or parent.metadata.get("last_artifact_id")
        if not artifact_id:
            raise ValueError("Parent node has no result artifact for visualization")
        parent_artifact = self.store.get_artifact(pipeline_id, str(artifact_id))
        payload = dict(parent_artifact.payload)
        artifact = ResultArtifact(
            id=make_id("artifact"),
            node_id=node.id,
            payload=payload,
            metadata={
                "visualization_type": str(node.request.get("visualization_type", "plot2d")) if isinstance(node.request, dict) else "plot2d",
                "request": dict(node.request) if isinstance(node.request, dict) else {},
            },
            recommended_views=recommend_views(payload),
        )
        self.store.store_artifact(pipeline_id, artifact)
        node.result_ref = artifact.id
        self.store.upsert_node(pipeline_id, node)
        self.store.update_node(
            pipeline_id,
            node.id,
            status="done",
            metadata={"last_artifact_id": artifact.id},
            propagate_dirty=False,
        )
        return artifact

    @staticmethod
    def _extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return [dict(v) for v in value]
        return []

    def _resolve_dataset_node(self, pipeline_id: str, start_node_id: str) -> PipelineNode:
        cursor = self.store.get_node(pipeline_id, start_node_id)
        visited: set[str] = set()
        while True:
            if cursor.id in visited:
                break
            visited.add(cursor.id)
            if cursor.kind == "dataset":
                return cursor
            if not cursor.parent_id:
                break
            cursor = self.store.get_node(pipeline_id, cursor.parent_id)
        raise ValueError("Could not resolve a dataset ancestor for this node")

    def get_result(self, pipeline_id: str, node_id: str) -> dict[str, Any]:
        node = self.store.get_node(pipeline_id, node_id)
        if not node.result_ref and "last_artifact_id" in node.metadata:
            node.result_ref = str(node.metadata["last_artifact_id"])
        if not node.result_ref:
            return {}
        artifact = self.store.get_artifact(pipeline_id, node.result_ref)
        return artifact.__dict__

    def get_catalog(self) -> dict[str, Any]:
        try:
            from reaxkit.core.analysis_task_registry import TASK_REGISTRY
            import reaxkit.analysis  # noqa: F401  (ensure tasks registered)
            task_names = sorted(TASK_REGISTRY.keys())
            schemas = self._analysis_schemas(TASK_REGISTRY)
        except Exception:
            task_names = []
            schemas = {}

        return {
            "analysis_tasks": task_names,
            "analysis_schemas": schemas,
            "utility_nodes": ["filter_rows", "frame_range", "denoise_ema", "denoise_sma", "column_transform"],
            "visualization_nodes": ["table", "plot", "histogram", "scene3d"],
        }

    @staticmethod
    def _analysis_schemas(task_registry: dict[str, type]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for task_name, task_cls in task_registry.items():
            try:
                request_type = task_cls.run.__annotations__.get("request")
                if request_type is None or not is_dataclass(request_type):
                    continue
                fields_payload = []
                for fld in fields(request_type):
                    if fld.default is not MISSING:
                        default_value = fld.default
                    elif fld.default_factory is not MISSING:  # type: ignore[attr-defined]
                        default_value = fld.default_factory()  # type: ignore[misc]
                    else:
                        default_value = None
                    fields_payload.append(
                        {
                            "name": fld.name,
                            "type": str(fld.type),
                            "default": default_value,
                        }
                    )
                out[task_name] = {
                    "request_type": request_type.__name__,
                    "fields": fields_payload,
                }
            except Exception:
                continue
        return out
