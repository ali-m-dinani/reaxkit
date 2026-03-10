"""Pipeline runtime orchestration for Web UI operations."""

from __future__ import annotations

import collections.abc
import importlib
import logging
import sys
import typing
import types
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, get_args, get_origin, get_type_hints

from reaxkit.webui.backend.adapters.engine_probe import detect_engine
from reaxkit.webui.backend.adapters.executor_adapter import run_analysis_task
from reaxkit.webui.backend.adapters.result_normalizer import normalize_result, recommend_views
from reaxkit.webui.backend.pipeline_store import PipelineStore
from reaxkit.webui.backend.schemas import DatasetInfo, PipelineNode, ResultArtifact, make_id
from reaxkit.webui.backend.tabular_payload import extract_tabular_rows
from reaxkit.webui.backend.utility_registry import (
    apply_utility_rows,
    canonical_utility_name,
    utility_specs_payload,
)
from reaxkit.presentation.specs import serialize_presentation_specs

logger = logging.getLogger(__name__)


class PipelineRuntime:
    """High-level operations over pipeline store and ReaxKit runtime."""

    def __init__(self, store: PipelineStore) -> None:
        self.store = store

    def create_pipeline(self, name: str = "Untitled Pipeline") -> dict[str, Any]:
        pipeline = self.store.create_pipeline(name=name)
        logger.debug("runtime.create_pipeline name=%s pipeline_id=%s", name, getattr(pipeline, "id", ""))
        return pipeline.to_dict()

    def get_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        logger.debug("runtime.get_pipeline pipeline_id=%s", pipeline_id)
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
        logger.debug(
            "runtime.load_dataset pipeline_id=%s run_dir=%s engine=%s source_keys=%s",
            pipeline_id,
            run_dir,
            engine or engine_detected,
            sorted(source_map.keys()),
        )
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
            logger.debug("runtime.load_dataset updated existing dataset node_id=%s", latest.id)
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
        logger.debug("runtime.load_dataset created dataset node_id=%s", node.id)
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
        logger.debug(
            "runtime.update_dataset_sources node_id=%s source_keys=%s",
            node_id,
            sorted(sources.keys()),
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
        logger.debug(
            "runtime.add_node pipeline_id=%s node_id=%s kind=%s name=%s parent=%s",
            pipeline_id,
            node.id,
            node.kind,
            node.name,
            node.parent_id,
        )
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
        logger.debug(
            "runtime.update_node pipeline_id=%s node_id=%s status=%s request_keys=%s metadata_keys=%s",
            pipeline_id,
            node_id,
            updated.status,
            sorted((request or {}).keys()),
            sorted((metadata or {}).keys()),
        )
        return updated.__dict__

    def delete_node(self, pipeline_id: str, node_id: str) -> dict[str, Any]:
        """Delete a node (and descendants) from the pipeline."""
        logger.debug("runtime.delete_node pipeline_id=%s node_id=%s", pipeline_id, node_id)
        return self.store.delete_node(pipeline_id, node_id)

    def apply_node(self, pipeline_id: str, node_id: str) -> dict[str, Any]:
        node = self.store.get_node(pipeline_id, node_id)
        logger.debug(
            "runtime.apply_node start pipeline_id=%s node_id=%s kind=%s name=%s",
            pipeline_id,
            node_id,
            node.kind,
            node.name,
        )
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
            logger.debug(
                "runtime.apply_node done node_id=%s status=%s has_artifact=%s",
                node_id,
                result.get("node", {}).get("status") if isinstance(result, dict) else "",
                bool((result.get("artifact") if isinstance(result, dict) else None)),
            )
            return result
        except Exception as exc:
            self.store.update_node(pipeline_id, node_id, status="error", propagate_dirty=False)
            logger.exception("runtime.apply_node failed node_id=%s error=%s", node_id, exc)
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
        logger.info(
            "runtime._run_analysis_node node_id=%s task=%s run_dir=%s request_keys=%s",
            node.id,
            task_name,
            run_dir,
            sorted((node.request or {}).keys()) if isinstance(node.request, dict) else [],
        )
        result, task_cls = run_analysis_task(task_name, node.request, runtime_args)
        payload = normalize_result(result)
        table_rows, table_cols = self._payload_table_overview(payload)
        logger.info(
            "runtime._run_analysis_node payload_keys=%s table_rows=%s table_cols=%s",
            sorted(payload.keys()),
            table_rows,
            table_cols,
        )
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
        artifact_id = self._resolve_materialized_artifact_id(pipeline_id, parent.id)
        if not artifact_id:
            raise ValueError("Parent node has no result artifact to transform")
        parent_artifact = self.store.get_artifact(pipeline_id, str(artifact_id))
        base_rows = self._extract_rows(parent_artifact.payload)
        if not base_rows:
            raise ValueError("Parent artifact has no tabular payload")

        util_name = canonical_utility_name(str(node.metadata.get("utility_name") or node.name))
        util_req = node.request if isinstance(node.request, dict) else {}
        logger.debug(
            "runtime._run_utility_node node_id=%s utility=%s request=%s base_rows=%s",
            node.id,
            util_name,
            util_req,
            len(base_rows),
        )
        other_rows: list[dict[str, Any]] | None = None
        if util_name == "join_tables":
            right_node_id = str(util_req.get("right_source_node_id") or "").strip()
            if not right_node_id:
                raise ValueError("join_tables requires right_source_node_id")
            right_artifact_id = self._resolve_materialized_artifact_id(pipeline_id, right_node_id)
            if not right_artifact_id:
                raise ValueError(f"Selected join source '{right_node_id}' has no result artifact")
            right_artifact = self.store.get_artifact(pipeline_id, str(right_artifact_id))
            other_rows = self._extract_rows(right_artifact.payload)
            if not other_rows:
                raise ValueError("Selected join source has no tabular payload")
        out_rows = apply_utility_rows(util_name, base_rows, util_req, other_rows=other_rows)
        logger.debug(
            "runtime._run_utility_node utility=%s out_rows=%s",
            util_name,
            len(out_rows),
        )

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
        table_rows, table_cols = self._payload_table_overview(payload)
        logger.debug(
            "runtime._run_visualization_node node_id=%s parent_node_id=%s parent_artifact_id=%s payload_keys=%s table_rows=%s table_cols=%s",
            node.id,
            parent.id,
            artifact_id,
            sorted(payload.keys()),
            table_rows,
            table_cols,
        )
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

    def _resolve_materialized_artifact_id(self, pipeline_id: str, start_node_id: str) -> str | None:
        cursor = self.store.get_node(pipeline_id, str(start_node_id))
        visited: set[str] = set()
        artifact_id = cursor.result_ref or cursor.metadata.get("last_artifact_id")
        while not artifact_id and cursor.parent_id:
            if cursor.id in visited:
                break
            visited.add(cursor.id)
            cursor = self.store.get_node(pipeline_id, str(cursor.parent_id))
            artifact_id = cursor.result_ref or cursor.metadata.get("last_artifact_id")
        return str(artifact_id) if artifact_id else None

    @staticmethod
    def _extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return extract_tabular_rows(payload)

    @staticmethod
    def _payload_table_overview(payload: dict[str, Any]) -> tuple[int, list[str]]:
        rows = extract_tabular_rows(payload)
        if not rows:
            return 0, []
        first = rows[0]
        if not isinstance(first, dict):
            return len(rows), []
        return len(rows), [str(k) for k in first.keys()]

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
            logger.debug("runtime.get_result node_id=%s no result_ref", node_id)
            return {}
        artifact = self.store.get_artifact(pipeline_id, node.result_ref)
        payload = artifact.payload if isinstance(artifact.payload, dict) else {}
        table_rows, table_cols = self._payload_table_overview(payload)
        logger.debug(
            "runtime.get_result node_id=%s artifact_id=%s payload_keys=%s table_rows=%s table_cols=%s",
            node_id,
            artifact.id,
            sorted(payload.keys()),
            table_rows,
            table_cols,
        )
        return artifact.__dict__

    def get_catalog(self) -> dict[str, Any]:
        try:
            from reaxkit.core.analysis_task_registry import TASK_REGISTRY
            try:
                import reaxkit.analysis  # noqa: F401  (best-effort registration)
            except Exception:
                # Fall back to per-module imports to avoid all-or-nothing failures from optional deps.
                try:
                    from reaxkit.core.analysis_cli_routing_registry import get_registered_analysis_commands

                    for spec in get_registered_analysis_commands().values():
                        module_path = str(getattr(spec, "module_path", "")).strip()
                        if not module_path:
                            continue
                        try:
                            importlib.import_module(module_path)
                        except Exception:
                            continue
                except Exception:
                    pass
            task_names = sorted(str(name) for name in TASK_REGISTRY.keys())
            schemas = self._analysis_schemas({str(k): v for k, v in TASK_REGISTRY.items()})
        except Exception:
            try:
                from reaxkit.core.analysis_cli_routing_registry import get_registered_analysis_commands

                task_names = sorted(str(name) for name in get_registered_analysis_commands().keys())
            except Exception:
                task_names = []
            schemas = {}

        utility_specs = utility_specs_payload()
        return {
            "analysis_tasks": task_names,
            "analysis_schemas": schemas,
            "utility_nodes": [str(spec.get("name")) for spec in utility_specs],
            "utility_specs": utility_specs,
            "visualization_nodes": ["table", "plot", "histogram", "scene3d"],
        }

    @staticmethod
    def _analysis_schemas(task_registry: dict[str, type]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for task_name, task_cls in task_registry.items():
            try:
                run_fn = task_cls.run
                mod = sys.modules.get(getattr(run_fn, "__module__", ""))
                globalns = vars(mod) if mod is not None else {}
                localns = vars(task_cls)
                try:
                    hints = get_type_hints(run_fn, globalns=globalns, localns=localns)
                except Exception:
                    hints = getattr(run_fn, "__annotations__", {})
                request_type = hints.get("request")
                if request_type is None or not is_dataclass(request_type):
                    continue
                req_mod = sys.modules.get(getattr(request_type, "__module__", ""))
                req_globalns = vars(req_mod) if req_mod is not None else {}
                req_localns = vars(request_type)
                try:
                    request_hints = get_type_hints(request_type, globalns=req_globalns, localns=req_localns)
                except Exception:
                    request_hints = {}
                fields_payload = []
                for fld in fields(request_type):
                    field_type = request_hints.get(fld.name, fld.type)
                    if fld.default is not MISSING:
                        default_value = fld.default
                    elif fld.default_factory is not MISSING:  # type: ignore[attr-defined]
                        default_value = fld.default_factory()  # type: ignore[misc]
                    else:
                        default_value = None
                    semantic: dict[str, Any] = {}
                    if fld.metadata:
                        for key in ("label", "help", "choices", "min", "max", "units"):
                            if key in fld.metadata:
                                semantic[key] = fld.metadata[key]
                    semantic = PipelineRuntime._enrich_semantic(
                        task_name=str(task_name),
                        field_name=fld.name,
                        field_type=field_type,
                        default_value=default_value,
                        semantic=semantic,
                    )
                    fields_payload.append(
                        {
                            "name": fld.name,
                            "type": str(field_type),
                            "kind": PipelineRuntime._field_kind(field_type),
                            "default": default_value,
                            "semantic": semantic,
                        }
                    )
                out[task_name] = {
                    "request_type": request_type.__name__,
                    "fields": fields_payload,
                }
            except Exception:
                continue
        return out

    @staticmethod
    def _humanize_field_name(name: str) -> str:
        tokens = str(name).strip("_").split("_")
        return " ".join(tok.upper() if tok.isupper() else tok.capitalize() for tok in tokens if tok)

    @staticmethod
    def _is_generic_help(text: str) -> bool:
        norm = str(text or "").strip().lower()
        return "parameter for" in norm or norm.endswith("parameter.")

    @staticmethod
    def _semantic_defaults(field_name: str) -> dict[str, Any]:
        defaults: dict[str, dict[str, Any]] = {
            "frames": {"help": "Frame indices to include. Empty means all frames.", "units": "frame_index"},
            "every": {"help": "Stride over selected frames.", "min": 1, "units": "frames"},
            "atom_ids": {"help": "Atom IDs to include. Empty means all atoms.", "units": "index"},
            "atom_ids_a": {"help": "Atom IDs for group A. Empty means all atoms.", "units": "index"},
            "atom_ids_b": {"help": "Atom IDs for group B. Empty means all atoms.", "units": "index"},
            "atom_types": {"help": "Element symbols to include."},
            "atom_types_a": {"help": "Element symbols for group A."},
            "atom_types_b": {"help": "Element symbols for group B."},
            "bins": {"help": "Number of histogram bins.", "min": 1, "units": "bins"},
            "r_max": {"help": "Maximum radius cutoff.", "min": 0.0, "units": "distance"},
            "threshold": {"help": "Threshold for event/status decision.", "min": 0.0},
            "hysteresis": {"help": "Hysteresis band around threshold.", "min": 0.0},
            "min_bo": {"help": "Minimum bond order to include.", "min": 0.0},
            "bo_threshold": {"help": "Bond-order threshold for filtering.", "min": 0.0},
            "xaxis": {"help": "X-axis selection for output plots."},
            "backend": {"help": "Computation backend implementation."},
            "scope": {"help": "Aggregation scope for analysis result."},
            "components": {"help": "Selected components to include."},
            "fields": {"help": "Selected fields/columns to include."},
            "section": {"help": "Named section to query."},
            "default": {"help": "Fallback value when the requested key is missing."},
            "key": {"help": "Lookup key or metric name."},
            "epochs": {"help": "Epoch indices to include. Empty means all epochs."},
            "sort": {"help": "Sort key/column."},
            "sort_by": {"help": "Sort key/column."},
            "ascending": {"help": "Sort order.", "choices": [True, False]},
        }
        return dict(defaults.get(field_name, {}))

    @staticmethod
    def _literal_choices(annotation: Any) -> list[Any] | None:
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is typing.Literal:
            return list(args)
        if origin in (types.UnionType, typing.Union):
            for item in args:
                sub = PipelineRuntime._literal_choices(item)
                if sub:
                    return sub
        return None

    @staticmethod
    def _enrich_semantic(
        *,
        task_name: str,
        field_name: str,
        field_type: Any,
        default_value: Any,
        semantic: dict[str, Any],
    ) -> dict[str, Any]:
        out = dict(semantic or {})
        label = str(out.get("label") or PipelineRuntime._humanize_field_name(field_name))
        out["label"] = label

        default_help = PipelineRuntime._semantic_defaults(field_name).get("help")
        help_text = out.get("help")
        if not isinstance(help_text, str) or not help_text.strip() or PipelineRuntime._is_generic_help(help_text):
            if isinstance(default_help, str) and default_help:
                out["help"] = default_help
            else:
                out["help"] = f"{label} for {task_name.replace('_', ' ')}."

        for k, v in PipelineRuntime._semantic_defaults(field_name).items():
            if k not in out:
                out[k] = v

        if "choices" not in out:
            literal = PipelineRuntime._literal_choices(field_type)
            if literal:
                out["choices"] = literal
            elif PipelineRuntime._field_kind(field_type) == "bool":
                out["choices"] = [True, False]

        if "choices" in out and out["choices"] is not None and not isinstance(out["choices"], list):
            try:
                out["choices"] = list(out["choices"])
            except Exception:
                out["choices"] = [out["choices"]]

        return out

    @staticmethod
    def _field_kind(annotation: Any) -> str:
        if isinstance(annotation, str):
            norm = annotation.replace(" ", "").lower()
            if norm in {"int", "builtins.int"}:
                return "int"
            if norm in {"float", "builtins.float"}:
                return "float"
            if norm in {"bool", "builtins.bool"}:
                return "bool"
            if norm in {"str", "builtins.str"}:
                return "str"
            if "list[int]" in norm or "sequence[int]" in norm or "tuple[int]" in norm or "set[int]" in norm:
                return "list[int]"
            if "list[float]" in norm or "sequence[float]" in norm or "tuple[float]" in norm or "set[float]" in norm:
                return "list[float]"
            if "list[str]" in norm or "sequence[str]" in norm or "tuple[str]" in norm or "set[str]" in norm:
                return "list[str]"
            if "list[" in norm or "sequence[" in norm or "tuple[" in norm or "set[" in norm:
                return "list[any]"
            return str(annotation)

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin in (types.UnionType, typing.Union):
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return PipelineRuntime._field_kind(non_none[0])
            kinds = [PipelineRuntime._field_kind(a) for a in non_none]
            return "|".join(kinds) if kinds else "any"

        if annotation is bool:
            return "bool"
        if annotation is int:
            return "int"
        if annotation is float:
            return "float"
        if annotation is str:
            return "str"

        if origin in (list, tuple, set, collections.abc.Sequence):
            item_kind = PipelineRuntime._field_kind(args[0]) if args else "any"
            return f"list[{item_kind}]"

        return str(annotation)
