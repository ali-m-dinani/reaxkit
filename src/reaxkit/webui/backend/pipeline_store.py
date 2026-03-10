"""In-memory pipeline state store for the Web UI backend."""

from __future__ import annotations

from dataclasses import asdict
from threading import RLock
from typing import Any

from reaxkit.webui.backend.schemas import PipelineNode, PipelineState, ResultArtifact, make_id


class PipelineStore:
    """Thread-safe in-memory storage for UI pipelines."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._pipelines: dict[str, PipelineState] = {}

    def create_pipeline(self, name: str = "Untitled Pipeline") -> PipelineState:
        with self._lock:
            pipeline = PipelineState(id=make_id("pipe"), name=name)
            self._pipelines[pipeline.id] = pipeline
            return pipeline

    def get_pipeline(self, pipeline_id: str) -> PipelineState:
        with self._lock:
            if pipeline_id not in self._pipelines:
                raise KeyError(f"Unknown pipeline '{pipeline_id}'")
            return self._pipelines[pipeline_id]

    def upsert_node(self, pipeline_id: str, node: PipelineNode) -> PipelineNode:
        with self._lock:
            pipeline = self.get_pipeline(pipeline_id)
            pipeline.nodes[node.id] = node
            parent_id = node.parent_id
            if parent_id:
                children = pipeline.children.setdefault(parent_id, [])
                if node.id not in children:
                    children.append(node.id)
            pipeline.touch()
            return node

    def get_node(self, pipeline_id: str, node_id: str) -> PipelineNode:
        pipeline = self.get_pipeline(pipeline_id)
        if node_id not in pipeline.nodes:
            raise KeyError(f"Unknown node '{node_id}' for pipeline '{pipeline_id}'")
        return pipeline.nodes[node_id]

    def update_node(
        self,
        pipeline_id: str,
        node_id: str,
        *,
        request: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        status: str | None = None,
        propagate_dirty: bool = True,
    ) -> PipelineNode:
        with self._lock:
            node = self.get_node(pipeline_id, node_id)
            if request is not None:
                node.request = request
            if metadata is not None:
                node.metadata.update(metadata)
            if status is not None:
                node.status = status
            node.touch()
            if propagate_dirty:
                self._mark_descendants_dirty(pipeline_id, node_id)
            self.get_pipeline(pipeline_id).touch()
            return node

    def store_artifact(self, pipeline_id: str, artifact: ResultArtifact) -> ResultArtifact:
        with self._lock:
            pipeline = self.get_pipeline(pipeline_id)
            pipeline.artifacts[artifact.id] = artifact
            pipeline.touch()
            return artifact

    def get_artifact(self, pipeline_id: str, artifact_id: str) -> ResultArtifact:
        pipeline = self.get_pipeline(pipeline_id)
        if artifact_id not in pipeline.artifacts:
            raise KeyError(f"Unknown artifact '{artifact_id}' for pipeline '{pipeline_id}'")
        return pipeline.artifacts[artifact_id]

    def snapshot(self, pipeline_id: str) -> dict[str, Any]:
        """Return a JSON-serializable pipeline snapshot."""
        pipeline = self.get_pipeline(pipeline_id)
        return asdict(pipeline)

    def delete_node(self, pipeline_id: str, node_id: str) -> dict[str, Any]:
        """Delete a node and its descendants, including associated artifacts."""
        with self._lock:
            pipeline = self.get_pipeline(pipeline_id)
            if node_id not in pipeline.nodes:
                raise KeyError(f"Unknown node '{node_id}' for pipeline '{pipeline_id}'")

            to_delete: list[str] = []
            queue = [str(node_id)]
            seen: set[str] = set()
            while queue:
                current = queue.pop(0)
                if current in seen:
                    continue
                seen.add(current)
                to_delete.append(current)
                queue.extend(pipeline.children.get(current, []))

            root = pipeline.nodes.get(str(node_id))
            if root is not None and root.parent_id and root.parent_id in pipeline.children:
                pipeline.children[root.parent_id] = [cid for cid in pipeline.children[root.parent_id] if cid != str(node_id)]

            artifact_ids: set[str] = set()
            for nid in to_delete:
                node = pipeline.nodes.get(nid)
                if node is None:
                    continue
                if node.result_ref:
                    artifact_ids.add(str(node.result_ref))
                last_id = node.metadata.get("last_artifact_id")
                if last_id:
                    artifact_ids.add(str(last_id))

            for artifact_id, artifact in list(pipeline.artifacts.items()):
                if str(artifact.node_id) in seen or str(artifact_id) in artifact_ids:
                    artifact_ids.add(str(artifact_id))

            # Keep artifacts that are still referenced by remaining nodes.
            referenced_by_remaining: set[str] = set()
            for rid, rnode in pipeline.nodes.items():
                if rid in seen:
                    continue
                if rnode.result_ref:
                    referenced_by_remaining.add(str(rnode.result_ref))
                last_id = rnode.metadata.get("last_artifact_id")
                if last_id:
                    referenced_by_remaining.add(str(last_id))
            artifact_ids = {aid for aid in artifact_ids if aid not in referenced_by_remaining}

            for aid in artifact_ids:
                pipeline.artifacts.pop(aid, None)

            for nid in to_delete:
                pipeline.nodes.pop(nid, None)
                pipeline.children.pop(nid, None)

            pipeline.touch()
            return {
                "deleted_node_ids": to_delete,
                "deleted_artifact_ids": sorted(artifact_ids),
            }

    def load_snapshot(self, snapshot: dict[str, Any]) -> PipelineState:
        """Load a pipeline snapshot into the store (replacing same id if present)."""
        with self._lock:
            pipeline_id = str(snapshot.get("id") or make_id("pipe"))
            pipeline = PipelineState(
                id=pipeline_id,
                name=str(snapshot.get("name") or "Imported Pipeline"),
            )
            pipeline.created_at = str(snapshot.get("created_at") or pipeline.created_at)
            pipeline.updated_at = str(snapshot.get("updated_at") or pipeline.updated_at)

            nodes_data = snapshot.get("nodes", {})
            if isinstance(nodes_data, dict):
                for key, raw in nodes_data.items():
                    if not isinstance(raw, dict):
                        continue
                    node = PipelineNode(
                        id=str(raw.get("id") or key),
                        kind=str(raw.get("kind") or "analysis"),  # type: ignore[arg-type]
                        name=str(raw.get("name") or "node"),
                        parent_id=raw.get("parent_id"),
                        status=str(raw.get("status") or "idle"),  # type: ignore[arg-type]
                        request=dict(raw.get("request") or {}),
                        result_ref=raw.get("result_ref"),
                        metadata=dict(raw.get("metadata") or {}),
                        created_at=str(raw.get("created_at") or ""),
                        updated_at=str(raw.get("updated_at") or ""),
                    )
                    pipeline.nodes[node.id] = node

            children_data = snapshot.get("children", {})
            if isinstance(children_data, dict):
                for pid, children in children_data.items():
                    if isinstance(children, list):
                        pipeline.children[str(pid)] = [str(c) for c in children]

            artifacts_data = snapshot.get("artifacts", {})
            if isinstance(artifacts_data, dict):
                for key, raw in artifacts_data.items():
                    if not isinstance(raw, dict):
                        continue
                    artifact = ResultArtifact(
                        id=str(raw.get("id") or key),
                        node_id=str(raw.get("node_id") or ""),
                        payload=dict(raw.get("payload") or {}),
                        metadata=dict(raw.get("metadata") or {}),
                        recommended_views=list(raw.get("recommended_views") or []),
                        created_at=str(raw.get("created_at") or ""),
                    )
                    pipeline.artifacts[artifact.id] = artifact

            self._pipelines[pipeline.id] = pipeline
            return pipeline

    def _mark_descendants_dirty(self, pipeline_id: str, node_id: str) -> None:
        pipeline = self.get_pipeline(pipeline_id)
        queue = list(pipeline.children.get(node_id, []))
        for candidate in pipeline.nodes.values():
            if not isinstance(candidate, PipelineNode):
                continue
            req = candidate.request if isinstance(candidate.request, dict) else {}
            meta = candidate.metadata if isinstance(candidate.metadata, dict) else {}
            right_source_id = str(req.get("right_source_node_id") or "").strip()
            source_node_ids = [str(v).strip() for v in (meta.get("source_node_ids") or []) if str(v).strip()]
            if right_source_id == str(node_id) or str(node_id) in source_node_ids:
                queue.append(candidate.id)
        while queue:
            child = queue.pop(0)
            child_node = pipeline.nodes.get(child)
            if child_node is None:
                continue
            child_node.status = "dirty"
            child_node.touch()
            queue.extend(pipeline.children.get(child, []))
