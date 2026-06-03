"""Typed schemas for Web UI pipeline state and API payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

NodeKind = Literal["dataset", "analysis", "utility", "visualization"]
NodeStatus = Literal["idle", "dirty", "running", "done", "error"]


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def make_id(prefix: str) -> str:
    """Create stable readable ids used in pipeline state."""
    return f"{prefix}_{uuid4().hex[:10]}"


@dataclass
class DatasetInfo:
    """Metadata for dataset nodes."""

    engine_detected: str
    engine_override: str | None = None
    sources: dict[str, str] = field(default_factory=dict)
    frames: int | None = None
    atoms: int | None = None
    fields: list[str] = field(default_factory=list)


@dataclass
class PipelineNode:
    """A node in the user-defined analysis pipeline."""

    id: str
    kind: NodeKind
    name: str
    parent_id: str | None = None
    status: NodeStatus = "idle"
    request: dict[str, Any] = field(default_factory=dict)
    result_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def touch(self) -> None:
        """Update the modified timestamp."""
        self.updated_at = utc_now_iso()


@dataclass
class ResultArtifact:
    """Serialized result from a pipeline node."""

    id: str
    node_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    recommended_views: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class PipelineState:
    """All state for one pipeline graph."""

    id: str
    name: str
    nodes: dict[str, PipelineNode] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=dict)
    artifacts: dict[str, ResultArtifact] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def touch(self) -> None:
        """Update the modified timestamp."""
        self.updated_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        """Convert nested dataclasses into plain dicts."""
        return asdict(self)

