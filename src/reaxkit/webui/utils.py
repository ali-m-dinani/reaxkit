"""Dash UI utility helpers."""

from __future__ import annotations

from typing import Any


def node_label(node: dict[str, Any], depth: int) -> str:
    """Render an indented label for pipeline node selector."""
    indent = "  " * max(0, depth)
    name = str(node.get("name", "node"))
    status = str(node.get("status", "idle"))
    return f"{indent}{name} [{status}]"


def pipeline_roots(snapshot: dict[str, Any]) -> list[str]:
    """Return root node ids for a pipeline snapshot."""
    nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
    out: list[str] = []
    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
        if not node.get("parent_id"):
            out.append(str(node_id))
    return out


def flatten_nodes(snapshot: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    """Flatten pipeline tree into (depth, node) rows for UI rendering."""
    nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
    children = snapshot.get("children", {}) if isinstance(snapshot, dict) else {}
    ordered: list[tuple[int, dict[str, Any]]] = []

    def walk(node_id: str, depth: int) -> None:
        node = nodes.get(node_id)
        if not isinstance(node, dict):
            return
        ordered.append((depth, node))
        for child_id in children.get(node_id, []):
            walk(str(child_id), depth + 1)

    for root_id in pipeline_roots(snapshot):
        walk(root_id, 0)
    return ordered

