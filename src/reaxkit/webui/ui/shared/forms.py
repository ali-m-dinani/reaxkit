"""Small, metadata-only helpers for stable editors and searchable hierarchy."""
from __future__ import annotations

import hashlib
import json

from dash import html


def filter_tree_rows(rows, query):
    """Keep matching rows and their ancestors, without rebuilding pipeline data."""
    needle = str(query or "").strip().casefold()
    if not needle:
        return rows
    keep, ancestors = set(), []
    for index, row in enumerate(rows):
        props = row.to_plotly_json()["props"]
        depth = int(props.get("data-depth", 0))
        while ancestors and ancestors[-1][1] >= depth:
            ancestors.pop()
        if needle in str(props.get("data-label", "")).casefold():
            keep.add(index)
            keep.update(item[0] for item in ancestors)
        ancestors.append((index, depth))
    visible = [row for index, row in enumerate(rows) if index in keep]
    if not visible:
        return [html.Div("No matching nodes. Try a different name.", className="rk-hint")]
    if not any(getattr(row, "tabIndex", -1) == 0 for row in visible):
        visible[0].tabIndex = 0
    return visible


def properties_key(snapshot, session, result, config, curve):
    """Analysis drafts survive status/request changes; other editors track their inputs."""
    selected = str((session or {}).get("selected_node_id") or "")
    nodes = (snapshot or {}).get("nodes", [])
    if isinstance(nodes, dict):
        nodes = nodes.values()
    node = next((n for n in nodes if str(n.get("id")) == selected), None)
    identity = [(session or {}).get("pipeline_id"), selected]
    if selected.startswith('virtual:'):
        identity += [(config or {}).get('draft_viz_type'), (config or {}).get('engine_name')]
    if node and node.get('kind') == 'utility':
        identity += [node.get('status'), node.get('request', {}).get('right_source_node_id')]
    if node and node.get("kind") == "analysis":
        identity += [node.get("kind"), node.get("metadata", {}).get("task_name"), node.get("name")]
    else:
        # Status ticks and live field commits must not replace focused controls.
        identity += [[(n.get('id'), n.get('name'), n.get('kind'), n.get('result_ref')) for n in nodes],
                     (node or {}).get('request', {}).get('visualization_type'), curve]
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
