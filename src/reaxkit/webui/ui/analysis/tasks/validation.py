"""Validate browser drafts before changing a scientific request."""
from __future__ import annotations

import math
import re


def _integers(raw, limit=100_000):
    """Bound selector expansion before allocation, including hostile ranges."""
    if isinstance(raw, (list, tuple)):
        tokens = raw
    else:
        tokens = re.split(r"[,\s]+", str(raw).strip())
    result = []
    for token in tokens:
        text = str(token).strip()
        if ":" in text:
            parts = text.split(":")
            if len(parts) not in (2, 3) or not parts[1]:
                raise ValueError("Use start:stop[:step] with a finite stop")
            start, stop = int(parts[0] or 0), int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            values = range(start, stop, step)
        elif match := re.fullmatch(r"(-?\d+)-(-?\d+)", text):
            start, stop = map(int, match.groups())
            step = 1 if stop >= start else -1
            values = range(start, stop + step, step)
        else:
            values = [int(text)]
        if len(values) + len(result) > limit:
            raise ValueError("Selection exceeds 100,000 entries; use sampling (every) or a smaller selection")
        result.extend(values)
    return result


def validate_request(schema, old_request, ids, values):
    """Return a complete request and field errors; never silently drop bad tokens."""
    fields = {f["name"]: f for f in (schema or {}).get("fields", []) if f.get("name")}
    request, errors = dict(old_request or {}), []
    for field_id, raw in zip(ids or [], values or []):
        name = str(field_id.get("name", ""))
        if name not in fields:
            continue
        field = fields[name]
        kind = str(field.get("kind", "")).replace(" ", "").lower()
        semantic = field.get("semantic") or {}
        is_list = any(token in kind for token in ("list[", "sequence[", "tuple[", "set["))
        try:
            if kind == "bool":
                value = isinstance(raw, list) and "on" in raw
            elif raw is None or raw == "":
                value = field.get("default")
                if value is None and field.get("required"):
                    raise ValueError("A value is required")
            elif is_list:
                if semantic.get("choices") is not None:
                    value = raw if isinstance(raw, list) else [raw]
                elif "[int]" in kind:
                    value = _integers(raw)
                else:
                    parts = raw if isinstance(raw, list) else [p.strip() for p in str(raw).split(",")]
                    value = [float(p) for p in parts] if "[float]" in kind else parts
            elif kind in {"int", "float"}:
                number = float(raw)
                if not math.isfinite(number) or (kind == "int" and not number.is_integer()):
                    raise ValueError("Enter a finite " + kind)
                value = int(number) if kind == "int" else number
            else:
                value = raw
            for item in value if isinstance(value, list) else [value]:
                if item is None:
                    continue
                if semantic.get("choices") is not None and item not in semantic["choices"]:
                    raise ValueError("Choose one of the listed values")
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    if not math.isfinite(item):
                        raise ValueError("Enter finite numbers")
                    if semantic.get("min") is not None and item < semantic["min"]:
                        raise ValueError(f"Minimum is {semantic['min']}")
                    if semantic.get("max") is not None and item > semantic["max"]:
                        raise ValueError(f"Maximum is {semantic['max']}")
            request[name] = value
        except (ValueError, TypeError, OverflowError) as exc:
            errors.append(f"{semantic.get('label') or name}: {exc}")
    return request, errors


def task_schema(service, node):
    name = str(node.get("metadata", {}).get("task_name") or node.get("name") or "").lower()
    return service.get_catalog().get("analysis_schemas", {}).get(name, {})
