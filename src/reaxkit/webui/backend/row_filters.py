"""Legacy row-filter semantics shared by previews, paged tables, and exports."""
from typing import Any


def _as_num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_filters_from_raw(raw):
    if not isinstance(raw, list):
        return []
    return [{'column': str(v['column']), 'op': str(v.get('op') or '=='), 'value': str(v.get('value') or '').strip()}
            for v in raw if isinstance(v, dict) and v.get('column')]

def _compare_row_filter(row_value: Any, op: str, filter_value: str) -> bool:
    op_norm = str(op or "==").strip().lower()
    left_num = _as_num(row_value)
    left_text = str(row_value or "").strip()
    raw = str(filter_value or "").strip()

    if op_norm in {"in", "not in"}:
        tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
        hit = left_text in tokens
        return (not hit) if op_norm == "not in" else hit
    if op_norm == "contains":
        return raw.lower() in left_text.lower()
    if op_norm == "between":
        parts = [tok.strip() for tok in raw.split(",") if tok.strip()]
        if len(parts) != 2:
            return True
        lo_num = _as_num(parts[0])
        hi_num = _as_num(parts[1])
        if left_num is not None and lo_num is not None and hi_num is not None:
            lo = min(lo_num, hi_num)
            hi = max(lo_num, hi_num)
            return lo <= left_num <= hi
        lo_txt, hi_txt = sorted(parts)
        return lo_txt <= left_text <= hi_txt

    right_num = _as_num(raw)
    if left_num is not None and right_num is not None:
        if op_norm == "==":
            return left_num == right_num
        if op_norm == "!=":
            return left_num != right_num
        if op_norm == ">":
            return left_num > right_num
        if op_norm == ">=":
            return left_num >= right_num
        if op_norm == "<":
            return left_num < right_num
        if op_norm == "<=":
            return left_num <= right_num
        return True

    if op_norm == "==":
        return left_text == raw
    if op_norm == "!=":
        return left_text != raw
    if op_norm == ">":
        return left_text > raw
    if op_norm == ">=":
        return left_text >= raw
    if op_norm == "<":
        return left_text < raw
    if op_norm == "<=":
        return left_text <= raw
    return True


def _apply_row_filters(rows: list[dict[str, Any]], raw_filters: Any) -> list[dict[str, Any]]:
    filters = _row_filters_from_raw(raw_filters)
    if not filters:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        keep = True
        for fil in filters:
            col = fil["column"]
            if col not in row:
                continue
            if not _compare_row_filter(row.get(col), fil["op"], fil["value"]):
                keep = False
                break
        if keep:
            out.append(row)
    return out


