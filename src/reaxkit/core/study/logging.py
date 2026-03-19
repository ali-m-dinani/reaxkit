"""Shared logging/time helpers for study workflows."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def local_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def duration_minutes(started_at: str | None, finished_at: str | None) -> float | None:
    if not started_at or not finished_at:
        return None
    try:
        start_dt = datetime.strptime(started_at, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(finished_at, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None
    delta_min = (end_dt - start_dt).total_seconds() / 60.0
    return round(delta_min, 3) if delta_min >= 0 else None


def stage_label_width(max_stage_chars: int) -> int:
    return 23 + int(max_stage_chars) + 2


def analysis_label_width(max_analysis_chars: int) -> int:
    return 24 + int(max_analysis_chars) + 2


def log_stage_event(
    case_id: str,
    replicate_id: str,
    stage_name: str,
    tag: str,
    detail: str | None = None,
    *,
    stage_block_width: int | None = None,
) -> None:
    tag_block = f"[{str(tag).upper()}]".ljust(8)
    stage_block = f"{case_id} {replicate_id} {stage_name}"
    if stage_block_width is not None:
        stage_block = stage_block.ljust(stage_block_width)
    line = f"{tag_block}{stage_block}{local_now()}"
    if detail:
        line = f"{line}  {detail}"
    print(line)


def log_task_event(tag: str, info: str, detail: str | None = None) -> None:
    tag_block = f"[{str(tag).upper()}]".ljust(8)
    line = f"{tag_block}{str(info)}  {local_now()}"
    if detail:
        line = f"{line}  {detail}"
    print(line)

