"""
Shared logging/time helpers for study workflows.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> str:
    """
    Utc now.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    None
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.logging import utc_now
    # Configure required arguments for your case.
    result = utc_now(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return datetime.now(timezone.utc).isoformat()


def local_now() -> str:
    """
    Local now.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    None
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.logging import local_now
    # Configure required arguments for your case.
    result = local_now(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def duration_minutes(started_at: str | None, finished_at: str | None) -> float | None:
    """
    Duration minutes.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    started_at : str | None
        Input parameter used by this function.
    finished_at : str | None
        Input parameter used by this function.
    
    Returns
    -----
    float | None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.logging import duration_minutes
    # Configure required arguments for your case.
    result = duration_minutes(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
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
    """
    Stage label width.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    max_stage_chars : int
        Input parameter used by this function.
    
    Returns
    -----
    int
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.logging import stage_label_width
    # Configure required arguments for your case.
    result = stage_label_width(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return 23 + int(max_stage_chars) + 2


def analysis_label_width(max_analysis_chars: int) -> int:
    """
    Analysis label width.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    max_analysis_chars : int
        Input parameter used by this function.
    
    Returns
    -----
    int
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.logging import analysis_label_width
    # Configure required arguments for your case.
    result = analysis_label_width(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
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
    """
    Log stage event.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    case_id : str
        Input parameter used by this function.
    replicate_id : str
        Input parameter used by this function.
    stage_name : str
        Input parameter used by this function.
    tag : str
        Input parameter used by this function.
    detail : str | None, optional
        Input parameter used by this function.
    stage_block_width : int | None, optional
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.logging import log_stage_event
    # Configure required arguments for your case.
    result = log_stage_event(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    tag_block = f"[{str(tag).upper()}]".ljust(8)
    stage_block = f"{case_id} {replicate_id} {stage_name}"
    if stage_block_width is not None:
        stage_block = stage_block.ljust(stage_block_width)
    line = f"{tag_block}{stage_block}{local_now()}"
    if detail:
        line = f"{line}  {detail}"
    print(line)


def log_task_event(tag: str, info: str, detail: str | None = None) -> None:
    """
    Log task event.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    tag : str
        Input parameter used by this function.
    info : str
        Input parameter used by this function.
    detail : str | None, optional
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.logging import log_task_event
    # Configure required arguments for your case.
    result = log_task_event(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    tag_block = f"[{str(tag).upper()}]".ljust(8)
    line = f"{tag_block}{str(info)}  {local_now()}"
    if detail:
        line = f"{line}  {detail}"
    print(line)

