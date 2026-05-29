"""
Schema and validation helpers for study YAML.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from reaxkit.core.study.naming import slug_underscore


@dataclass(frozen=True)
class StageDef:
    """
    Stage Def.
    
    This dataclass defines a structured container used by ReaxKit core workflows.
    
    Fields
    -----
    name : str
        Field value used by this structured record.
    payload : dict[str, Any]
        Field value used by this structured record.
    """
    name: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class AnalysisDef:
    """
    Analysis Def.
    
    This dataclass defines a structured container used by ReaxKit core workflows.
    
    Fields
    -----
    analysis_id : str
        Field value used by this structured record.
    title : str
        Field value used by this structured record.
    run_stage : str
        Field value used by this structured record.
    payload : dict[str, Any]
        Field value used by this structured record.
    """
    analysis_id: str
    title: str
    run_stage: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class AggregateDef:
    """
    Aggregate Def.
    
    This dataclass defines a structured container used by ReaxKit core workflows.
    
    Fields
    -----
    title : str
        Field value used by this structured record.
    analysis_title : str
        Field value used by this structured record.
    x : str
        Field value used by this structured record.
    y : list[str]
        Field value used by this structured record.
    reducer : str
        Field value used by this structured record.
    stats : list[str]
        Field value used by this structured record.
    on_missing : str
        Field value used by this structured record.
    """
    title: str
    analysis_title: str
    x: str
    y: list[str]
    reducer: str
    stats: list[str]
    on_missing: str


@dataclass(frozen=True)
class ArtifactRef:
    """
    Artifact Ref.
    
    This dataclass defines a structured container used by ReaxKit core workflows.
    
    Fields
    -----
    stage : str
        Field value used by this structured record.
    artifact : str
        Field value used by this structured record.
    """
    stage: str
    artifact: str


def load_study_yaml(path: Path) -> dict[str, Any]:
    """
    Load study yaml.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    path : Path
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, Any]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.schema import load_study_yaml
    # Configure required arguments for your case.
    result = load_study_yaml(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    if not path.exists():
        raise FileNotFoundError(f"Study file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Study YAML must be a mapping at top level.")
    return raw


def load_source_study_doc(study_manifest: dict[str, Any]) -> dict[str, Any]:
    """
    Load source study doc.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    study_manifest : dict[str, Any]
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, Any]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.schema import load_source_study_doc
    # Configure required arguments for your case.
    result = load_source_study_doc(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    source_yaml = Path(str(study_manifest.get("source_yaml") or "")).resolve()
    if not source_yaml.exists():
        raise FileNotFoundError(f"Study source YAML not found: {source_yaml}")
    return load_study_yaml(source_yaml)


def analysis_defs_from_doc(doc: dict[str, Any]) -> dict[str, AnalysisDef]:
    """
    Analysis defs from doc.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    doc : dict[str, Any]
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, AnalysisDef]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.schema import analysis_defs_from_doc
    # Configure required arguments for your case.
    result = analysis_defs_from_doc(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    _, _, _, _, analyses, _ = validate_study(doc)
    out: dict[str, AnalysisDef] = {}
    for a in analyses:
        out[a.title] = a
    return out


def aggregate_defs_from_doc(doc: dict[str, Any]) -> dict[str, AggregateDef]:
    """
    Aggregate defs from doc.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    doc : dict[str, Any]
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, AggregateDef]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.schema import aggregate_defs_from_doc
    # Configure required arguments for your case.
    result = aggregate_defs_from_doc(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    raw = doc.get("aggregate") or []
    if not isinstance(raw, list):
        raise ValueError("aggregate must be a list when provided.")
    out: dict[str, AggregateDef] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each aggregate entry must be a mapping.")
        title = str(item.get("title") or "").strip()
        analysis_title = str(item.get("analysis_title") or "").strip()
        x = str(item.get("x") or "").strip()
        y_raw = item.get("y")
        reducer = str(item.get("reducer") or "identity").strip().lower()
        stats_raw = item.get("stats") or ["mean", "std", "min", "max", "sem", "n"]
        on_missing = str(item.get("on_missing") or "skip").strip().lower()

        if not title:
            raise ValueError("Each aggregate entry requires non-empty 'title'.")
        if title in out:
            raise ValueError(f"Duplicate aggregate title: {title}")
        if not analysis_title:
            raise ValueError(f"aggregate.{title}: analysis_title is required.")
        if not x:
            raise ValueError(f"aggregate.{title}: x is required.")

        if isinstance(y_raw, str):
            y = [y_raw.strip()] if y_raw.strip() else []
        elif isinstance(y_raw, list):
            y = [str(v).strip() for v in y_raw if str(v).strip()]
        else:
            y = []
        if not y:
            raise ValueError(f"aggregate.{title}: y must be a non-empty string or list.")

        if not isinstance(stats_raw, list) or not stats_raw:
            raise ValueError(f"aggregate.{title}: stats must be a non-empty list.")
        stats = [str(v).strip().lower() for v in stats_raw if str(v).strip()]
        if not stats:
            raise ValueError(f"aggregate.{title}: stats must contain at least one entry.")
        if on_missing not in {"skip", "fail"}:
            raise ValueError(f"aggregate.{title}: on_missing must be 'skip' or 'fail'.")

        out[title] = AggregateDef(
            title=title,
            analysis_title=analysis_title,
            x=x,
            y=y,
            reducer=reducer,
            stats=stats,
            on_missing=on_missing,
        )
    return out


def validate_study(
    doc: dict[str, Any],
) -> tuple[str, dict[str, list[Any]], int, list[StageDef], list[AnalysisDef], str | None]:
    """
    Validate study.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    doc : dict[str, Any]
        Input parameter used by this function.
    
    Returns
    -----
    tuple[str, dict[str, list[Any]], int, list[StageDef], list[AnalysisDef], str | None]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.schema import validate_study
    # Configure required arguments for your case.
    result = validate_study(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    study_name = str(doc.get("study_name") or "").strip()
    if not study_name:
        raise ValueError("study_name is required.")

    params_raw = doc.get("parameters") or {}
    if not isinstance(params_raw, dict) or not params_raw:
        raise ValueError("parameters must be a non-empty mapping.")
    parameters: dict[str, list[Any]] = {}
    for key, values in params_raw.items():
        key_s = str(key).strip()
        if not key_s:
            raise ValueError("Parameter names cannot be empty.")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Parameter '{key_s}' must be a non-empty list.")
        parameters[key_s] = list(values)

    replicates = int(doc.get("replicates", 1))
    if replicates < 1:
        raise ValueError("replicates must be >= 1.")

    run_raw = doc.get("run")
    if run_raw is None:
        run_raw = doc.get("workflow")
    if not isinstance(run_raw, list) or not run_raw:
        raise ValueError("run must be a non-empty list (or provide legacy workflow).")
    stages: list[StageDef] = []
    seen: set[str] = set()
    for item in run_raw:
        if not isinstance(item, dict):
            raise ValueError("Each run stage must be a mapping.")
        stage_name = str(item.get("stage") or "").strip()
        if not stage_name:
            raise ValueError("Each run stage requires a non-empty 'stage' name.")
        if stage_name in seen:
            raise ValueError(f"Duplicate run stage: {stage_name}")
        seen.add(stage_name)
        stages.append(StageDef(name=stage_name, payload=dict(item)))

    analysis_raw = doc.get("analysis") or []
    if not isinstance(analysis_raw, list):
        raise ValueError("analysis must be a list when provided.")
    analyses: list[AnalysisDef] = []
    seen_analysis_ids: set[str] = set()
    for idx, item in enumerate(analysis_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each analysis entry must be a mapping.")
        title = str(item.get("title") or "").strip()
        if not title:
            title = str(item.get("command") or "").strip()
        run_stage = str(item.get("run_stage") or "").strip()
        if not title:
            raise ValueError("Each analysis entry requires non-empty 'title'.")
        if not run_stage:
            raise ValueError("Each analysis entry requires non-empty 'run_stage'.")
        analysis_id = str(item.get("analysis_id") or f"analysis_{idx:02d}_{slug_underscore(title)}").strip()
        if analysis_id in seen_analysis_ids:
            raise ValueError(f"Duplicate analysis_id: {analysis_id}")
        seen_analysis_ids.add(analysis_id)
        analyses.append(
            AnalysisDef(
                analysis_id=analysis_id,
                title=title,
                run_stage=run_stage,
                payload=dict(item),
            )
        )

    template_raw = doc.get("template")
    template_dir: str | None = None
    if template_raw is not None:
        template_dir = str(template_raw).strip() or None

    return study_name, parameters, replicates, stages, analyses, template_dir

