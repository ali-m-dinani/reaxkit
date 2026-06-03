"""
Initialization helpers for study folder generation.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

import itertools
import shutil
from pathlib import Path
from typing import Any


def render_value(value: Any, context: dict[str, Any]) -> Any:
    """
    Render value.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    value : Any
        Input parameter used by this function.
    context : dict[str, Any]
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.init import render_value
    # Configure required arguments for your case.
    result = render_value(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    if isinstance(value, str):
        try:
            return value.format(**context)
        except KeyError as exc:
            missing = str(exc).strip("'")
            raise ValueError(f"Missing placeholder '{missing}' in study context.") from exc
    if isinstance(value, dict):
        return {str(k): render_value(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [render_value(v, context) for v in value]
    return value


def enumerate_cases(parameters: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Enumerate cases.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    parameters : dict[str, list[Any]]
        Input parameter used by this function.
    
    Returns
    -----
    list[dict[str, Any]]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.init import enumerate_cases
    # Configure required arguments for your case.
    result = enumerate_cases(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    keys = list(parameters.keys())
    values_grid = [parameters[k] for k in keys]
    combos = []
    for vals in itertools.product(*values_grid):
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


def resolve_template_path(template_value: str, *, study_dir: Path) -> Path:
    """
    Resolve template path.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    template_value : str
        Input parameter used by this function.
    study_dir : Path
        Input parameter used by this function.
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.init import resolve_template_path
    # Configure required arguments for your case.
    result = resolve_template_path(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    candidate = Path(str(template_value))
    if candidate.is_absolute():
        return candidate
    return (study_dir / candidate).resolve()


def copy_template_into_replicate(*, template_root: Path, replicate_dir: Path) -> None:
    """
    Copy template into replicate.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    template_root : Path
        Input parameter used by this function.
    replicate_dir : Path
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.init import copy_template_into_replicate
    # Configure required arguments for your case.
    result = copy_template_into_replicate(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    if not template_root.exists():
        raise FileNotFoundError(f"Study template directory not found: {template_root}")
    if not template_root.is_dir():
        raise NotADirectoryError(f"Study template path is not a directory: {template_root}")

    for source in template_root.iterdir():
        destination = replicate_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def apply_stage_slurm_job_name(*, stage_dir: Path, case_number: int, replicate_number: int, stage_name: str) -> None:
    """
    Apply stage slurm job name.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    stage_dir : Path
        Input parameter used by this function.
    case_number : int
        Input parameter used by this function.
    replicate_number : int
        Input parameter used by this function.
    stage_name : str
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.init import apply_stage_slurm_job_name
    # Configure required arguments for your case.
    result = apply_stage_slurm_job_name(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    submit_script = stage_dir / "submit_and_wait.sh"
    if not submit_script.exists():
        return

    job_name = f"C{case_number:02d}_R{replicate_number:02d}_{stage_name}"
    lines = submit_script.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    replaced = False
    for line in lines:
        if line.strip().startswith("#SBATCH --job-name="):
            out_lines.append(f"#SBATCH --job-name={job_name}")
            replaced = True
        else:
            out_lines.append(line)

    if not replaced:
        insert_at = 1 if out_lines and out_lines[0].startswith("#!") else 0
        out_lines.insert(insert_at, f"#SBATCH --job-name={job_name}")

    submit_script.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def collect_template_stage_relpaths(*, template_root: Path, stage_names: list[str]) -> dict[str, Path]:
    """
    Collect template stage relpaths.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    template_root : Path
        Input parameter used by this function.
    stage_names : list[str]
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, Path]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.init import collect_template_stage_relpaths
    # Configure required arguments for your case.
    result = collect_template_stage_relpaths(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    stage_to_relpath: dict[str, Path] = {}
    for stage_name in stage_names:
        matches: list[Path] = []
        for candidate in template_root.rglob(stage_name):
            if candidate.is_dir():
                matches.append(candidate)
        if not matches:
            raise ValueError(
                f"Template validation failed: stage folder '{stage_name}' not found under template '{template_root}'."
            )
        if len(matches) > 1:
            rels = ", ".join(str(p.relative_to(template_root)) for p in matches)
            raise ValueError(
                f"Template validation failed: stage folder '{stage_name}' is ambiguous under template '{template_root}': {rels}"
            )
        stage_to_relpath[stage_name] = matches[0].relative_to(template_root)
    return stage_to_relpath

