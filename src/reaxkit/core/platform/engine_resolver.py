"""
Registry + resolver for engine adapters.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from reaxkit.core.platform.exceptions import ParseError

ENGINE_REGISTRY: dict[str, type] = {}


def register_engine(name: str):
    """
    Decorator to register an engine adapter class.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    name : str
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.platform.engine_resolver import register_engine
    # Configure required arguments for your case.
    result = register_engine(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """

    def wrapper(cls):
        ENGINE_REGISTRY[name] = cls
        cls.name = name
        return cls

    return wrapper


def _collect_engine_hints(target: Path) -> Dict[str, List[str]]:
    """Best-effort marker scan used to explain detection failures."""
    hints: Dict[str, List[str]] = {"reaxff": [], "ams": [], "lammps": []}

    if target.is_dir():
        reaxff_markers = [
            "xmolout",
            "fort.7",
            "fort.13",
            "fort.57",
            "fort.73",
            "fort.74",
            "fort.76",
            "fort.78",
            "fort.79",
            "fort.99",
            "ffield",
            "params",
            "trainset.in",
            "control",
            "molfra.out",
            "summary.txt",
            "geo",
            "vels",
        ]
        for name in reaxff_markers:
            if (target / name).exists():
                hints["reaxff"].append(name)

        for rkf in target.glob("*.rkf"):
            hints["ams"].append(rkf.name)

        for name in ("dump.lammpstrj", "log.lammps"):
            if (target / name).exists():
                hints["lammps"].append(name)
    else:
        lower_name = target.name.lower()
        if target.suffix.lower() == ".rkf":
            hints["ams"].append(target.name)
        if target.name in {"dump.lammpstrj", "log.lammps"}:
            hints["lammps"].append(target.name)
        if any(
            token in lower_name
            for token in (
                "xmolout",
                "fort.",
                "ffield",
                "params",
                "trainset",
                "molfra",
                "summary",
                "control",
                "geo",
                "vels",
            )
        ):
            hints["reaxff"].append(target.name)

    return hints


def _format_hints(hints: Dict[str, List[str]]) -> str:
    """
    Format hints.
    """
    parts: list[str] = []
    for engine_name in ("reaxff", "ams", "lammps"):
        rows = hints.get(engine_name) or []
        if not rows:
            continue
        shown = ", ".join(rows[:6])
        suffix = " ..." if len(rows) > 6 else ""
        parts.append(f"{engine_name}: [{shown}{suffix}]")
    return "; ".join(parts)


def resolve_engine(path: str, engine: str | None = None):
    """
    Resolve adapter by explicit engine or confidence-based auto-detection.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    path : str
        Input parameter used by this function.
    engine : str | None, optional
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.platform.engine_resolver import resolve_engine
    # Configure required arguments for your case.
    result = resolve_engine(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    if engine:
        if engine not in ENGINE_REGISTRY:
            raise ParseError(f"Unknown engine '{engine}'. Available engines: {sorted(ENGINE_REGISTRY)}")
        return ENGINE_REGISTRY[engine]()

    target = Path(path)
    scores: dict[str, float] = {}

    for name, cls in ENGINE_REGISTRY.items():
        score = cls().detect(target)
        scores[name] = float(score)

    best_name = max(scores, key=scores.get) if scores else None
    best_score = scores.get(best_name, -1.0) if best_name is not None else -1.0

    if best_name is None or best_score <= 0:
        hints = _collect_engine_hints(target)
        hinted = {k: v for k, v in hints.items() if v}
        if not hinted:
            raise ParseError(
                "Could not detect engine: the provided path has no recognizable ReaxFF/AMS/LAMMPS files. "
                "Set --engine explicitly and/or point --input/--run-dir to a directory containing engine outputs."
            )
        score_text = ", ".join(f"{k}={v:.2f}" for k, v in sorted(scores.items()))
        hint_text = _format_hints(hinted)
        raise ParseError(
            "Could not detect engine: engine-related files were found, but auto-detection confidence was zero for all engines. "
            f"Detected file hints -> {hint_text}. "
            f"Detector scores -> {score_text}. "
            "Pass --engine explicitly."
        )

    return ENGINE_REGISTRY[best_name]()
