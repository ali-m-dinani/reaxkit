"""
ReaxFF kopple2 file generation utilities.

This module provides deterministic helpers for generating or writing a
default ReaxFF ``kopple2`` input file from a canonical template.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


KOPPLE2_TEMPLATE = """
REMARK 1: This file allows the user to establish links between 
different force field parameters during the force field optimization procedure.
REMARK 2: Template below shows that values for parameter '5 1 7' is retained for all 5 other
parameters mentioned below it.
5   1   7   5       !Parameter identifier; Nr. of links
5   2   7
5   3   7
5   4   7           !Parameters linked to parameter 5   1   7
5   5   7
5   6   7
"""

__all__ = [
    "Kopple2GeneratorSpec",
    "DEFAULT_KOPPLE2_SPEC",
    "KOPPLE2_GENERATOR_REGISTRY",
    "gen_template_kopple2",
]


@dataclass(frozen=True)
class Kopple2GeneratorSpec:
    """
    Declarative settings for generating a ReaxFF ``kopple2`` file.

    Parameters
    ----------
    template_text : str, optional
        Fully formatted ``kopple2`` file content. Defaults to the bundled
        canonical template.
    """

    template_text: str = KOPPLE2_TEMPLATE


DEFAULT_KOPPLE2_SPEC = Kopple2GeneratorSpec()


def _gen_template_kopple2_text(spec: Kopple2GeneratorSpec = DEFAULT_KOPPLE2_SPEC) -> str:
    """
    Generate the default ReaxFF ``kopple2`` file content as text.
    """
    return spec.template_text


def _write_kopple2_template(
    out_path: str | Path = "kopple2",
    spec: Kopple2GeneratorSpec = DEFAULT_KOPPLE2_SPEC,
) -> Path:
    """
    Write the default template ``kopple2`` file with no modifications.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_gen_template_kopple2_text(spec), encoding="utf-8")
    return out_path


def gen_template_kopple2(
    out_path: str | Path = "kopple2",
    spec: Kopple2GeneratorSpec = DEFAULT_KOPPLE2_SPEC,
) -> Path:
    """
    Generate template ``kopple2`` file.
    """
    return _write_kopple2_template(out_path=out_path, spec=spec)


KOPPLE2_GENERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    "kopple2": {
        "label": "ReaxFF Kopple2 File",
        "default_filename": "kopple2",
        "spec_type": Kopple2GeneratorSpec,
        "generate": _gen_template_kopple2_text,
        "write": gen_template_kopple2,
    }
}
