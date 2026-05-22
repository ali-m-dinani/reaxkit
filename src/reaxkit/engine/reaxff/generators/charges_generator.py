"""
ReaxFF charges file generation utilities.

This module provides deterministic helpers for generating or writing a
default ReaxFF ``charges`` input file from a canonical template.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHARGES_TEMPLATE = """
   REMARK 1: Charges file will be used when When a value of 5 is given for icharge keyword in the control-file
   REMARK 2: The below template is for a system containing 5 methane (molecule 1 to 5) 
and 5 water molecules (molecule 6 to 10).
   2              ! Number of molecule types (format i4)
   1   5   6  10  ! Molecule type definition (format 20i4)
Methane           ! Molecule 1 identifier
   5              ! Number of atoms in molecule 1
   1   -0.4800    ! Atom number; charge (format (i4,f10.6) )
   2    0.1200
   3    0.1200
   4    0.1200
   5    0.1200
Water             ! Molecule 2 identifier
   3              ! Number of atoms in molecule 2
   1   -0.8200    ! Atom number; charge
   2    0.4100
   3    0.4100
"""

__all__ = [
    "ChargesGeneratorSpec",
    "DEFAULT_CHARGES_SPEC",
    "CHARGES_GENERATOR_REGISTRY",
    "gen_template_charges",
]


@dataclass(frozen=True)
class ChargesGeneratorSpec:
    """
    Declarative settings for generating a ReaxFF ``charges`` file.

    Parameters
    ----------
    template_text : str, optional
        Fully formatted ``charges`` file content. Defaults to the bundled
        canonical template.
    """

    template_text: str = CHARGES_TEMPLATE


DEFAULT_CHARGES_SPEC = ChargesGeneratorSpec()


def _gen_template_charges_text(spec: ChargesGeneratorSpec = DEFAULT_CHARGES_SPEC) -> str:
    """
    Generate the default ReaxFF ``charges`` file content as text.
    """
    return spec.template_text


def _write_charges_template(
    out_path: str | Path = "charges",
    spec: ChargesGeneratorSpec = DEFAULT_CHARGES_SPEC,
) -> Path:
    """
    Write the default template ``charges`` file with no modifications.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_gen_template_charges_text(spec), encoding="utf-8")
    return out_path


def gen_template_charges(
    out_path: str | Path = "charges",
    spec: ChargesGeneratorSpec = DEFAULT_CHARGES_SPEC,
) -> Path:
    """
    Generate template ``charges`` file.
    """
    return _write_charges_template(out_path=out_path, spec=spec)


CHARGES_GENERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    "charges": {
        "label": "ReaxFF Charges File",
        "default_filename": "charges",
        "spec_type": ChargesGeneratorSpec,
        "generate": _gen_template_charges_text,
        "write": gen_template_charges,
    }
}
