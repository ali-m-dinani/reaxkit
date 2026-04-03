"""
ReaxFF addmol file generation utilities.

This module provides deterministic helpers for generating or writing default
ReaxFF ``addmol.bgf`` and ``addmol.vel`` files from canonical templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

ADDMOL_TEMPLATE = """BIOGRF 200
DESCRP O2
REMARK 1 FREQADD: Indicates how often the molecule should be added to the simulation box.
REMARK 2 VELADD: 1: random initial velocities, according to TADDMOL; 2: read in velocities from addmol.vel
REMARK 3 STARTX, STARTY, STARTZ: position of the centre-of-mass of the added molecule. If < -5000.0: random position. 
REMARK 4 ADDIST:  minimal distance between the added molecule and the other atoms in the simulation box.
REMARK 5 NATTEMPT: Indicates the number of attempts to add the molecule to the simulation box before giving up.
REMARK 6 TADDMOL: Temperature of added molecule; only used with VELADD=1.
FREQADD  1000
VELADD  1
STARTX -9000.0
STARTY -9000.0
STARTZ -9000.0
ADDIST 3.0
NATTEMPT 050
TADDMOL 250.0
FORMAT ATOM
(a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5)
HETATM     1 O                   0.00000   0.00000   0.00000 O      1 3  0.00000
HETATM     2 O                   1.24500   0.00000   0.00000 O      1 3  0.00000
END
"""

ADDMOL_VEL_TEMPLATE = """Atom velocities (Angstrom/s):
   0.676920600871422E+13   0.250389491659681E+13   0.385204179579294E+03
   0.638733784812228E+13   0.125025292253893E+13  -0.372288199177400E+03
"""

__all__ = [
    "AddmolGeneratorSpec",
    "DEFAULT_ADDMOL_SPEC",
    "ADDMOL_GENERATOR_REGISTRY",
    "generate_addmol_template",
    "generate_addmol_vel_template",
    "write_addmol",
    "write_addmol_template",
    "write_addmol_vel_template",
]


@dataclass(frozen=True)
class AddmolGeneratorSpec:
    """
    Declarative settings for generating ``addmol`` files.

    Parameters
    ----------
    template_text : str, optional
        Fully formatted ``addmol.bgf`` content. Defaults to the bundled
        canonical template.
    vel_template_text : str, optional
        Fully formatted ``addmol.vel`` content. Defaults to the bundled
        canonical template.
    """

    template_text: str = ADDMOL_TEMPLATE
    vel_template_text: str = ADDMOL_VEL_TEMPLATE


DEFAULT_ADDMOL_SPEC = AddmolGeneratorSpec()


def generate_addmol_template(spec: AddmolGeneratorSpec = DEFAULT_ADDMOL_SPEC) -> str:
    """
    Generate the default ReaxFF ``addmol.bgf`` file content as text.
    """
    return spec.template_text


def generate_addmol_vel_template(spec: AddmolGeneratorSpec = DEFAULT_ADDMOL_SPEC) -> str:
    """
    Generate the default ReaxFF ``addmol.vel`` file content as text.
    """
    return spec.vel_template_text


def write_addmol_template(
    out_path: str | Path = "addmol.bgf",
    spec: AddmolGeneratorSpec = DEFAULT_ADDMOL_SPEC,
) -> Path:
    """
    Write the default template ``addmol.bgf`` file with no modifications.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(generate_addmol_template(spec), encoding="utf-8")
    return out_path


def write_addmol_vel_template(
    out_path: str | Path = "addmol.vel",
    spec: AddmolGeneratorSpec = DEFAULT_ADDMOL_SPEC,
) -> Path:
    """
    Write the default template ``addmol.vel`` file with no modifications.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(generate_addmol_vel_template(spec), encoding="utf-8")
    return out_path


def write_addmol(
    out_path: str | Path = "addmol.bgf",
    spec: AddmolGeneratorSpec = DEFAULT_ADDMOL_SPEC,
) -> Path:
    """
    Backward-compatible wrapper that writes both ``addmol.bgf`` and ``addmol.vel``.

    Returns the path to the written ``addmol.bgf`` file.
    """
    bgf_path = write_addmol_template(out_path=out_path, spec=spec)
    vel_path = Path(bgf_path).with_name("addmol.vel")
    write_addmol_vel_template(out_path=vel_path, spec=spec)
    return bgf_path


ADDMOL_GENERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    "addmol": {
        "label": "ReaxFF Addmol File",
        "default_filename": "addmol.bgf",
        "spec_type": AddmolGeneratorSpec,
        "generate": generate_addmol_template,
        "write": write_addmol,
    }
}
