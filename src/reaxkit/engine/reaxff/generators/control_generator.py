"""
ReaxFF control file generation utilities.

This module provides deterministic helpers for generating or writing a
default ReaxFF ``control`` input file from a canonical, aligned template.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

from reaxkit.domain.data_models import ControlParametersData

CONTROL_VALUE_WIDTH = 7
CONTROL_KEY_WIDTH = 10

CONTROL_TEMPLATE = """# General parameters
      1 iexx
      1 iexy
      1 iexz
    7.5 vlbora
      1 icobo      0: use uncorrected bond orders for mol.nrs. in xmolout, fort.7 and fort.71 1: use corrected bond orders
      0 itrout     1: create diff_traj.xyz-output file with unfolded coordinates
      1 itrans     0: do not back-translate atoms  1: back translate atoms
      1 icentr     0: keep position   1: put centre of mass in centre periodic cell  2: put centre of mass at origin
      0 imetho     0: Normal MD-run 1: Energy minimisation 2:MD-energy minimisation
      1 igeofo     0:xyz-input geometry 1: Biograf input geometry 2: xmol-input geometry
 80.000 axis1      a (for non-periodical systems)
 80.000 axis2      b (for non-periodical systems)
 80.000 axis3      c (for non-periodical systems)
 0.0001 cutof2     BO-cutoff for valency angles and torsion angles
  0.300 cutof3     BO-cutoff for bond order for graphs
      7 icharg     Charges. 1:EEM 2:- 3: Shielded EEM 4: Full system EEM 5: Fixed (unit 26) 6: Fragment EEM
      1 ichaen     Charges. 1:include charge energy 0: Do not include charge energy
      1 iappen     1: Append fort.7 and fort.8
      0 isurpr     1: Surpress lots of output 2: Read in all geometries at the same time
     25 irecon     Frequency of reading control-file
      0 icheck     0: Normal run 1:Check first derivatives;2: Single run
      0 idebug     0: normal run 1: debug run
      3 ixmolo     0: only x,y,z-coordinates in xmolout  1: x,y,z + velocities + molnr. in xmolout 2:x,y,z+mol.nr.  3: x,y,z+mol.nr.+Estrain
10.0000 volcha     volume change (%) with 'S' and 'B' labels
      0 iconne     0: Normal run 1: Run with fixed connection table 2: Read in from cnt.in
      0 imolde     0: Normal run 1: Run with fixed molecule definition (moldef.in)
# MD-parameters
      1 imdmet     MD-method. 1:NVT/Berendsen thermostat 2:do not use;3:NVE 4: NPT/Berendsen thermo/barostat
  0.250 tstep      MD-time step (fs)
0500.00 mdtemp     MD-temperature
0500.00 mdtem2     2nd MD-temperature
0000.00 tincr      Increase/decrease temperature
      2 itdmet     0: T-damp atoms 1: Energy cons 2:System 3: Mols 4: Anderson 5: Mols+2 types of damping
  100.0 tdamp1     1st Berendsen/Anderson temperature damping constant (fs)
  100.0 tdamp2     2nd Berendsen/Anderson temperature damping constant (fs)
      0 ntdamp     Nr. of atoms with 1st damping constant and 1st MD-temperature
0000.00 mdpres     MD-pressure (MPa)
05000.0 pdamp1     Berendsen pressure damping constant (fs)
      0 inpt       0: Change all cell parameters in NPT-run  1: fixed x 2: fixed y 3: fixed z
0155000 nmdit      Number of MD-iterations
 000000 nmdeqi     Number of MD-equilibrium iterations
  00001 ichupd     Charge update frequency
    025 iout1      Output to unit 71 and unit 73
    100 iout2      Save coordinates
      0 ivels      1:Set vels and accels from moldyn.vel to zero
  00025 itrafr     Frequency of trarot-calls
      1 iout3      0: create moldyn.xxxx-files 1: do not create moldyn.xxxx-files
      1 iravel     1: Random initial velocities
0.00001 endmd      End point criterium for MD energy minimisation
 025000 iout6      Save velocity file
 000025 irten      Frequency of removal of rotational and translational energy
      0 npreit     Nr. of iterations in previous runs
   0.00 range      Range for back-translation of atoms
# MM-parameters
1.00000 endmm      End point criterium for MM energy minimisation
 -00001 imaxmo     <0 MD-based energy minimization >0 Steepest descent maximum movement (1/1D6 A) 0: Conjugate gradient
  00000 imaxit     Maximum number of iterations
    005 iout4      Frequency of structure output during minimisation
      0 iout5      1:Remove fort.57 and fort.58 files
1.00250 celopt     Cell parameter change
      0 icelo2     Change all cell parameters (0) or only x/y/z axis (1/2/3)
# FF-optimisation parameters
   0.25 parsca     Parameter optimization: parameter step scaling
 0.0100 parext     Parameter optimization: extrapolation
      0 icelop     0: No cell parameter optimisation 1:Cell parameter optimisation
      1 igeopt     0: Always use same start gemetries 1:Use latest geometries in optimisation
      0 iincop     heat increment optimisation 1: yes 0: no
25.0000 accerr     Accepted increase in error force field
    251 nmoset     Nr. of molecules in training set
#Outdated parameters
      0 ideve1     0: Normal run 1:Check for radical/double bond distances
   2000 ideve2     Frequency of radical/double bond check
      0 nreac      0: reactive; 1: non-reactive; 2: Place default atoms
      1 ibiola     0: output *.geo and *.bgf-files 1: surpress *.geo and *.bgf output files
100.000 tdhoov     Hoover-Nose temperature damping constant (fs)
 01.000 achoov     100*Accuracy Hoover-Nose
      0 itfix      1:Keep temperature fixed at exactly tset
"""


__all__ = [
    "ControlGeneratorSpec",
    "DEFAULT_CONTROL_SPEC",
    "CONTROL_GENERATOR_REGISTRY",
    "generate_control_template",
    "write_control_from_data",
    "write_control",
    "write_control_template",
    "write_control_template_with_overrides",
]


@dataclass(frozen=True)
class ControlGeneratorSpec:
    """
    Declarative settings for generating a ReaxFF ``control`` file.

    Parameters
    ----------
    template_text : str, optional
        Fully formatted ``control`` file content. Defaults to the bundled
        canonical template.
    """

    template_text: str = CONTROL_TEMPLATE


DEFAULT_CONTROL_SPEC = ControlGeneratorSpec()


def generate_control_template(spec: ControlGeneratorSpec = DEFAULT_CONTROL_SPEC) -> str:
    """
    Generate the default ReaxFF ``control`` file content as text.

    Parameters
    ----------
    spec : ControlGeneratorSpec, optional
        Control generation settings.

    Returns
    -------
    str
        The fully formatted ``control`` file content.
    """
    return spec.template_text


def _format_control_value(value: Any) -> str:
    text = str(value).strip()
    if len(text) > CONTROL_VALUE_WIDTH:
        raise ValueError(
            f"Value {text!r} exceeds canonical control width {CONTROL_VALUE_WIDTH}."
        )
    return text.rjust(CONTROL_VALUE_WIDTH)


def _format_control_line(value: Any, key: str, comment_tail: str) -> str:
    key_text = str(key).strip()
    if len(key_text) > CONTROL_KEY_WIDTH:
        raise ValueError(
            f"Control key {key_text!r} exceeds canonical width {CONTROL_KEY_WIDTH}."
        )
    key_col = key_text.ljust(CONTROL_KEY_WIDTH)
    comment = str(comment_tail).strip()
    if comment:
        return f"{_format_control_value(value)} {key_col} {comment}\n"
    return f"{_format_control_value(value)} {key_col}\n"


def _apply_control_overrides(text: str, overrides: dict[str, Any] | None = None) -> str:
    normalized = {str(k).strip().lower(): v for k, v in (overrides or {}).items() if str(k).strip()}
    out_lines: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(r"^\s*([\d\-.Ee+]+)\s+([A-Za-z_]\w*)\s*(.*)$")
    for line in text.splitlines(keepends=True):
        match = pattern.match(line)
        if not match:
            out_lines.append(line)
            continue
        old_value, key, tail = match.groups()
        lookup = key.lower()
        value = normalized.get(lookup, old_value)
        if lookup in normalized:
            seen.add(lookup)
        out_lines.append(_format_control_line(value=value, key=key, comment_tail=tail))

    missing = sorted(k for k in normalized.keys() if k not in seen)
    if missing:
        raise KeyError(f"Unknown control key(s): {', '.join(missing)}")
    return "".join(out_lines)


def _normalize_overrides(overrides: dict[str, Any] | None = None) -> dict[str, str]:
    return {str(k).strip().lower(): str(v) for k, v in (overrides or {}).items() if str(k).strip()}


def _control_overrides_from_data(data: ControlParametersData) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for section_name in ("general", "md", "mm", "ff", "outdated"):
        section = getattr(data, section_name, {}) or {}
        if not isinstance(section, dict):
            raise TypeError(f"ControlParametersData.{section_name} must be a dict.")
        for key, value in section.items():
            key_norm = str(key).strip().lower()
            if key_norm:
                overrides[key_norm] = value
    return overrides


def write_control_template(
    out_path: str | Path = "control",
    spec: ControlGeneratorSpec = DEFAULT_CONTROL_SPEC,
) -> Path:
    """
    Write the default template ``control`` file with no modifications.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(generate_control_template(spec), encoding="utf-8")
    return out_path


def write_control_template_with_overrides(
    out_path: str | Path = "control",
    spec: ControlGeneratorSpec = DEFAULT_CONTROL_SPEC,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """
    Write the template ``control`` file with key/value overrides.

    Parameters
    ----------
    out_path : str | pathlib.Path, optional
        Output file path to write. Parent directories are created when needed.
    spec : ControlGeneratorSpec, optional
        Control generation settings.

    Returns
    -------
    pathlib.Path
        The path of the written control file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = _apply_control_overrides(generate_control_template(spec), overrides=_normalize_overrides(overrides))
    out_path.write_text(rendered, encoding="utf-8")
    return out_path


def write_control_from_data(
    data: ControlParametersData,
    out_path: str | Path = "control",
    spec: ControlGeneratorSpec = DEFAULT_CONTROL_SPEC,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """
    Write a ``control`` file from ``ControlParametersData`` and optional overrides.
    """
    merged = _normalize_overrides(_control_overrides_from_data(data))
    merged.update(_normalize_overrides(overrides))
    return write_control_template_with_overrides(
        out_path=out_path,
        spec=spec,
        overrides=merged,
    )


def write_control(
    out_path: str | Path = "control",
    spec: ControlGeneratorSpec = DEFAULT_CONTROL_SPEC,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """
    Backward-compatible wrapper for writing template control files with overrides.
    """
    return write_control_template_with_overrides(
        out_path=out_path,
        spec=spec,
        overrides=overrides,
    )


CONTROL_GENERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    "control": {
        "label": "ReaxFF Control File",
        "default_filename": "control",
        "spec_type": ControlGeneratorSpec,
        "generate": generate_control_template,
        "write": write_control_template_with_overrides,
    }
}
