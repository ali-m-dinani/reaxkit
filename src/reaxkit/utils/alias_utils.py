"""Alias resolution utilities for tolerant column/key matching across ReaxKit files."""
from __future__ import annotations
from typing import Dict, List, Iterable, Optional

# Canonical → aliases. Keep both generic (for 1-zone) and enumerated (for multi-zone).
_DEFAULT_ALIAS_MAP: Dict[str, List[str]] = {
    # Common summary aliases
    "iter": ["iteration", "Iter", "Iter.", "Iteration", "#start", "start"],
    "E_pot": ["Epot(kcal)", "Epot(kcal/mol)", "E_potential"], #used in xmolout and summary.txt
    "frame": ['frm'],
    "time": ["Time(fs)", "Time"], #also in summary.txt
    "num_of_atoms": ['num_atoms','number_of_atoms','count_of_atoms'], #used in xmolout and fort.7
    "V": ["Vol(A^3)", "Volume", "volume"], #summary.txt and fort.74
    "D": ["Dens(kg/dm3)", "Density", "density"], #summary.txt and fort.74

    # molfra.out alias
    "freq": ["frequency", "count"],
    "molecular_formula": ["mol_formula", "molecule_formula", "molecule"],
    "molecular_mass": ["mol_mass", "mass"],
    "total_molecules": ["tot_mol"],
    "total_atoms": ["tot_atom"],
    "total_molecular_mass": ["tot_mol_mass"],

    # fort.78 alias
    "field_x":   ["Ex", "Efield_x", "Field_x", "fieldX"],
    "field_y":   ["Ey", "Efield_y", "Field_y", "fieldY"],
    "field_z":   ["Ez", "Efield_z", "Field_z", "fieldZ"],
    "E_field_x": ["Efx"],
    "E_field_y": ["Efy"],
    "E_field_z": ["Efz"],
    "E_field":   ["Ef"],

    # summary.txt alias
    "nmol": ["Nmol"],
    "T": ["T(K)", "Temp", "temp"],
    "P": ["Pres(MPa)", "Pressure", "pressure"],
    "elap_time": ["Elap", "time_elapsed", "elapsed_time"],

    # fort.13 alias
    "total_ff_error": ["tot_err", "err_tot"],

    # eregime.in (generic + enumerated)
    "field_zones": ["#V", "V"],
    "field_dir": ["direction", "dir", "direction1"],
    "field": ["E", "Magnitude(V/A)", "Magnitude1(V/A)", "E1"],
    "field_dir1": ["direction1"],
    "field1": ["Magnitude1(V/A)", "E1"],
    "field_dir2": ["direction2"],
    "field2": ["Magnitude2(V/A)", "E2"],
    "field_dir3": ["direction3"],
    "field3": ["Magnitude3(V/A)", "E3"],

    # xmolout alias
    "atom_type": ['type_of_atom', 'atm_type'],
    'x': ['x_coordinate', 'x_coord', 'coord_x', 'coordinate_x'],
    'y': ['y_coordinate', 'y_coord', 'coord_y', 'coordinate_y'],
    'z': ['z_coordinate', 'z_coord', 'coord_z', 'coordinate_z'],

    # fort.7 alias
    "molecule_num": ['molecular_number', 'molecular_num'],
    "partial_charge": ['charge', 'q'],

    #fort.99 alias
    "error": ["Err", "Error"],

    #electrostatics
    "mu_x (debye)": ["mu_x", "dipole_x", "dipole_moment_x"],
    "mu_y (debye)": ["mu_y", "dipole_y", "dipole_moment_y"],
    "mu_z (debye)": ["mu_z", "dipole_z", "dipole_moment_z"],
    "P_x (uC/cm^2)": ["pol_x", "polarization_x"],
    "P_y (uC/cm^2)": ["pol_y", "polarization_y"],
    "P_z (uC/cm^2)": ["pol_z", "polarization_z"],
}

def resolve_alias_from_columns(
    cols: Iterable[str],
    canonical: str,
    aliases: Optional[Dict[str, List[str]]] = None
) -> Optional[str]:
    """
    Return the actual column present in 'cols' that matches 'canonical' or any of its aliases.
    Case-insensitive, with forgiving heuristics.
    """
    if cols is None:
        return None

    orig_cols = list(cols)
    lower_map = {c.lower(): c for c in orig_cols}
    aliases = aliases or _DEFAULT_ALIAS_MAP

    candidates = [canonical]
    if canonical in aliases:
        candidates.extend(aliases[canonical])

    # Exact (case-insensitive)
    for cand in candidates:
        hit = lower_map.get(cand.lower())
        if hit is not None:
            return hit

    # Heuristics on canonical (startswith/contains)
    cname = canonical.lower()
    for c in orig_cols:
        cl = c.lower()
        if cl.startswith(cname) or cname in cl:
            return c

    return None

def _resolve_alias(source, canonical: str) -> str:
    """
    Backwards-compatible resolver:
      - If 'source' has .dataframe(), use its columns.
      - Else if 'source' is a DataFrame, use its columns.
      - Else if it's an iterable of column names, use that.
    Raises KeyError if nothing resolves.
    """
    try:
        cols = list(source.dataframe().columns)  # type: ignore[attr-defined]
    except Exception:
        try:
            cols = list(getattr(source, "columns"))
        except Exception:
            cols = list(source)  # assume iterable of str

    hit = resolve_alias_from_columns(cols, canonical, _DEFAULT_ALIAS_MAP)
    if hit is None:
        raise KeyError(
            f"Could not resolve alias '{canonical}'. "
            f"Available columns: {list(cols)}"
        )
    return hit

def available_keys_from_columns(cols: Iterable[str]) -> List[str]:
    """
    Return all usable keys: raw columns + any aliases that resolve.
    """
    cols_set = set(cols)
    keys = set(cols_set)
    for alias, cands in _DEFAULT_ALIAS_MAP.items():
        if any(c in cols_set for c in cands) or alias in cols_set:
            keys.add(alias)
    return sorted(keys)

# Re-export for callers that already import these names
available_keys = available_keys_from_columns

def normalize_choice(value: str, domain: str = "xaxis") -> str:
    """
    Normalize user-provided keywords (e.g., CLI flags like --xaxis iter)
    using the _DEFAULT_ALIAS_MAP definitions.

    Example:
        normalize_choice("iter") -> "iter"
        normalize_choice("frm")  -> "frame"
        normalize_choice("Time")    -> "time"
    """
    v = (value or "").strip().lower()
    if not v:
        return v

    for canonical, aliases in _DEFAULT_ALIAS_MAP.items():
        all_names = [canonical.lower()] + [a.lower() for a in aliases]
        if v in all_names:
            return canonical

    return v
