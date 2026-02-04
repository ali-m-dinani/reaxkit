"""
Trainset generation utilities for ReaxFF parameter training.

This module provides end-to-end helpers for generating elastic-energy
training targets (bulk EOS and elastic constants), optional strained
geometries, YAML-based configuration, and Materials Project-based
bootstrapping for trainset creation.

Typical use cases include:

- generating ``trainset_elastic.in`` plus energy tables (E vs strain/volume)
- generating strained XYZ/GEO structures for ReaxFF runs
- writing/reading a ``trainset_elastic.yaml`` settings file
- creating a ready-to-run trainset from a Materials Project material ID
"""


from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple, Optional, Any, Literal
from mp_api.client import MPRester
import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell, cell_to_cellpar
import shutil
from pathlib import Path

from reaxkit.io.generators.geo_generator import read_structure, write_structure, xtob
from reaxkit.utils.equation_of_states import vinet_energy_trainset
from reaxkit.utils.constants import const

# =============================================================================
# OVERVIEW OF THE THE CODE
# =============================================================================

"""
This is a refactor/translation of the Fortran program (elastic_energy_v2) developed by Y. Shin.

This code is used to generate trainset data, and is structured into 4 parts:

1. ELASTIC_ENERGY SECTION:
    which is used to obtain energy vs volume for orthogonal systems. That's because it treats
    strain as independent scalar components instead of a full strain tensor,
    which breaks down for non-orthogonal cells.
    This part yields to sorts of data:
    - Bulk modulus: Energy vs Volume using an EOS (Vinet)
    - Elastic constants (c11..c66): Energy vs strain using quadratic strain-energy forms

    there is a top-level function here, generate_all_energy_vs_volume_data, which calls
    two other functions generate_bulk_data and generate_elastic_data to compute energy
    vs volume change according to the explanation above.

2. ELASTIC_GEO SECTION:
    which is used to obtain expanded or compressed geometries in xyz and bgf format for any
    crystal structure (i.e., not limited to orthognal systems), using the main function
    generate_strained_geometries_with_xtob, which calls other related functions.

3. YAML file management for settings of trainset
    this part is used to write (using write_trainset_settings_yaml fucntion) or read
    (using read_trainset_settings_yaml) a settings file  with .yaml format. This file determines
    cell dimensions, bulk modulus, and any other required settings for getting
    (using generate_trainset_from_yaml function) elastic energy or expanded/compressed geo files.

4. MP API Handler:
    which is used to get crystal structure, cell dimension and angles, and mechanical properties
    directly from material's project website, and generate the corresponding trainset.
    The main function here is generate_trainset_settings_yaml_from_mp_simple which:
        1. makes the connection to MP website and gets the data
        2. writes an informative yaml file using write_trainset_settings_yaml fucntion
        3. makes two geometry files in .xyz and .cif format of the material
        4. generates training set using Yaml file and xyz file using generate_trainset_from_yaml

"""

# =============================================================================
# 1. ELASTIC_ENERGY SECTION
# =============================================================================

# -----------------------------------------------------------------------------
# Constants (match Fortran code)
# -----------------------------------------------------------------------------

# AVOGADRO_CONSTANT is NA in the original Fortran code
# ENERGY_CONVERSION_FACTOR is factor in Fortran
AVOGADRO_CONSTANT = const("AVOGADRO_CONSTANT")
ENERGY_CONVERSION_FACTOR = 10.0 * 4.184 / AVOGADRO_CONSTANT

# -----------------------------------------------------------------------------
# Small utilities for elastic_energy
# -----------------------------------------------------------------------------

def _fortran_nint(nonnegative_value: float) -> int:
    """
    Round a value to the nearest integer using Fortran-style NINT behavior.

    The Fortran code uses nint(tstrain/dstrain) with tstrain>=0.
    We implement "round half up" for x>=0 (ties are rare for these inputs).

    Works on
    --------
    Numeric values (utility)

    Parameters
    ----------
    nonnegative_value : float
        Value to round.

    Returns
    -------
    int
        Nearest integer (ties rounded away from zero for typical non-negative inputs).

    Examples
    --------
    >>> _fortran_nint(1.2)
    1
    >>> _fortran_nint(1.5)
    2
    """
    # We support negative too, though it won't be used here.
    if nonnegative_value < 0:
        return -_fortran_nint(-nonnegative_value)
    return int(math.floor(nonnegative_value + 0.5))


def _compute_orthogonal_lattice_cell_volume(
    *,
    a_length: float,
    b_length: float,
    c_length: float,
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
) -> float:
    """
    Compute the unit-cell volume from lattice lengths and angles.

    This follows the same trig construction used in the Fortran snippet
    (with cosphi/sinphi). For orthogonal cells, reduces to a*b*c.

    Fortran mapping
    --------------
    - a_length, b_length, c_length correspond to L2(1), L2(2), L2(3) when used for the bulk volume
      (and also for elastic volume in your snippet).
    - alpha_deg, beta_deg, gamma_deg correspond to ang2(1..3).

    Works on
    --------
    Lattice parameters for (typically) orthogonal cells

    Parameters
    ----------
    a_length, b_length, c_length : float
        Cell edge lengths (Å).
    alpha_deg, beta_deg, gamma_deg : float
        Cell angles (degrees).

    Returns
    -------
    float
        Unit-cell volume (Å^3).

    Examples
    --------
    >>> _compute_orthogonal_lattice_cell_volume(
    ...     a_length=2.0, b_length=3.0, c_length=4.0,
    ...     alpha_deg=90.0, beta_deg=90.0, gamma_deg=90.0
    ... )
    24.0
    """
    degrees_to_radians = math.pi / 180.0  # degrees_to_radians is dgrrdn in Fortran (via rdndgr)
    alpha_rad = alpha_deg * degrees_to_radians  # alpha_rad is halfa in Fortran
    beta_rad = beta_deg * degrees_to_radians    # beta_rad is hbeta in Fortran
    gamma_rad = gamma_deg * degrees_to_radians  # gamma_rad is hgamma in Fortran

    sin_alpha = math.sin(alpha_rad)  # sin_alpha is sinalf in Fortran
    cos_alpha = math.cos(alpha_rad)  # cos_alpha is cosalf in Fortran
    sin_beta = math.sin(beta_rad)    # sin_beta is sinbet in Fortran
    cos_beta = math.cos(beta_rad)    # cos_beta is cosbet in Fortran

    denominator = sin_alpha * sin_beta  # denominator is sinalf*sinbet in Fortran
    if abs(denominator) < 1e-16:
        raise ValueError("Invalid cell angles: sin(alpha)*sin(beta) ~ 0, cannot compute volume.")

    cos_intermediate_angle = (math.cos(gamma_rad) - cos_alpha * cos_beta) / denominator
    # cos_intermediate_angle is cosphi in the original Fortran code
    # Fortran clamps only > 1.0; we clamp to [-1,1] for numerical safety.
    cos_intermediate_angle = max(-1.0, min(1.0, cos_intermediate_angle))

    sin_intermediate_angle = math.sqrt(max(0.0, 1.0 - cos_intermediate_angle * cos_intermediate_angle))
    # sin_intermediate_angle is sinphi in the original Fortran code

    # Build the same determinant components as the Fortran snippet.
    basis_x_component = a_length * sin_beta * sin_intermediate_angle
    # basis_x_component is tm11 in the original Fortran code

    basis_y_component = b_length * sin_alpha
    # basis_y_component is tm22 in the original Fortran code

    basis_z_component = c_length
    # basis_z_component is tm33 in the original Fortran code

    reference_volume = basis_x_component * basis_y_component * basis_z_component
    # reference_volume is vol in the original Fortran code

    return reference_volume


def _build_symmetric_grid(
    *,
    max_abs_value: float,
    step: float,
    grid_mode: str,
) -> List[float]:
    """
    Build a symmetric grid spanning [-max_abs_value, +max_abs_value].

    Fortran mapping
    --------------
    - max_abs_value corresponds to:
        * tstrain2 in bulk block
        * tstrain in elastic blocks
    - step corresponds to:
        * dstrain=0.004 in bulk block
        * dstrain=0.005 in elastic blocks
    - num_strain_steps corresponds to nstrain in Fortran

    Works on
    --------
    Strain/parameter grids for bulk and elastic target generation

    Parameters
    ----------
    max_abs_value : float
        Maximum absolute value to include.
    step : float
        Grid spacing (> 0).
    grid_mode : {"bulk","elastic"}
        Grid logic mode matching the generator block.

    Returns
    -------
    list[float]
        Symmetric grid values including 0.

    Examples
    --------
    >>> _build_symmetric_grid(max_abs_value=0.01, step=0.005, grid_mode="elastic")
    [-0.01, -0.005, 0.0, 0.005, 0.01]
    """
    if step <= 0:
        raise ValueError("step must be positive.")

    rounded_ratio = _fortran_nint(max_abs_value / step)  # rounded_ratio is t in Fortran

    # For both bulk and elastic blocks in your snippet, logic effectively becomes:
    # if rounded_ratio*step < max_abs_value -> nstrain = rounded_ratio + 1
    # else nstrain = rounded_ratio
    # (bulk block used .ge, elastic block used .eq but falls through)
    if grid_mode not in ("bulk", "elastic"):
        raise ValueError("grid_mode must be 'bulk' or 'elastic'.")

    num_strain_steps = rounded_ratio + 1 if (rounded_ratio * step < max_abs_value) else rounded_ratio
    # num_strain_steps is nstrain in the original Fortran code

    return [step * n for n in range(-num_strain_steps, num_strain_steps + 1)]


def _make_label(prefix: str, signed_index: int) -> str:
    """
    Construct a trainset label for a signed strain index.

    Construct labels like Fortran:
      bulk_c0001, bulk_e0001, bulk_0
      c11_c0001, c11_e0001, c11_0

    Works on
    --------
    Trainset naming for bulk/elastic targets

    Parameters
    ----------
    prefix : str
        Label prefix (e.g., "bulk", "c11").
    signed_index : int
        Signed index (negative = compression, positive = expansion, 0 = reference).

    Returns
    -------
    str
        Label string (e.g., ``"bulk_c0001"``, ``"c11_e0002"``, ``"c11_0"``).

    Examples
    --------
    >>> _make_label("bulk", -3)
    'bulk_c0003'
    >>> _make_label("c11", 0)
    'c11_0'
    """
    if signed_index == 0:
        return f"{prefix}_0"
    compression_or_expansion = "c" if signed_index < 0 else "e"
    return f"{prefix}_{compression_or_expansion}{abs(signed_index):04d}"


def _index_from_grid_value(grid_value: float, step: float) -> int:
    """
    Convert a grid value to its signed integer index for labeling.

    Convert a grid value back to its signed integer index (used for label naming).
    Fortran uses eps = step*n exactly, so this is safe; we still round for robustness.

    Works on
    --------
    Grid-to-index conversion for trainset label naming

    Parameters
    ----------
    grid_value : float
        Grid value (typically a strain).
    step : float
        Grid spacing.

    Returns
    -------
    int
        Signed integer index corresponding to ``grid_value / step``.

    Examples
    --------
    >>> _index_from_grid_value(-0.01, 0.005)
    -2
    """
    if step == 0:
        return 0
    return int(round(grid_value / step))


# -----------------------------------------------------------------------------
# Public API: generators for elastic_energy
# -----------------------------------------------------------------------------

def _generate_bulk_data(
    *,
    bulk_modulus_gpa: float,
    bulk_modulus_pressure_derivative: float,
    max_volumetric_strain_percent: float,
    cell: Dict[str, float],
    linear_strain_step: float = 0.004,
    reference_energy: float = 0.0,
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """
    Generate bulk EOS energy-vs-volume targets for trainset fitting.

    Works on
    --------
    Elastic-energy training targets (bulk EOS), driven by lattice parameters

    Parameters
    ----------
    bulk_modulus_gpa : float
        Bulk modulus B0 (GPa).
    bulk_modulus_pressure_derivative : float
        Pressure derivative B0' (dimensionless).
    max_volumetric_strain_percent : float
        Maximum volumetric strain magnitude (%).
    cell : dict
        Reference cell parameters with keys: ``a``, ``b``, ``c``, ``alpha``, ``beta``, ``gamma``.
    linear_strain_step : float, optional
        Sampling step for the equivalent linear strain grid.
    reference_energy : float, optional
        Reference energy offset E0.

    Returns
    -------
    tuple[list[tuple[float, float]], list[str]]
        (1) Table rows of ``(volume, energy)`` for writing ``EvsStrain_bulk.dat``.
        (2) Trainset ENERGY-block lines for inserting into ``trainset_elastic.in``.

    Examples
    --------
    >>> cell = {"a": 2.9, "b": 2.9, "c": 3.5, "alpha": 90, "beta": 90, "gamma": 90}
    >>> table, lines = _generate_bulk_data(
    ...     bulk_modulus_gpa=180.0,
    ...     bulk_modulus_pressure_derivative=4.0,
    ...     max_volumetric_strain_percent=6.0,
    ...     cell=cell,
    ... )
    """
    reference_volume = _compute_orthogonal_lattice_cell_volume(
        a_length=cell["a"], b_length=cell["b"], c_length=cell["c"],
        alpha_deg=cell["alpha"], beta_deg=cell["beta"], gamma_deg=cell["gamma"]
    )
    # reference_volume is vol in the original Fortran code (computed from L2/ang2)

    equivalent_linear_strain_max = (1.0 + max_volumetric_strain_percent / 100.0) ** (1.0 / 3.0) - 1.0
    # equivalent_linear_strain_max is tstrain2 in the original Fortran code

    linear_strain_grid = _build_symmetric_grid(
        max_abs_value=equivalent_linear_strain_max,
        step=linear_strain_step,
        grid_mode="bulk",
    )
    # linear_strain_grid corresponds to the n loop from -nstrain..+nstrain in bulk block

    bulk_table: List[Tuple[float, float]] = []
    trainset_lines: List[str] = []

    for linear_strain in linear_strain_grid:
        signed_step_index = _index_from_grid_value(linear_strain, linear_strain_step)
        # signed_step_index is n in the original Fortran code (bulk block)

        strained_volume = reference_volume * (1.0 + linear_strain) ** 3
        # strained_volume is strain0 in the original Fortran code (bulk block)

        eos_energy = vinet_energy_trainset(
            volume=strained_volume,
            reference_volume=reference_volume,
            bulk_modulus_gpa=bulk_modulus_gpa,
            bulk_modulus_pressure_derivative=bulk_modulus_pressure_derivative,
            reference_energy=reference_energy,
            energy_conversion_factor=ENERGY_CONVERSION_FACTOR,
        )
        # eos_energy is db in the original Fortran code

        if eos_energy == 0.0:
            eos_energy = 1e-4  # eos_energy is db safeguard in Fortran

        label = _make_label("bulk", signed_step_index)  # label is title0 in Fortran

        # Match the semantic structure of the Fortran trainset line:
        # ' 1.0   +   ' title0 ' /1  -  ' 'bulk_0 /1' '          ' db
        label_field_width = 11  # enough for 'bulk_c0005' and also pads 'bulk_0'
        trainset_line = f" 1.0   +   {label:<{label_field_width}} /1  -  bulk_0 /1          {eos_energy:12.4f}"

        bulk_table.append((strained_volume, eos_energy))
        trainset_lines.append(trainset_line)

    return bulk_table, trainset_lines


def _generate_elastic_data(
    *,
    elastic_constants_gpa: Dict[str, float],
    max_strain_percent: float,
    volume_reference_cell: Dict[str, float],
    strain_step: float = 0.005,
) -> Dict[str, Tuple[List[Tuple[float, float]], List[str]]]:
    """
    Generate elastic-constant energy-vs-strain targets for trainset fitting.

    Works on
    --------
    Elastic-energy training targets (c11..c66), driven by lattice parameters

    Parameters
    ----------
    elastic_constants_gpa : dict
        Elastic constants in GPa with keys:
        ``c11,c22,c33,c12,c13,c23,c44,c55,c66``.
    max_strain_percent : float
        Maximum linear strain magnitude (%).
    volume_reference_cell : dict
        Reference cell parameters with keys: ``a``, ``b``, ``c``, ``alpha``, ``beta``, ``gamma``.
    strain_step : float, optional
        Linear strain step size (unitless).

    Returns
    -------
    dict[str, tuple[list[tuple[float, float]], list[str]]]
        Mapping ``mode -> (table_rows, trainset_lines)``, where:
        - table_rows: ``[(strain, energy), ...]``
        - trainset_lines: ENERGY-block lines for ``trainset_elastic.in``

    Examples
    --------
    >>> cij = {"c11": 300, "c22": 300, "c33": 250, "c12": 120, "c13": 140,
    ...        "c23": 140, "c44": 80, "c55": 80, "c66": 60}
    >>> cell = {"a": 2.9, "b": 2.9, "c": 3.5, "alpha": 90, "beta": 90, "gamma": 90}
    >>> out = _generate_elastic_data(
    ...     elastic_constants_gpa=cij,
    ...     max_strain_percent=3.0,
    ...     volume_reference_cell=cell,
    ... )
    >>> sorted(out.keys())[:3]
    ['c11', 'c12', 'c13']
    """
    reference_volume = _compute_orthogonal_lattice_cell_volume(
        a_length=volume_reference_cell["a"], b_length=volume_reference_cell["b"], c_length=volume_reference_cell["c"],
        alpha_deg=volume_reference_cell["alpha"], beta_deg=volume_reference_cell["beta"], gamma_deg=volume_reference_cell["gamma"]
    )
    # reference_volume is vol in the original Fortran code (used in elastic energy prefactors)

    max_linear_strain = max_strain_percent / 100.0  # max_linear_strain is tstrain in Fortran
    linear_strain_grid = _build_symmetric_grid(
        max_abs_value=max_linear_strain,
        step=strain_step,
        grid_mode="elastic",
    )
    # linear_strain_grid corresponds to n loop -nstrain..+nstrain in each elastic block

    c = elastic_constants_gpa  # c is cii(...) conceptually in Fortran, but named dict here

    # Coefficient definitions replicate Fortran formulas (a*eps^2 with b=c=0):
    def normal_strain_prefactor(cij: float) -> float:
        return cij * reference_volume / (2.0 * ENERGY_CONVERSION_FACTOR)
        # normal_strain_prefactor corresponds to a11/a22/a33 in Fortran (via a=cii*vol/(2*factor))

    def shear_strain_prefactor(cij: float) -> float:
        return 2.0 * cij * reference_volume / ENERGY_CONVERSION_FACTOR
        # shear_strain_prefactor corresponds to a44/a55/a66 in Fortran (via a=2*cii*vol/factor)

    def coupling_strain_prefactor(cij: float, cii: float, cjj: float) -> float:
        return (-cij + (cii + cjj) / 2.0) * reference_volume / ENERGY_CONVERSION_FACTOR
        # coupling_strain_prefactor corresponds to a12/a13/a23 in Fortran

    energy_quadratic_coefficients = {
        "c11": normal_strain_prefactor(c["c11"]),
        "c22": normal_strain_prefactor(c["c22"]),
        "c33": normal_strain_prefactor(c["c33"]),
        "c12": coupling_strain_prefactor(c["c12"], c["c11"], c["c22"]),
        "c13": coupling_strain_prefactor(c["c13"], c["c11"], c["c33"]),
        "c23": coupling_strain_prefactor(c["c23"], c["c22"], c["c33"]),
        "c44": shear_strain_prefactor(c["c44"]),
        "c55": shear_strain_prefactor(c["c55"]),
        "c66": shear_strain_prefactor(c["c66"]),
    }
    # energy_quadratic_coefficients corresponds to a11,a22,... etc in Fortran

    result: Dict[str, Tuple[List[Tuple[float, float]], List[str]]] = {}

    for mode_name, quadratic_prefactor in energy_quadratic_coefficients.items():
        # quadratic_prefactor is 'a' (a11/a22/...) in the original Fortran code
        table_rows: List[Tuple[float, float]] = []
        trainset_lines: List[str] = []

        for linear_strain in linear_strain_grid:
            signed_step_index = _index_from_grid_value(linear_strain, strain_step)
            # signed_step_index is n in the original Fortran elastic blocks

            energy = quadratic_prefactor * (linear_strain ** 2)
            # energy corresponds to d11/d22/... in Fortran (d = a*eps^2 + b*eps + c; b=c=0)

            if energy == 0.0:
                energy = 1e-4  # energy safeguard matches Fortran's dxx=0 -> 0.0001

            label = _make_label(mode_name, signed_step_index)  # label is title0 in Fortran

            label_field_width = 12  # or 14 if you want extra room
            trainset_line = f" 1.0   +   {label:<{label_field_width}} /1  -  {mode_name}_0 /1          {energy:12.4f}"
            # trainset_line matches the Fortran ENERGY line semantic structure

            table_rows.append((linear_strain, energy))
            trainset_lines.append(trainset_line)

        result[mode_name] = (table_rows, trainset_lines)

    return result


def generate_all_energy_vs_volume_data(
    *,
    out_dir: str,
    bulk_inputs: Dict[str, float],
    elastic_inputs: Dict[str, float],
    bulk_cell: Dict[str, float],
    elastic_volume_cell: Optional[Dict[str, float]] = None,
    bulk_options: Optional[Dict[str, float]] = None,
    elastic_options: Optional[Dict[str, float]] = None,
    trainset_filename: str = "trainset_elastic.in",
) -> None:
    """
    Write bulk and elastic energy targets to trainset and table files.

    High-level generator that:
      1) generates bulk + elastic energy targets
      2) writes:
           - trainset_elastic.in
           - EvsStrain_bulk.dat
           - EvsStrain_c11.dat ... EvsStrain_c66.dat

    This function performs both generation and writing and returns None.

    Works on
    --------
    Elastic-energy training targets written to disk (trainset + tables)

    Parameters
    ----------
    out_dir : str
        Output directory to write files into.
    bulk_inputs : dict
        Bulk target inputs with keys such as ``B0_gpa``, ``B0_prime``,
        and ``max_volumetric_strain_percent``.
    elastic_inputs : dict
        Elastic target inputs including ``max_strain_percent`` and ``cij`` values.
    bulk_cell : dict
        Bulk reference cell with keys: ``a,b,c,alpha,beta,gamma``.
    elastic_volume_cell : dict or None, optional
        Cell used to compute volume prefactors for elastic targets. If None, uses ``bulk_cell``.
    bulk_options : dict or None, optional
        Optional overrides (e.g., ``linear_strain_step``, ``reference_energy``).
    elastic_options : dict or None, optional
        Optional overrides (e.g., ``strain_step``).
    trainset_filename : str, optional
        Output trainset file name (default: ``"trainset_elastic.in"``).

    Returns
    -------
    None
        Writes ``trainset_elastic.in`` and E-vs-strain/volume tables to ``out_dir``.

    Examples
    --------
    >>> generate_all_energy_vs_volume_data(
    ...     out_dir="out",
    ...     bulk_inputs={"B0_gpa": 180, "B0_prime": 4.0, "max_volumetric_strain_percent": 6.0},
    ...     elastic_inputs={"max_strain_percent": 3.0, "c11": 300, "c22": 300, "c33": 250,
    ...                    "c12": 120, "c13": 140, "c23": 140, "c44": 80, "c55": 80, "c66": 60},
    ...     bulk_cell={"a": 2.9, "b": 2.9, "c": 3.5, "alpha": 90, "beta": 90, "gamma": 90},
    ... )
    """
    import os

    bulk_options = bulk_options or {}
    elastic_options = elastic_options or {}
    elastic_volume_cell = elastic_volume_cell or bulk_cell

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Orthogonality warnings
    # -------------------------
    def _warn_if_nonorthogonal(cell: dict, label: str) -> None:
        angles = [cell.get("alpha", 90.0),
                  cell.get("beta", 90.0),
                  cell.get("gamma", 90.0)]
        tol = 1e-6
        if any(abs(a - 90.0) > tol for a in angles):
            print(
                f"⚠️  WARNING: {label} cell is non-orthogonal "
                f"(angles = {angles}).\n"
                "    Elastic energy targets assume an orthogonal lattice.\n"
                "    Geometry generation is correct, but elastic energies may be inconsistent.\n"
            )

    _warn_if_nonorthogonal(elastic_volume_cell, label="Elastic")
    _warn_if_nonorthogonal(bulk_cell, label="Bulk")

    # -------------------------
    # Bulk targets
    # -------------------------
    bulk_table, bulk_trainset_lines = _generate_bulk_data(
        bulk_modulus_gpa=bulk_inputs["B0_gpa"],
        bulk_modulus_pressure_derivative=bulk_inputs["B0_prime"],
        max_volumetric_strain_percent=bulk_inputs["max_volumetric_strain_percent"],
        cell=bulk_cell,
        linear_strain_step=float(bulk_options.get("linear_strain_step", 0.004)),
        reference_energy=float(bulk_options.get("reference_energy", 0.0)),
    )

    # -------------------------
    # Elastic targets
    # -------------------------
    elastic_constants = {
        k: elastic_inputs[k]
        for k in ("c11", "c22", "c33", "c12", "c13", "c23", "c44", "c55", "c66")
    }

    elastic_targets = _generate_elastic_data(
        elastic_constants_gpa=elastic_constants,
        max_strain_percent=elastic_inputs["max_strain_percent"],
        volume_reference_cell=elastic_volume_cell,
        strain_step=float(elastic_options.get("strain_step", 0.005)),
    )

    # -------------------------
    # Write trainset file
    # -------------------------
    mode_order = ["c11", "c22", "c33", "c12", "c13", "c23", "c44", "c55", "c66"]

    trainset_lines: List[str] = [
        "ENERGY",
        "# Volume Bulk_EOS",
        *bulk_trainset_lines,
    ]

    for mode in mode_order:
        trainset_lines.append(f"# Volume {mode.upper()}_EOS")
        trainset_lines.extend(elastic_targets[mode][1])

    trainset_lines.append("ENDENERGY")

    with open(os.path.join(out_dir, trainset_filename), "w", encoding="utf-8") as f:
        f.write("\n".join(trainset_lines) + "\n")

    # -------------------------
    # Write tables
    # -------------------------
    def _write_two_column_table(path: str, header: str, rows):
        with open(path, "w", encoding="utf-8") as f:
            f.write(header.rstrip() + "\n")
            for x, y in rows:
                f.write(f"{x:8.3f}  {y:12.4f}\n")

    _write_two_column_table(
        os.path.join(out_dir, "EvsStrain_bulk.dat"),
        "# Volume   Energy",
        bulk_table,
    )

    for mode in mode_order:
        _write_two_column_table(
            os.path.join(out_dir, f"EvsStrain_{mode}.dat"),
            "# Strain   Energy",
            elastic_targets[mode][0],
        )


# =============================================================================
# 2. ELASTIC_GEO SECTION
# =============================================================================


# -----------------------------
# Strain matrices
# -----------------------------

def _deformation_matrix(mode: str, eps: float) -> np.ndarray:
    """
    Build a 3x3 deformation matrix for a named strain mode.

    Return 3x3 deformation matrix d(mode, eps) similar to elastic_geo Fortran.
    mode: bulk, c11,c22,c33,c12,c13,c23,c44,c55,c66

    Works on
    --------
    Strain-mode deformation matrices for strained-geometry generation

    Parameters
    ----------
    mode : str
        Strain mode (e.g., ``"bulk"``, ``"c11"``, ``"c44"``).
    eps : float
        Strain magnitude (unitless).

    Returns
    -------
    numpy.ndarray
        3x3 deformation matrix.

    Examples
    --------
    >>> D = _deformation_matrix("c11", 0.01)
    >>> D.shape
    (3, 3)
    """
    I = np.eye(3, dtype=float)

    if mode == "bulk":
        return np.diag([1.0 + eps, 1.0 + eps, 1.0 + eps])

    if mode == "c11":
        d = I.copy()
        d[0, 0] = 1.0 + eps
        return d
    if mode == "c22":
        d = I.copy()
        d[1, 1] = 1.0 + eps
        return d
    if mode == "c33":
        d = I.copy()
        d[2, 2] = 1.0 + eps
        return d

    # Coupled modes (Fortran uses u = 1/sqrt(1-eps^2))
    if mode in {"c12", "c13", "c23"}:
        u = 1.0 / np.sqrt(max(1e-30, 1.0 - eps * eps))
        d = I.copy()
        if mode == "c12":
            d[0, 0] = u * (1.0 + eps)
            d[1, 1] = u * (1.0 - eps)
            d[2, 2] = 1.0
        elif mode == "c13":
            d[0, 0] = u * (1.0 + eps)
            d[2, 2] = u * (1.0 - eps)
            d[1, 1] = 1.0
        else:  # c23
            d[1, 1] = u * (1.0 + eps)
            d[2, 2] = u * (1.0 - eps)
            d[0, 0] = 1.0
        return d

    # Shear modes (Fortran uses u = 1/(1-eps^2)^(1/3))
    if mode in {"c44", "c55", "c66"}:
        u = 1.0 / (max(1e-30, 1.0 - eps * eps) ** (1.0 / 3.0))
        d = I.copy()
        if mode == "c44":
            d[1, 2] = eps
            d[2, 1] = eps
        elif mode == "c55":
            d[0, 2] = eps
            d[2, 0] = eps
        else:  # c66
            d[0, 1] = eps
            d[1, 0] = eps
        return u * d

    raise ValueError(f"Unknown mode: {mode!r}")


def _symmetric_strain_grid(max_abs: float, step: float) -> List[float]:
    """
    Build a symmetric strain grid spanning [-max_abs, +max_abs].

    Works on
    --------
    Strain grids for strained-geometry generation

    Parameters
    ----------
    max_abs : float
        Maximum absolute strain magnitude.
    step : float
        Strain step size.

    Returns
    -------
    list[float]
        Symmetric strain grid including 0.

    Examples
    --------
    >>> _symmetric_strain_grid(0.01, 0.005)
    [-0.01, -0.005, 0.0, 0.005, 0.01]
    """
    n = int(np.ceil(max_abs / step))
    grid = [k * step for k in range(-n, n + 1)]
    grid = [x for x in grid if abs(x) <= max_abs + 1e-12]
    if 0.0 not in grid:
        grid.append(0.0)
        grid.sort()
    return grid


def _strain_title(prefix: str, eps: float, idx_abs: int) -> str:
    """
    Format a canonical title string for a strain state.

    Works on
    --------
    Strain-state naming for strained-geometry outputs

    Parameters
    ----------
    prefix : str
        Mode prefix (e.g., ``"bulk"``, ``"c11"``).
    eps : float
        Strain value (unitless).
    idx_abs : int
        Absolute strain-step index.

    Returns
    -------
    str
        Title string used for output file naming.

    Examples
    --------
    >>> _strain_title("bulk", -0.01, idx_abs=2)
    'bulk_c0002'
    """

    if abs(eps) < 1e-15:
        return f"{prefix}_0"
    return f"{prefix}_{'c' if eps < 0 else 'e'}{idx_abs:04d}"


def _make_base_atoms_from_xyz_and_cell(
    xyz_path: str | Path,
    cell: np.ndarray,
) -> Atoms:
    """
    Read XYZ via read_structure(), attach the provided cell, and enable PBC.
    """
    atoms = read_structure(xyz_path, format="xyz")
    atoms.set_cell(cell, scale_atoms=False)
    atoms.set_pbc(True)
    return atoms


def generate_strained_geometries_with_xtob(
    *,
    elastic_xyz: str | Path,
    bulk_xyz: Optional[str | Path],
    elastic_cell: Dict[str, float],  # keys: a,b,c,alpha,beta,gamma
    bulk_cell: Dict[str, float],
    max_strain_elastic: float,        # e.g. 0.02 for ±2%
    dstrain_elastic: float,           # e.g. 0.005
    max_strain_bulk_linear: float,    # linear strain, not volumetric
    dstrain_bulk_linear: float,       # e.g. 0.004
    out_dir: str | Path,
    sort_by: Optional[str] = None,
) -> Dict[str, List[Path]]:
    """
    Generate strained XYZ structures and convert them to GEO via xtob.

    Creates strained XYZ files (with comment=title on line 2) and converts each
    to GEO using xtob().

    Output folders:
      out_dir/xyz_strained/*.xyz
      out_dir/geo_strained/*.bgf

    Works on
    --------
    XYZ input structures + GEO/XTLGRF outputs via ``xtob``

    Parameters
    ----------
    elastic_xyz : str or pathlib.Path
        Base XYZ used for elastic strain modes.
    bulk_xyz : str or pathlib.Path or None
        Optional base XYZ used for bulk mode. If None, reuse ``elastic_xyz``.
    elastic_cell : dict
        Elastic reference cell with keys: ``a,b,c,alpha,beta,gamma``.
    bulk_cell : dict
        Bulk reference cell with keys: ``a,b,c,alpha,beta,gamma``.
    max_strain_elastic : float
        Maximum absolute linear strain for elastic modes (unitless).
    dstrain_elastic : float
        Linear strain step for elastic modes (unitless).
    max_strain_bulk_linear : float
        Maximum absolute linear bulk strain (unitless).
    dstrain_bulk_linear : float
        Linear bulk strain step (unitless).
    out_dir : str or pathlib.Path
        Output directory where ``xyz_strained`` and ``geo_strained`` are created.
    sort_by : str or None, optional
        Sorting key passed to ``xtob`` (e.g., ``"z"``).

    Returns
    -------
    dict[str, list[pathlib.Path]]
        Mapping mode name to written GEO paths (e.g., ``"bulk"``, ``"c11"``).

    Examples
    --------
    >>> cell = {"a": 2.9, "b": 2.9, "c": 3.5, "alpha": 90, "beta": 90, "gamma": 90}
    >>> out = generate_strained_geometries_with_xtob(
    ...     elastic_xyz="ground_elastic.xyz",
    ...     bulk_xyz=None,
    ...     elastic_cell=cell,
    ...     bulk_cell=cell,
    ...     max_strain_elastic=0.02,
    ...     dstrain_elastic=0.005,
    ...     max_strain_bulk_linear=0.01,
    ...     dstrain_bulk_linear=0.004,
    ...     out_dir="out",
    ... )
    """
    out_dir = Path(out_dir)
    xyz_dir = out_dir / "xyz_strained"
    geo_dir = out_dir / "geo_strained"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    geo_dir.mkdir(parents=True, exist_ok=True)

    def idx_abs_from_eps(eps: float, step: float) -> int:
        """Fortran-like abs(n) index where eps = n * step."""
        if abs(eps) < 1e-15:
            return 0
        return abs(int(round(eps / step)))

    cell_e = cellpar_to_cell([
        elastic_cell["a"], elastic_cell["b"], elastic_cell["c"],
        elastic_cell["alpha"], elastic_cell["beta"], elastic_cell["gamma"],
    ])
    cell_b = cellpar_to_cell([
        bulk_cell["a"], bulk_cell["b"], bulk_cell["c"],
        bulk_cell["alpha"], bulk_cell["beta"], bulk_cell["gamma"],
    ])

    # Base atoms for elastic
    base_e = _make_base_atoms_from_xyz_and_cell(elastic_xyz, cell_e)
    frac_e = base_e.get_scaled_positions(wrap=False)

    # Base atoms for bulk (reuse elastic if not provided)
    if bulk_xyz is None:
        base_b = base_e.copy()
        base_b.set_cell(cell_b, scale_atoms=False)
        base_b.set_pbc(True)
    else:
        base_b = _make_base_atoms_from_xyz_and_cell(bulk_xyz, cell_b)
    frac_b = base_b.get_scaled_positions(wrap=False)

    out: Dict[str, List[Path]] = {
        m: [] for m in ["bulk", "c11", "c22", "c33", "c12", "c13", "c23", "c44", "c55", "c66"]
    }

    # ---- Bulk ----
    bulk_grid = _symmetric_strain_grid(max_strain_bulk_linear, dstrain_bulk_linear)
    for j, eps in enumerate(bulk_grid):
        d = _deformation_matrix("bulk", eps)
        new_cell = d @ cell_b
        a, b, c, alpha, beta, gamma = cell_to_cellpar(new_cell)

        idx_abs = idx_abs_from_eps(eps, dstrain_bulk_linear)
        title = _strain_title("bulk", eps, idx_abs=idx_abs)
        xyz_path = xyz_dir / f"{title}.xyz"
        geo_path = geo_dir / f"{title}.bgf"

        atoms = base_b.copy()
        atoms.set_cell(new_cell, scale_atoms=False)
        atoms.set_scaled_positions(frac_b)

        # IMPORTANT: comment goes to line 2 in XYZ
        write_structure(atoms, xyz_path, format="xyz", comment=title)

        xtob(
            xyz_file=xyz_path,
            geo_file=geo_path,
            box_lengths=(float(a), float(b), float(c)),
            box_angles=(float(alpha), float(beta), float(gamma)),
            sort_by=sort_by,
            ascending=True,
        )
        out["bulk"].append(geo_path)

    # ---- Elastic modes ----
    elastic_grid = _symmetric_strain_grid(max_strain_elastic, dstrain_elastic)
    modes = ["c11", "c22", "c33", "c12", "c13", "c23", "c44", "c55", "c66"]
    for mode in modes:
        for j, eps in enumerate(elastic_grid):
            d = _deformation_matrix(mode, eps)
            new_cell = d @ cell_e
            a, b, c, alpha, beta, gamma = cell_to_cellpar(new_cell)

            idx_abs = idx_abs_from_eps(eps, dstrain_elastic)
            title = _strain_title(mode, eps, idx_abs=idx_abs)
            xyz_path = xyz_dir / f"{title}.xyz"
            geo_path = geo_dir / f"{title}.geo"

            atoms = base_e.copy()
            atoms.set_cell(new_cell, scale_atoms=False)
            atoms.set_scaled_positions(frac_e)

            write_structure(atoms, xyz_path, format="xyz", comment=title)

            xtob(
                xyz_file=xyz_path,
                geo_file=geo_path,
                box_lengths=(float(a), float(b), float(c)),
                box_angles=(float(alpha), float(beta), float(gamma)),
                sort_by=sort_by,
                ascending=True,
            )
            out[mode].append(geo_path)

    return out

# =============================================================================
# 3. YAML file management for settings of trainset
# =============================================================================

# -----------------------------------------------------------------------------
# Yaml producer to input the cell dimensions and angles along with other
# settings for generating energy vs volume for expanded or compressed cells
# -----------------------------------------------------------------------------

def write_trainset_settings_yaml(
    *,
    out_path: str,
    name: str = "AlN example",
    source: str = "manual",
    mp_id: Optional[str] = None,
    # Elastic inputs
    elastic_max_strain_percent: float = 3.0,
    elastic_dstrain: float = 0.005,
    cij_gpa: Optional[Dict[str, float]] = None,
    elastic_cell: Optional[Dict[str, float]] = None,
    # Bulk inputs
    B0_gpa: float = 174.0,
    B0_prime: float = 1.5,
    bulk_max_volumetric_strain_percent: float = 6.0,
    bulk_dstrain_linear: float = 0.004,
    bulk_cell: Optional[Dict[str, float]] = None,
    # Output names
    trainset_file: str = "trainset_elastic.in",
    tables: Optional[Dict[str, str]] = None,
    elastic_xyz: Optional[str | Path] = "ground_elastic.xyz",
    bulk_xyz: Optional[str | Path] = "null",
    geo_enable: bool = True
) -> None:
    """
    Write a trainset settings YAML file for elastic-energy trainset generation.

    Works on
    --------
    YAML configuration files for trainset generation (trainset_elastic.yaml)

    Parameters
    ----------
    out_path : str
        Output YAML file path.
    name : str, optional
        Descriptive material name stored in metadata.
    source : str, optional
        Settings source label (e.g., ``"manual"`` or ``"materials_project"``).
    mp_id : str or None, optional
        Materials Project ID to store in metadata.
    elastic_max_strain_percent : float, optional
        Maximum elastic strain magnitude (%).
    elastic_dstrain : float, optional
        Elastic strain step size (unitless).
    cij_gpa : dict or None, optional
        Elastic constants in GPa with keys ``c11..c66``.
    elastic_cell : dict or None, optional
        Elastic reference cell with keys ``a,b,c,alpha,beta,gamma``.
    B0_gpa : float, optional
        Bulk modulus B0 (GPa).
    B0_prime : float, optional
        Bulk modulus pressure derivative B0' (dimensionless).
    bulk_max_volumetric_strain_percent : float, optional
        Maximum volumetric strain magnitude (%).
    bulk_dstrain_linear : float, optional
        Bulk linear strain step (unitless).
    bulk_cell : dict or None, optional
        Bulk reference cell with keys ``a,b,c,alpha,beta,gamma``.
    trainset_file : str, optional
        Trainset file name to store under output settings.
    tables : dict or None, optional
        Output table filenames keyed by mode (e.g., ``"bulk"``, ``"c11"``).
    elastic_xyz : str or pathlib.Path or None, optional
        Base XYZ used for elastic geometry generation when enabled.
    bulk_xyz : str or pathlib.Path or None, optional
        Optional base XYZ for bulk geometry generation when enabled.
    geo_enable : bool, optional
        Whether the YAML enables geometry generation.

    Returns
    -------
    None
        Writes a YAML settings file to disk.

    Examples
    --------
    >>> write_trainset_settings_yaml(
    ...     out_path="trainset_elastic.yaml",
    ...     name="AlN example",
    ...     source="manual",
    ... )
    """
    import os
    from typing import List

    # -------------------------
    # Defaults (match your example)
    # -------------------------
    if cij_gpa is None:
        cij_gpa = {
            "c11": 287,
            "c22": 287,
            "c33": 219,
            "c12": 100,
            "c13": 144,
            "c23": 144,
            "c44": 76,
            "c55": 76,
            "c66": 50,
        }

    if elastic_cell is None:
        elastic_cell = {
            "a": 2.85086,
            "b": 2.85086,
            "c": 3.49456,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        }

    if bulk_cell is None:
        bulk_cell = {
            "a": 2.85086,
            "b": 2.85086,
            "c": 3.49456,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        }

    if tables is None:
        tables = {
            "bulk": "EvsStrain_bulk.dat",
            "c11": "EvsStrain_c11.dat",
            "c22": "EvsStrain_c22.dat",
            "c33": "EvsStrain_c33.dat",
            "c12": "EvsStrain_c12.dat",
            "c13": "EvsStrain_c13.dat",
            "c23": "EvsStrain_c23.dat",
            "c44": "EvsStrain_c44.dat",
            "c55": "EvsStrain_c55.dat",
            "c66": "EvsStrain_c66.dat",
        }

    # -------------------------
    # Minimal validation
    # -------------------------
    required_cij = ("c11", "c22", "c33", "c12", "c13", "c23", "c44", "c55", "c66")
    missing = [k for k in required_cij if k not in cij_gpa]
    if missing:
        raise ValueError(f"cij_gpa is missing required keys: {missing}")

    for cell_name, cell in (("elastic_cell", elastic_cell), ("bulk_cell", bulk_cell)):
        for k in ("a", "b", "c", "alpha", "beta", "gamma"):
            if k not in cell:
                raise ValueError(f"{cell_name} missing key '{k}'")

    # -------------------------
    # YAML writing (manual; stable schema; no external deps)
    # -------------------------
    def _q(s: str) -> str:
        """Quote a string for YAML safely enough for our simple schema."""
        s2 = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s2}"'

    mp_id_yaml = "null" if mp_id is None else _q(mp_id)

    # Requirement (1): header comment must match the file name exactly
    yaml_filename = os.path.basename(out_path)

    lines: List[str] = []
    lines.append(f"# {yaml_filename}")
    lines.append("")
    # Requirement (2): short docstring-like comment at top
    lines.append("# This is the settings file used by ReaxKit's trainset generator.")
    lines.append("# Edit the values below (especially strains/moduli/cell) to match your material/system.")
    lines.append("")

    lines.append("metadata:")
    lines.append(f"  name: {_q(name)}")
    lines.append(f"  source: {_q(source)}  # 'manual' or 'materials_project'")
    lines.append(f"  mp_id: {mp_id_yaml}  # Optional: e.g. \"mp-661\"")
    lines.append("")

    lines.append("units:")
    lines.append('  elastic_constants: "GPa"')
    lines.append('  bulk_modulus: "GPa"')
    lines.append('  angles: "deg"')
    lines.append('  lengths: "angstrom"')
    lines.append('  strain: "percent"')
    lines.append("")

    # Requirement (4): explain elastic vs bulk with 1–2 comment lines before each section
    lines.append("# Elastic section: generates energy-vs-strain targets for c11..c66 (small linear strains).")
    lines.append("# Use this for harmonic elastic response around the reference cell.")
    lines.append("elastic:")
    lines.append(f"  max_strain_percent: {elastic_max_strain_percent}  # Max linear strain magnitude (%) for elastic targets")
    # Requirement (3) + (5): inline comment for dstrain explaining what it is and user input
    lines.append(f"  dstrain: {elastic_dstrain}  # Strain step size (unitless). Default = 0.5% = 0.005")
    lines.append("  cij_gpa:  # Elastic constants in GPa (c11,c22,c33,c12,c13,c23,c44,c55,c66)")
    for k in required_cij:
        lines.append(f"    {k}: {cij_gpa[k]}")
    lines.append("")
    lines.append("  cell:  # Elastic reference cell (a,b,c in Å; angles in deg)")
    lines.append(f"    a: {elastic_cell['a']}")
    lines.append(f"    b: {elastic_cell['b']}")
    lines.append(f"    c: {elastic_cell['c']}")
    lines.append(f"    alpha: {elastic_cell['alpha']}")
    lines.append(f"    beta: {elastic_cell['beta']}")
    lines.append(f"    gamma: {elastic_cell['gamma']}")
    lines.append("")

    # PATCH 1: keep separate structure sections (and make intent explicit)
    lines.append("# Input structures (XYZ). Used when geo.enable=true.")
    lines.append("structure 1:")
    lines.append(f'  elastic_xyz: {elastic_xyz}  # required if geo.enable=true')
    lines.append("")

    lines.append("# Bulk section: generates energy-vs-volume targets using an EOS (Vinet) over a wider strain range.")
    lines.append("# Use this to constrain compressibility (B0, B0') around the reference volume.")
    lines.append("bulk:")
    lines.append(f"  B0_gpa: {B0_gpa}  # Bulk modulus B0 at P=0 (GPa)")
    lines.append(f"  B0_prime: {B0_prime}  # Pressure derivative B0' = dB/dP at P=0 (dimensionless)")
    lines.append(f"  max_volumetric_strain_percent: {bulk_max_volumetric_strain_percent}  # Max volumetric strain magnitude (%)")
    # Requirement (5): inline comment for dstrain_linear
    lines.append(
        f"  dstrain_linear: {bulk_dstrain_linear}  # Linear isotropic strain step ε (unitless). "
        f"Volume uses V=V0*(1+ε)^3. Default = 0.4% = 0.004"
    )
    lines.append("")
    lines.append("  cell:  # Bulk/EOS reference cell (used to compute V0; a,b,c in Å; angles in deg)")
    lines.append(f"    a: {bulk_cell['a']}")
    lines.append(f"    b: {bulk_cell['b']}")
    lines.append(f"    c: {bulk_cell['c']}")
    lines.append(f"    alpha: {bulk_cell['alpha']}")
    lines.append(f"    beta: {bulk_cell['beta']}")
    lines.append(f"    gamma: {bulk_cell['gamma']}")
    lines.append("")

    # PATCH 1: second structure section
    lines.append("structure 2:")
    lines.append(f"  bulk_xyz: {bulk_xyz}  # optional; if null, reuse elastic_xyz")
    lines.append("")

    # PATCH 1: geo generation options (opt-in)
    lines.append("# Geometry generation options.")
    lines.append("geo:")
    lines.append(f"  enable: {geo_enable}  # set true to generate strained xyz + geo")
    lines.append("  sort_by: null  # e.g. 'z' or null")
    lines.append("")

    # Output section: you can usually keep this as-is unless you want different filenames.
    lines.append("# Output section: you can usually keep this as-is unless you want different filenames.")
    lines.append("# Added: output folders for strained XYZ and GEO files.")
    lines.append("output:")
    lines.append(f"  trainset_file: {_q(trainset_file)}")
    lines.append(f"  xyz_strained_dir: {_q('xyz_strained')}")
    lines.append(f"  geo_strained_dir: {_q('geo_strained')}")
    lines.append("  tables:")
    for key in ("bulk", "c11", "c22", "c33", "c12", "c13", "c23", "c44", "c55", "c66"):
        lines.append(f"    {key}: {_q(tables[key])}")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------------------------------------------------------
# reads a Yaml settings file and generates a trainset
# -----------------------------------------------------------------------------

def read_trainset_settings_yaml(yaml_path: str) -> dict:
    """
    Read a trainset settings YAML file into a configuration dictionary.

    Works on
    --------
    YAML configuration files for trainset generation (trainset_elastic.yaml)

    Parameters
    ----------
    yaml_path : str
        Path to a YAML settings file.

    Returns
    -------
    dict
        Parsed configuration mapping containing ``elastic``, ``bulk``, and ``output`` sections.

    Examples
    --------
    >>> cfg = read_trainset_settings_yaml("trainset_elastic.yaml")
    >>> sorted(cfg.keys())[:3]
    ['bulk', 'elastic', 'metadata']
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read trainset YAML files. "
            "Install with: pip install pyyaml"
        ) from exc

    from pathlib import Path

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/dictionary.")

    # Minimal structural validation (structures are required only if geo.enable=true)
    required_sections = ("elastic", "bulk", "output")
    missing = [k for k in required_sections if k not in data]
    if missing:
        raise ValueError(f"YAML is missing required sections: {missing}")

    geo_cfg = data.get("geo", {}) or {}
    enable_geo = bool(geo_cfg.get("enable", False))

    if enable_geo:
        # Require "structure 1" and validate elastic_xyz exists
        s1 = data.get("structure 1")
        if not isinstance(s1, dict):
            raise ValueError("Missing required section: 'structure 1' (required when geo.enable=true)")

        elastic_xyz = s1.get("elastic_xyz")
        if not elastic_xyz:
            raise ValueError("Missing required key: structure 1.elastic_xyz (required when geo.enable=true)")

        elastic_xyz_path = Path(elastic_xyz)
        if not elastic_xyz_path.is_absolute():
            elastic_xyz_path = (yaml_path.parent / elastic_xyz_path).resolve()
        if not elastic_xyz_path.exists():
            raise FileNotFoundError(f"structure 1.elastic_xyz does not exist: {elastic_xyz_path}")

        # Store resolved path back
        data["structure 1"]["elastic_xyz"] = str(elastic_xyz_path)

        # "structure 2" is optional; validate bulk_xyz if provided
        s2 = data.get("structure 2", {}) or {}
        if not isinstance(s2, dict):
            raise ValueError("'structure 2' must be a mapping if provided")

        bulk_xyz = s2.get("bulk_xyz")
        if bulk_xyz:
            bulk_xyz_path = Path(bulk_xyz)
            if not bulk_xyz_path.is_absolute():
                bulk_xyz_path = (yaml_path.parent / bulk_xyz_path).resolve()
            if not bulk_xyz_path.exists():
                raise FileNotFoundError(f"structure 2.bulk_xyz does not exist: {bulk_xyz_path}")
            data.setdefault("structure 2", {})["bulk_xyz"] = str(bulk_xyz_path)

    return data


def generate_trainset_from_yaml(
    yaml_path: str,
    out_dir: str,
    *,
    place_all_outputs_in_out_dir: bool = True,
    copy_input_xyz_into_out_dir: bool = True,
):
    """
    Generate a trainset and optional strained geometries from a YAML settings file.

    Works on
    --------
    YAML settings + XYZ inputs (optional) → trainset files and strained structures

    Parameters
    ----------
    yaml_path : str
        Path to the trainset settings YAML file.
    out_dir : str
        Output directory for generated files.
    place_all_outputs_in_out_dir : bool, optional
        If True, place all generated outputs (including geometry outputs) in ``out_dir``.
    copy_input_xyz_into_out_dir : bool, optional
        If True, copy input XYZ files into the output directory when geometry generation is enabled.

    Returns
    -------
    None
        Writes trainset files and (optionally) strained XYZ/GEO files to disk.

    Examples
    --------
    >>> generate_trainset_from_yaml("trainset_elastic.yaml", out_dir="out")
    """
    cfg = read_trainset_settings_yaml(yaml_path)
    yaml_path_p = Path(yaml_path).resolve()

    out_dir_p = Path(out_dir).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Bulk inputs
    # -------------------------
    bulk_cfg = cfg["bulk"]
    bulk_inputs = {
        "B0_gpa": bulk_cfg["B0_gpa"],
        "B0_prime": bulk_cfg["B0_prime"],
        "max_volumetric_strain_percent": bulk_cfg["max_volumetric_strain_percent"],
    }

    # -------------------------
    # Elastic inputs
    # -------------------------
    elastic_cfg = cfg["elastic"]
    elastic_inputs = {
        "max_strain_percent": elastic_cfg["max_strain_percent"],
        **elastic_cfg["cij_gpa"],
    }

    # -------------------------
    # Cells
    # -------------------------
    bulk_cell = bulk_cfg["cell"]
    elastic_cell = elastic_cfg.get("cell", bulk_cell)

    # -------------------------
    # Optional overrides
    # -------------------------
    bulk_options = {
        "linear_strain_step": bulk_cfg.get("dstrain_linear", 0.004)
    }
    elastic_options = {
        "strain_step": elastic_cfg.get("dstrain", 0.005)
    }

    # -------------------------
    # Generate elastic energy data (writes into out_dir)
    # -------------------------
    generate_all_energy_vs_volume_data(
        bulk_inputs=bulk_inputs,
        elastic_inputs=elastic_inputs,
        bulk_cell=bulk_cell,
        elastic_volume_cell=elastic_cell,
        bulk_options=bulk_options,
        elastic_options=elastic_options,
        out_dir=str(out_dir_p),
    )

    # -------------------------
    # Optional: generate strained geometries (writes xyz + geo)
    # -------------------------
    geo_cfg = cfg.get("geo", {}) or {}
    enable_geo = bool(geo_cfg.get("enable", False))
    if not enable_geo:
        return

    s1 = cfg["structure 1"]
    elastic_xyz = Path(s1["elastic_xyz"])
    if not elastic_xyz.is_absolute():
        elastic_xyz = (yaml_path_p.parent / elastic_xyz).resolve()
    if not elastic_xyz.exists():
        raise FileNotFoundError(f"structure 1.elastic_xyz does not exist: {elastic_xyz}")

    s2 = cfg.get("structure 2", {}) or {}
    bulk_xyz_val = s2.get("bulk_xyz")
    bulk_xyz = None
    if bulk_xyz_val:
        bulk_xyz = Path(bulk_xyz_val)
        if not bulk_xyz.is_absolute():
            bulk_xyz = (yaml_path_p.parent / bulk_xyz).resolve()
        if not bulk_xyz.exists():
            raise FileNotFoundError(f"structure 2.bulk_xyz does not exist: {bulk_xyz}")

    # Decide where geo outputs go
    geo_out_dir = out_dir_p if place_all_outputs_in_out_dir else yaml_path_p.parent

    # Optionally copy the input xyz files into geo_out_dir so folder is self-contained
    if copy_input_xyz_into_out_dir:
        elastic_xyz_dst = geo_out_dir / elastic_xyz.name
        if elastic_xyz.resolve() != elastic_xyz_dst.resolve():
            shutil.copy2(elastic_xyz, elastic_xyz_dst)
        elastic_xyz = elastic_xyz_dst

        if bulk_xyz is not None:
            bulk_xyz_dst = geo_out_dir / bulk_xyz.name
            if bulk_xyz.resolve() != bulk_xyz_dst.resolve():
                shutil.copy2(bulk_xyz, bulk_xyz_dst)
            bulk_xyz = bulk_xyz_dst

    # Convert YAML percent strains -> linear strain limits
    max_strain_elastic = elastic_cfg["max_strain_percent"] / 100.0
    dstrain_elastic = elastic_cfg.get("dstrain", 0.005)

    # bulk linear strain from volumetric percent (Fortran-style)
    max_vol = bulk_cfg["max_volumetric_strain_percent"] / 100.0
    max_strain_bulk_linear = (1.0 + max_vol) ** (1.0 / 3.0) - 1.0
    dstrain_bulk_linear = bulk_cfg.get("dstrain_linear", 0.004)

    sort_by = geo_cfg.get("sort_by")  # e.g. "z" or None

    generate_strained_geometries_with_xtob(
        elastic_xyz=str(elastic_xyz),
        bulk_xyz=None if bulk_xyz is None else str(bulk_xyz),
        elastic_cell=elastic_cell,
        bulk_cell=bulk_cell,
        max_strain_elastic=max_strain_elastic,
        dstrain_elastic=dstrain_elastic,
        max_strain_bulk_linear=max_strain_bulk_linear,
        dstrain_bulk_linear=dstrain_bulk_linear,
        out_dir=str(geo_out_dir),
        sort_by=sort_by,
    )


# =============================================================================
# 4. MP API Handler:
# Handles Material's project API to get mechanical properties, lattice
# dimensions and angles, and structure of the system
# =============================================================================

BulkModulusMode = Literal["voigt", "reuss", "vrh"]

def _tensor6x6_to_cij_dict(t6: List[List[float]]) -> Dict[str, float]:
    if t6 is None or len(t6) != 6 or any(len(row) != 6 for row in t6):
        raise ValueError("Elastic tensor must be a 6x6 matrix.")
    f = lambda i, j: float(t6[i][j])
    return {
        "c11": f(0, 0), "c22": f(1, 1), "c33": f(2, 2),
        "c12": f(0, 1), "c13": f(0, 2), "c23": f(1, 2),
        "c44": f(3, 3), "c55": f(4, 4), "c66": f(5, 5),
    }


def _extract_tensor6(elastic_tensor_obj: Any) -> Optional[List[List[float]]]:
    """Keep this tiny: support the 2–3 common mp-api shapes."""
    if elastic_tensor_obj is None:
        return None
    et = elastic_tensor_obj
    if hasattr(et, "ieee_format") and et.ieee_format is not None:
        return et.ieee_format
    if hasattr(et, "raw") and et.raw is not None:
        return et.raw
    if isinstance(et, (list, tuple)):
        return list(et)  # type: ignore[return-value]
    return None


def _pick_bulk_modulus(bm: Any, mode: BulkModulusMode) -> Optional[float]:
    if bm is None:
        return None
    val = getattr(bm, mode, None)  # bm.voigt / bm.reuss / bm.vrh
    return None if val is None else float(val)


def generate_trainset_settings_yaml_from_mp_simple(
    *,
    mp_id: str,
    out_yaml: str | Path,
    structure_dir: Optional[str | Path] = None,
    bulk_mode: BulkModulusMode = "vrh",
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Generate a trainset settings YAML and structures from a Materials Project ID.

    Minimal MP -> (structure + mechanics) -> CIF -> XYZ -> trainset_settings.yaml.

    - Fetches: structure, lattice (a,b,c,alpha,beta,gamma), elastic tensor (6x6), bulk modulus.
    - Writes: <mp_id>.cif and <mp_id>.xyz
    - Writes YAML where:
        - elastic_cell == bulk_cell == MP lattice
        - structure 1.elastic_xyz == structure 2.bulk_xyz == generated XYZ
        - geo.enable is set true (since geo comes from the XYZ)

    Works on
    --------
    Materials Project API + structure files (CIF/XYZ) + trainset settings YAML

    Parameters
    ----------
    mp_id : str
        Materials Project material ID (e.g., ``"mp-661"``).
    out_yaml : str or pathlib.Path
        Output YAML path to write.
    structure_dir : str or pathlib.Path or None, optional
        Directory to write structure files (CIF/XYZ). If None, uses the YAML folder.
    bulk_mode : {"voigt","reuss","vrh"}, optional
        Which bulk modulus value to store in YAML.
    api_key : str or None, optional
        Materials Project API key. If None, uses ``MP_API_KEY`` environment variable.
    verbose : bool, optional
        If True, print written paths to stdout.

    Returns
    -------
    dict[str, str]
        Mapping with keys: ``"cif"``, ``"xyz"``, ``"yaml"`` pointing to written file paths.

    Examples
    --------
    >>> out = generate_trainset_settings_yaml_from_mp_simple(
    ...     mp_id="mp-661",
    ...     out_yaml="trainset_elastic.yaml",
    ... )
    >>> sorted(out.keys())
    ['cif', 'xyz', 'yaml']
    """
    api_key = api_key or os.getenv("MP_API_KEY")
    if not api_key:
        raise RuntimeError("Set MP_API_KEY env var (or pass api_key=...).")

    out_yaml = Path(out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)

    sdir = Path(structure_dir) if structure_dir is not None else out_yaml.parent
    sdir.mkdir(parents=True, exist_ok=True)

    base = mp_id.replace(":", "_")
    cif_path = sdir / f"{base}.cif"
    xyz_path = sdir / f"{base}.xyz"

    # out_yaml: path to YAML you're writing
    # xyz_path: full path where you saved mp-661.xyz
    out_yaml_p = Path(out_yaml).resolve()
    xyz_path_p = Path(xyz_path).resolve()  # wherever you saved it (likely structure_dir/mp-661.xyz)

    # Write a RELATIVE path into YAML (relative to the YAML folder)
    elastic_xyz_for_yaml = xyz_path_p.relative_to(out_yaml_p.parent).as_posix()

    with MPRester(api_key) as mpr:
        # 1) summary: structure + lattice
        sdoc = mpr.materials.summary.search(
            material_ids=[mp_id],
            fields=["material_id", "formula_pretty", "structure"],
        )[0]
        structure = sdoc.structure
        lat = structure.lattice
        name = getattr(sdoc, "formula_pretty", None) or mp_id

        cell = {
            "a": float(lat.a), "b": float(lat.b), "c": float(lat.c),
            "alpha": float(lat.alpha), "beta": float(lat.beta), "gamma": float(lat.gamma),
        }

        # 2) elasticity: elastic tensor + bulk modulus
        edocs = mpr.materials.elasticity.search(
            material_ids=[mp_id],
            fields=["material_id", "elastic_tensor", "bulk_modulus"],
        )
        if not edocs:
            raise ValueError(f"No elasticity data for {mp_id} (cannot populate elastic/bulk).")
        edoc = edocs[0]

        tensor6 = _extract_tensor6(getattr(edoc, "elastic_tensor", None))
        if tensor6 is None:
            raise ValueError(f"{mp_id}: elastic_tensor missing/unreadable.")
        cij = _tensor6x6_to_cij_dict(tensor6)

        B0 = _pick_bulk_modulus(getattr(edoc, "bulk_modulus", None), bulk_mode)
        if B0 is None:
            raise ValueError(f"{mp_id}: bulk_modulus.{bulk_mode} missing/unreadable.")

    # 3) CIF -> XYZ (use geo_generator’s writer for XYZ)
    # Write CIF via pymatgen structure.to(...) (simple; if it fails you can add your CifWriter fallback)
    structure.to(filename=str(cif_path), fmt="cif")

    atoms = read_structure(cif_path, format="cif")
    write_structure(atoms, xyz_path, format="xyz", comment=mp_id)

    # 4) YAML (bulk == elastic, geo == same XYZ)
    # NOTE: this assumes your writer supports these fields now (geo + elastic_xyz + bulk_xyz).
    write_trainset_settings_yaml(
        out_path=str(out_yaml),
        name=f"{name} ({mp_id})",
        source="materials_project",
        mp_id=mp_id,
        cij_gpa=cij,
        B0_gpa=B0,
        elastic_cell=cell,
        bulk_cell=cell,
        elastic_xyz=str(elastic_xyz_for_yaml),
        bulk_xyz=str(elastic_xyz_for_yaml),
        geo_enable=True,
    )

    if verbose:
        print(f"[MP] CIF:  {cif_path}")
        print(f"[MP] XYZ:  {xyz_path}")
        print(f"[MP] YAML: {out_yaml}")

    return {"cif": str(cif_path), "xyz": str(xyz_path), "yaml": str(out_yaml)}



