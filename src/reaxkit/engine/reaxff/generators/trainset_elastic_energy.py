"""
Trainset elastic-energy generation utilities.

This module contains pure generation helpers and thin file writers for
bulk/elastic trainset targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import warnings

from reaxkit.core.constants import const
from reaxkit.utils.equation_of_states import vinet_energy_trainset


AVOGADRO_CONSTANT = const("AVOGADRO_CONSTANT")
ENERGY_CONVERSION_FACTOR = 10.0 * 4.184 / AVOGADRO_CONSTANT
ENERGY_MODE_ORDER = ["c11", "c22", "c33", "c12", "c13", "c23", "c44", "c55", "c66"]


@dataclass(frozen=True)
class CellSpec:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }


@dataclass(frozen=True)
class BulkEnergySpec:
    bulk_modulus_gpa: float
    bulk_modulus_pressure_derivative: float
    max_volumetric_strain_percent: float
    cell: CellSpec
    linear_strain_step: float = 0.004
    reference_energy: float = 0.0
    weight: float = 1.0


@dataclass(frozen=True)
class ElasticEnergySpec:
    elastic_constants_gpa: Dict[str, float]
    max_strain_percent: float
    volume_reference_cell: CellSpec
    strain_step: float = 0.005
    weight: float = 1.0


@dataclass(frozen=True)
class TrainsetEnergyResult:
    trainset_text: str
    bulk_table: List[Tuple[float, float]]
    elastic_tables: Dict[str, List[Tuple[float, float]]]
    warnings: List[str] = field(default_factory=list)


def _fortran_nint(nonnegative_value: float) -> int:
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
    degrees_to_radians = math.pi / 180.0
    alpha_rad = alpha_deg * degrees_to_radians
    beta_rad = beta_deg * degrees_to_radians
    gamma_rad = gamma_deg * degrees_to_radians

    sin_alpha = math.sin(alpha_rad)
    cos_alpha = math.cos(alpha_rad)
    sin_beta = math.sin(beta_rad)
    cos_beta = math.cos(beta_rad)

    denominator = sin_alpha * sin_beta
    if abs(denominator) < 1e-16:
        raise ValueError("Invalid cell angles: sin(alpha)*sin(beta) ~ 0, cannot compute volume.")

    cos_intermediate_angle = (math.cos(gamma_rad) - cos_alpha * cos_beta) / denominator
    cos_intermediate_angle = max(-1.0, min(1.0, cos_intermediate_angle))
    sin_intermediate_angle = math.sqrt(max(0.0, 1.0 - cos_intermediate_angle * cos_intermediate_angle))

    basis_x_component = a_length * sin_beta * sin_intermediate_angle
    basis_y_component = b_length * sin_alpha
    basis_z_component = c_length
    return basis_x_component * basis_y_component * basis_z_component


def _build_symmetric_grid(*, max_abs_value: float, step: float, grid_mode: str) -> List[float]:
    if step <= 0:
        raise ValueError("step must be positive.")
    rounded_ratio = _fortran_nint(max_abs_value / step)
    if grid_mode not in ("bulk", "elastic"):
        raise ValueError("grid_mode must be 'bulk' or 'elastic'.")
    num_strain_steps = rounded_ratio + 1 if (rounded_ratio * step < max_abs_value) else rounded_ratio
    return [step * n for n in range(-num_strain_steps, num_strain_steps + 1)]


def _make_label(prefix: str, signed_index: int) -> str:
    if signed_index == 0:
        return f"{prefix}_0"
    compression_or_expansion = "c" if signed_index < 0 else "e"
    return f"{prefix}_{compression_or_expansion}{abs(signed_index):04d}"


def _index_from_grid_value(grid_value: float, step: float) -> int:
    if step == 0:
        return 0
    return int(round(grid_value / step))


def _warn_if_nonorthogonal(cell: CellSpec, label: str) -> Optional[str]:
    angles = [cell.alpha, cell.beta, cell.gamma]
    tol = 1e-6
    if any(abs(a - 90.0) > tol for a in angles):
        return (
            f"{label} cell is non-orthogonal (angles = {angles}). "
            "Elastic energy targets assume an orthogonal lattice."
        )
    return None


def _generate_bulk_data(spec: BulkEnergySpec) -> Tuple[List[Tuple[float, float]], List[str]]:
    cell = spec.cell.as_dict()
    reference_volume = _compute_orthogonal_lattice_cell_volume(
        a_length=cell["a"],
        b_length=cell["b"],
        c_length=cell["c"],
        alpha_deg=cell["alpha"],
        beta_deg=cell["beta"],
        gamma_deg=cell["gamma"],
    )
    equivalent_linear_strain_max = (1.0 + spec.max_volumetric_strain_percent / 100.0) ** (1.0 / 3.0) - 1.0
    linear_strain_grid = _build_symmetric_grid(
        max_abs_value=equivalent_linear_strain_max,
        step=spec.linear_strain_step,
        grid_mode="bulk",
    )

    bulk_table: List[Tuple[float, float]] = []
    trainset_lines: List[str] = []
    for linear_strain in linear_strain_grid:
        signed_step_index = _index_from_grid_value(linear_strain, spec.linear_strain_step)
        strained_volume = reference_volume * (1.0 + linear_strain) ** 3
        eos_energy = vinet_energy_trainset(
            volume=strained_volume,
            reference_volume=reference_volume,
            bulk_modulus_gpa=spec.bulk_modulus_gpa,
            bulk_modulus_pressure_derivative=spec.bulk_modulus_pressure_derivative,
            reference_energy=spec.reference_energy,
            energy_conversion_factor=ENERGY_CONVERSION_FACTOR,
        )
        if eos_energy == 0.0:
            eos_energy = 1e-4
        label = _make_label("bulk", signed_step_index)
        trainset_lines.append(f" {spec.weight:.4f}   +   {label:<11} /1  -  bulk_0 /1          {eos_energy:12.4f}")
        bulk_table.append((strained_volume, eos_energy))
    return bulk_table, trainset_lines


def _generate_elastic_data(spec: ElasticEnergySpec) -> Dict[str, Tuple[List[Tuple[float, float]], List[str]]]:
    cell = spec.volume_reference_cell.as_dict()
    reference_volume = _compute_orthogonal_lattice_cell_volume(
        a_length=cell["a"],
        b_length=cell["b"],
        c_length=cell["c"],
        alpha_deg=cell["alpha"],
        beta_deg=cell["beta"],
        gamma_deg=cell["gamma"],
    )
    max_linear_strain = spec.max_strain_percent / 100.0
    linear_strain_grid = _build_symmetric_grid(
        max_abs_value=max_linear_strain,
        step=spec.strain_step,
        grid_mode="elastic",
    )
    c = spec.elastic_constants_gpa

    def normal_strain_prefactor(cij: float) -> float:
        return cij * reference_volume / (2.0 * ENERGY_CONVERSION_FACTOR)

    def shear_strain_prefactor(cij: float) -> float:
        return 2.0 * cij * reference_volume / ENERGY_CONVERSION_FACTOR

    def coupling_strain_prefactor(cij: float, cii: float, cjj: float) -> float:
        return (-cij + (cii + cjj) / 2.0) * reference_volume / ENERGY_CONVERSION_FACTOR

    coeffs = {
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

    result: Dict[str, Tuple[List[Tuple[float, float]], List[str]]] = {}
    for mode_name, quadratic_prefactor in coeffs.items():
        table_rows: List[Tuple[float, float]] = []
        trainset_lines: List[str] = []
        for linear_strain in linear_strain_grid:
            signed_step_index = _index_from_grid_value(linear_strain, spec.strain_step)
            energy = quadratic_prefactor * (linear_strain ** 2)
            if energy == 0.0:
                energy = 1e-4
            label = _make_label(mode_name, signed_step_index)
            trainset_lines.append(f" {spec.weight:.4f}   +   {label:<12} /1  -  {mode_name}_0 /1          {energy:12.4f}")
            table_rows.append((linear_strain, energy))
        result[mode_name] = (table_rows, trainset_lines)
    return result


def _generate_trainset_energy(bulk_spec: BulkEnergySpec, elastic_spec: ElasticEnergySpec) -> TrainsetEnergyResult:
    warnings_list: List[str] = []
    for label, cell in (("Elastic", elastic_spec.volume_reference_cell), ("Bulk", bulk_spec.cell)):
        warning_message = _warn_if_nonorthogonal(cell, label)
        if warning_message:
            warnings_list.append(warning_message)

    bulk_table, bulk_lines = _generate_bulk_data(bulk_spec)
    elastic_targets = _generate_elastic_data(elastic_spec)
    trainset_lines: List[str] = ["ENERGY", "# Volume Bulk_EOS", *bulk_lines]
    elastic_tables: Dict[str, List[Tuple[float, float]]] = {}
    for mode in ENERGY_MODE_ORDER:
        trainset_lines.append(f"# Volume {mode.upper()}_EOS")
        trainset_lines.extend(elastic_targets[mode][1])
        elastic_tables[mode] = elastic_targets[mode][0]
    trainset_lines.append("ENDENERGY")

    return TrainsetEnergyResult(
        trainset_text="\n".join(trainset_lines) + "\n",
        bulk_table=bulk_table,
        elastic_tables=elastic_tables,
        warnings=warnings_list,
    )


def _generate_trainset_energy_with_source_note(
    bulk_spec: BulkEnergySpec,
    elastic_spec: ElasticEnergySpec,
    *,
    source_note: str | None = None,
) -> TrainsetEnergyResult:
    result = _generate_trainset_energy(bulk_spec, elastic_spec)
    if not source_note:
        return result
    lines = result.trainset_text.splitlines()
    if not lines or lines[0] != "ENERGY":
        return result
    lines.insert(1, source_note)
    return TrainsetEnergyResult(
        trainset_text="\n".join(lines) + "\n",
        bulk_table=result.bulk_table,
        elastic_tables=result.elastic_tables,
        warnings=result.warnings,
    )


def _write_trainset_energy(
    result: TrainsetEnergyResult,
    *,
    out_dir: str | Path,
    trainset_filename: str = "trainset_elastic.in",
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainset_path = out_dir / trainset_filename
    trainset_path.write_text(result.trainset_text, encoding="utf-8")

    def _write_two_column_table(path: Path, header: str, rows: List[Tuple[float, float]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            fh.write(header.rstrip() + "\n")
            for x, y in rows:
                fh.write(f"{x:8.3f}  {y:12.4f}\n")

    written: Dict[str, Path] = {"trainset": trainset_path}
    bulk_path = out_dir / "EvsStrain_bulk.dat"
    _write_two_column_table(bulk_path, "# Volume   Energy", result.bulk_table)
    written["bulk"] = bulk_path
    for mode in ENERGY_MODE_ORDER:
        path = out_dir / f"EvsStrain_{mode}.dat"
        _write_two_column_table(path, "# Strain   Energy", result.elastic_tables[mode])
        written[mode] = path
    return written


def _generate_all_energy_vs_volume_data(
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
    bulk_options = bulk_options or {}
    elastic_options = elastic_options or {}
    elastic_volume_cell = elastic_volume_cell or bulk_cell

    bulk_spec = BulkEnergySpec(
        bulk_modulus_gpa=bulk_inputs["B0_gpa"],
        bulk_modulus_pressure_derivative=bulk_inputs["B0_prime"],
        max_volumetric_strain_percent=bulk_inputs["max_volumetric_strain_percent"],
        cell=CellSpec(**bulk_cell),
        linear_strain_step=float(bulk_options.get("linear_strain_step", 0.004)),
        reference_energy=float(bulk_options.get("reference_energy", 0.0)),
    )
    elastic_spec = ElasticEnergySpec(
        elastic_constants_gpa={key: elastic_inputs[key] for key in ENERGY_MODE_ORDER},
        max_strain_percent=elastic_inputs["max_strain_percent"],
        volume_reference_cell=CellSpec(**elastic_volume_cell),
        strain_step=float(elastic_options.get("strain_step", 0.005)),
    )
    result = _generate_trainset_energy(bulk_spec, elastic_spec)
    for message in result.warnings:
        warnings.warn(message, stacklevel=2)
    _write_trainset_energy(result, out_dir=out_dir, trainset_filename=trainset_filename)


def _gen_elastic_trainset_energy_vs_volume_data(
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
    """Canonical alias for elastic trainset energy-vs-volume generation."""
    return _generate_all_energy_vs_volume_data(
        out_dir=out_dir,
        bulk_inputs=bulk_inputs,
        elastic_inputs=elastic_inputs,
        bulk_cell=bulk_cell,
        elastic_volume_cell=elastic_volume_cell,
        bulk_options=bulk_options,
        elastic_options=elastic_options,
        trainset_filename=trainset_filename,
    )
