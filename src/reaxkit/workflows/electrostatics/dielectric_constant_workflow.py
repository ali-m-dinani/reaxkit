"""CLI workflow for Kubo--Green dielectric response from an Excel time series."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import reaxkit.engine  # noqa: F401

from reaxkit.analysis.electrostatics.dielectric_constant import (
    DIPOLE_TO_COULOMB_METERS,
    FREQUENCY_FROM_HZ,
    TIME_TO_SECONDS,
    VOLUME_TO_CUBIC_METERS,
    DielectricConstantRequest,
    DielectricConstantTask,
)
from reaxkit.analysis.electrostatics.electrostatics import (
    calculate_trajectory_volumes,
)
from reaxkit.core.platform.engine_resolver import resolve_engine
from reaxkit.domain.data_models import TrajectoryData

COMMAND = "get-dielectric-constant"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = ("get_dielectric_constant", "dielectric-constant")


@dataclass(frozen=True)
class VolumeResolution:
    """Resolved mean volume and its frame-level provenance."""

    value: float
    unit: str
    method: str
    frame_count: int | None = None
    minimum: float | None = None
    maximum: float | None = None


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _sheet_name(value: str) -> str | int:
    stripped = str(value).strip()
    return int(stripped) if stripped.isdigit() else stripped


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Configure the Excel-to-dielectric command parser."""
    if command not in (*ALL_COMMANDS, *ALL_LEGACY_COMMANDS):
        raise KeyError(command)
    parser.set_defaults(command=COMMAND)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Calculate static and complex dielectric response from a dipole time series using the Kubo--Green relation.

The Excel sheet must contain one uniformly sampled time column and one scalar
dipole column. Volume is calculated from a trajectory with hull (default), bbox,
or cell. Hull and bbox exclude empty cell vacuum; cell includes it. The mean
frame volume is used in the Kubo--Green prefactor. The dipole mean is removed
before the autocorrelation is formed.
Use --dipole-kind total for the isotropic 1/3 relation shown in the supplied
equation, or component for one signed Cartesian dipole component. A magnitude-
only series cannot retain vector orientation, so its dynamic response is an
isotropic scalar approximation.

Examples:
  1. Dipole moment in Debye sampled every femtosecond:
     reaxkit get-dielectric-constant --input dipole.xlsx --trajectory xmolout --time-unit fs --dipole-unit debye --temperature 300

  2. A z-component series with a 5 ps correlation cutoff:
     reaxkit get-dielectric-constant --input dipole.xlsx --trajectory dump.lammpstrj --engine lammps --volume-method bbox --time-column t --dipole-column mu_z --time-unit ps --dipole-unit e-angstrom --dipole-kind component --temperature 500 --max-lag 5 --output dielectric.xlsx
"""
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Read the dipole time series from this Excel workbook. Example: --input ./dipole.xlsx.",
    )
    parser.add_argument(
        "--sheet", default="0",
        help="Select a worksheet by zero-based index or name. Example: --sheet production.",
    )
    parser.add_argument(
        "--time-column", default="t",
        help="Select the time column. Example: --time-column t.",
    )
    parser.add_argument(
        "--dipole-column", default="dipole",
        help="Select the scalar dipole column. Example: --dipole-column mu_z.",
    )
    parser.add_argument(
        "--time-unit", choices=tuple(TIME_TO_SECONDS), required=True,
        help="Declare the unit of the time column. Example: --time-unit fs.",
    )
    parser.add_argument(
        "--dipole-unit", choices=tuple(DIPOLE_TO_COULOMB_METERS), required=True,
        help="Declare the dipole unit (c-m, debye, or e-angstrom). Example: --dipole-unit debye.",
    )
    parser.add_argument(
        "--temperature", type=_positive_float, required=True,
        help="Set the simulation temperature in kelvin. Example: --temperature 300.",
    )
    parser.add_argument(
        "--trajectory", "--xmolout", dest="trajectory", type=Path, default=None,
        help="Read coordinates used to calculate volume from this trajectory. Example: --trajectory ./run/xmolout.",
    )
    parser.add_argument(
        "--engine", choices=("reaxff", "ams", "lammps"), default=None,
        help="Override automatic trajectory-engine detection. Example: --engine lammps.",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("."),
        help="Set the simulation directory used for trajectory discovery. Example: --run-dir ./run.",
    )
    parser.add_argument(
        "--volume-method", choices=("hull", "bbox", "cell"), default="hull",
        help="Calculate volume from the atomic hull, occupied bounding box, or full cell. Example: --volume-method hull.",
    )
    parser.add_argument(
        "--volume", type=_positive_float, default=None,
        help="Override trajectory-based volume calculation with a fixed value. Example: --volume 25000.",
    )
    parser.add_argument(
        "--volume-unit", choices=tuple(VOLUME_TO_CUBIC_METERS), default="angstrom3",
        help="Declare the unit of a manual --volume value. Example: --volume-unit angstrom3.",
    )
    parser.add_argument(
        "--dipole-kind", choices=("total", "component"), default="total",
        help="Choose total for the isotropic factor 1/3 or component for factor 1. Example: --dipole-kind component.",
    )
    parser.add_argument(
        "--frequency-unit", choices=tuple(FREQUENCY_FROM_HZ), default="thz",
        help="Choose the reported frequency unit. Example: --frequency-unit cm-1.",
    )
    parser.add_argument(
        "--max-lag", type=_positive_float, default=None,
        help="Truncate the correlation integral at this lag, in --time-unit. Example: --max-lag 5.",
    )
    parser.add_argument(
        "--max-frequency", type=float, default=None,
        help="Keep frequencies at or below this value, in --frequency-unit. Example: --max-frequency 20.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write summary, spectrum, and autocorrelation sheets here. Example: --output dielectric.xlsx.",
    )
    return parser


def build_request(
    args: argparse.Namespace,
    volume: VolumeResolution | None = None,
) -> DielectricConstantRequest:
    """Map CLI flags to the typed dielectric request."""
    resolved = volume
    if resolved is None and args.volume is not None:
        resolved = VolumeResolution(
            value=float(args.volume),
            unit=args.volume_unit,
            method="manual",
        )
    if resolved is None:
        raise ValueError(
            "Volume has not been resolved. Provide --trajectory/--run-dir for "
            "hull, bbox, or cell calculation, or provide --volume explicitly."
        )
    return DielectricConstantRequest(
        temperature=float(args.temperature),
        volume=resolved.value,
        time_unit=args.time_unit,
        dipole_unit=args.dipole_unit,
        volume_unit=resolved.unit,
        time_column=str(args.time_column),
        dipole_column=str(args.dipole_column),
        dipole_kind=args.dipole_kind,
        frequency_unit=args.frequency_unit,
        max_lag=args.max_lag,
        max_frequency=args.max_frequency,
        volume_method=resolved.method,
        volume_frame_count=resolved.frame_count,
        volume_min=resolved.minimum,
        volume_max=resolved.maximum,
    )


def load_trajectory_for_volume(args: argparse.Namespace) -> TrajectoryData:
    """Load trajectory coordinates through the configured ReaxKit adapter."""
    source = Path(args.trajectory) if args.trajectory is not None else Path(args.run_dir)
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Trajectory or run directory not found: {source}")
    adapter = resolve_engine(str(source), engine=args.engine)
    load_args = {
        "run_dir": str(Path(args.run_dir).expanduser().resolve()),
        "input": str(source),
        "xmolout": str(source),
        "dump": str(source),
        "dump_file": str(source),
        "trajectory": str(source),
        "rkf": str(source),
        "kf": str(source),
    }
    return adapter.load(TrajectoryData, load_args)


def resolve_volume(args: argparse.Namespace) -> VolumeResolution:
    """Resolve a manual volume or calculate the mean trajectory volume."""
    if args.volume is not None:
        return VolumeResolution(
            value=float(args.volume),
            unit=args.volume_unit,
            method="manual",
        )
    trajectory = load_trajectory_for_volume(args)
    volumes = calculate_trajectory_volumes(trajectory, args.volume_method)
    return VolumeResolution(
        value=float(np.mean(volumes)),
        unit="angstrom3",
        method=args.volume_method,
        frame_count=int(volumes.size),
        minimum=float(np.min(volumes)),
        maximum=float(np.max(volumes)),
    )


def read_input_workbook(path: Path, sheet: str | int) -> pd.DataFrame:
    """Read the selected Excel worksheet with a focused dependency error."""
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Excel input file not found: {source}")
    try:
        return pd.read_excel(source, sheet_name=sheet)
    except ImportError as exc:
        raise RuntimeError(
            "Reading Excel files requires openpyxl. Install ReaxKit with its io extra: "
            "pip install 'reaxkit[io]'."
        ) from exc


def write_result_workbook(result, path: Path) -> Path:
    """Write the complete dielectric result as a three-sheet workbook."""
    output = path.expanduser().resolve()
    if output.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError("--output must use the .xlsx or .xlsm extension.")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            result.summary.to_excel(writer, sheet_name="summary", index=False)
            result.spectrum.to_excel(writer, sheet_name="spectrum", index=False)
            result.autocorrelation.to_excel(writer, sheet_name="autocorrelation", index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Writing Excel files requires openpyxl. Install ReaxKit with its io extra: "
            "pip install 'reaxkit[io]'."
        ) from exc
    return output


def run_main(command: str, args: argparse.Namespace) -> int:
    """Read the input workbook, calculate permittivity, and write the result."""
    if command not in (*ALL_COMMANDS, *ALL_LEGACY_COMMANDS):
        raise KeyError(command)
    if args.max_frequency is not None and args.max_frequency < 0.0:
        raise ValueError("--max-frequency must be greater than or equal to zero.")
    source = Path(args.input)
    data = read_input_workbook(source, _sheet_name(args.sheet))
    volume = resolve_volume(args)
    result = DielectricConstantTask().run(data, build_request(args, volume))
    output = Path(args.output) if args.output is not None else source.with_name(
        f"{source.stem}_dielectric.xlsx"
    )
    saved = write_result_workbook(result, output)
    print(f"Static dielectric constant: {result.static_dielectric_constant:.10g}")
    print(
        f"Volume: {volume.value:.10g} {volume.unit} "
        f"({volume.method}, {volume.frame_count or 1} frame(s))"
    )
    print(f"Wrote dielectric summary, spectrum, and autocorrelation to {saved}")
    return 0


__all__ = [
    "ALL_COMMANDS",
    "ALL_LEGACY_COMMANDS",
    "COMMAND",
    "build_parser",
    "build_request",
    "read_input_workbook",
    "resolve_volume",
    "run_main",
    "write_result_workbook",
]
