"""File workflow for coarse isomer representative detection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reaxkit.analysis.molecular_analysis.isomer_representative_detection import (
    IsomerRepresentativeDetectionRequest,
    IsomerRepresentativeDetectionResult,
    IsomerRepresentativeDetectionTask,
)
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.domain.data_models import ConnectivityTrajectoryData
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter

ALL_COMMANDS = ("detect-isomer-representatives",)
ALL_LEGACY_COMMANDS = ("detect_isomer_representatives", "detect-isomers", "detect_isomers")
COMMAND_ALIASES = {"detect-isomer-representatives": ALL_LEGACY_COMMANDS}


@dataclass(frozen=True)
class LegacyIsomerRepresentativeControl:
    """Legacy control file settings used by this file workflow."""

    atom_map: dict[str, int]
    input_formula: dict[str, int]
    isomer_run: int = 1
    isomer_prefixname: str = ""


@dataclass(frozen=True)
class IsomerRepresentativeFileWorkflowResult:
    """File outputs from the representative-detection workflow."""

    detection: IsomerRepresentativeDetectionResult
    output_xmolout_isomers: Path
    isomer_dir: Path | None
    log_path: Path


def _parse_symbol_counts(raw: str, *, field_name: str) -> dict[str, int]:
    """Parse ``Element:count`` comma-separated control values."""
    out: dict[str, int] = {}
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid {field_name} token {token!r}; expected Element:value.")
        symbol, value = token.split(":", 1)
        symbol = symbol.strip()
        if not symbol:
            raise ValueError(f"Invalid {field_name} token {token!r}; missing element symbol.")
        out[symbol] = int(value.strip())
    if not out:
        raise ValueError(f"{field_name} cannot be empty.")
    return out


def parse_legacy_isomer_representative_control(path: str | Path) -> LegacyIsomerRepresentativeControl:
    """Parse a legacy ``control_params`` file for workflow compatibility."""
    control_path = Path(path)
    if not control_path.is_file():
        raise FileNotFoundError(f"control_params file not found: {control_path}")

    values: dict[str, str] = {}
    for line in control_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(f"Invalid control_params line {line!r}; expected key=value.")
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()

    missing = [key for key in ("atom_map", "input_formula", "isomer_prefixname") if not values.get(key)]
    if missing:
        raise ValueError(f"control_params is missing required keys: {missing}")

    return LegacyIsomerRepresentativeControl(
        atom_map=_parse_symbol_counts(values["atom_map"], field_name="atom_map"),
        input_formula=_parse_symbol_counts(values["input_formula"], field_name="input_formula"),
        isomer_run=int(values.get("isomer_run", "1")),
        isomer_prefixname=values["isomer_prefixname"],
    )


def _write_representative_outputs(
    result: IsomerRepresentativeDetectionResult,
    *,
    data: ConnectivityTrajectoryData,
    output_xmolout_isomers: Path,
    isomer_dir: Path | None,
) -> None:
    """Write representative structures from canonical trajectory data."""
    trajectory = data.trajectory
    atom_id_to_index = {int(atom_id): idx for idx, atom_id in enumerate(trajectory.atom_ids)}
    molecule_nums = np.asarray(data.connectivity.simulation.molecule_nums, dtype=int)

    output_xmolout_isomers.parent.mkdir(parents=True, exist_ok=True)
    if isomer_dir is not None:
        isomer_dir.mkdir(parents=True, exist_ok=True)

    with output_xmolout_isomers.open("w", encoding="utf-8") as combined:
        for record in result.records:
            atom_indices = [atom_id_to_index[int(atom_id)] for atom_id in record.atom_ids]
            combined.write(f"{record.atom_count}\n")
            combined.write(f"{record.structure_name}\n")
            selected_lines: list[str] = []
            for atom_index in atom_indices:
                coords = trajectory.positions[record.frame_index, atom_index]
                element = str(trajectory.elements[atom_index])
                molecule_id = int(molecule_nums[record.frame_index, atom_index])
                line_with_molecule = (
                    f"{element:<2} {coords[0]:10.5f} {coords[1]:10.5f} {coords[2]:10.5f}"
                    f" {molecule_id:8d}"
                )
                combined.write(f"{line_with_molecule}\n")
                selected_lines.append(
                    f"{element:<2} {coords[0]:10.5f} {coords[1]:10.5f} {coords[2]:10.5f}"
                )

            if isomer_dir is not None:
                structure_dir = isomer_dir / record.structure_name
                structure_dir.mkdir(parents=True, exist_ok=True)
                with (structure_dir / "xmolout").open("w", encoding="utf-8") as out:
                    for line in selected_lines:
                        out.write(f"{line}\n")
                    out.write("&\n")


def _write_log(result: IsomerRepresentativeDetectionResult, log_path: Path) -> None:
    """Write a simple representative-detection log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(result.records)}\n")
        for record in result.records:
            handle.write(
                f"isomer: {record.structure_name} Iteration: {record.iteration} "
                f"Molecule No: {record.molecule_id}\n"
            )
            for label, count in record.bond_label_counts.items():
                handle.write(f"{label}: {count}\n")


def detect_isomer_representatives_from_reaxff_files(
    *,
    fort7_path: str | Path,
    xmolout_path: str | Path,
    control_path: str | Path,
    output_dir: str | Path,
    write_isomer_dirs: bool | None = None,
    max_representatives: int | None = None,
    output_name: str = "xmolout_isomers",
    log_name: str = "isomer_run_log.txt",
) -> IsomerRepresentativeFileWorkflowResult:
    """Run representative detection for ReaxFF files through canonical data."""
    fort7 = Path(fort7_path)
    xmolout = Path(xmolout_path)
    if not fort7.is_file():
        raise FileNotFoundError(f"fort.7 file not found: {fort7}")
    if not xmolout.is_file():
        raise FileNotFoundError(f"xmolout file not found: {xmolout}")

    control = parse_legacy_isomer_representative_control(control_path)
    data = ReaxFFAdapter().load_connectivity_trajectory(
        {"fort7": str(fort7), "xmolout": str(xmolout)},
    )
    request = IsomerRepresentativeDetectionRequest(
        target_formula=control.input_formula,
        structure_prefix=control.isomer_prefixname,
        max_representatives=max_representatives,
    )
    detection = IsomerRepresentativeDetectionTask().run(data, request)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    should_write_dirs = bool(control.isomer_run == 2) if write_isomer_dirs is None else bool(write_isomer_dirs)
    isomer_dir = out_dir / "isomers" if should_write_dirs else None
    output_xmolout = out_dir / output_name
    _write_representative_outputs(
        detection,
        data=data,
        output_xmolout_isomers=output_xmolout,
        isomer_dir=isomer_dir,
    )
    log_path = out_dir / log_name
    _write_log(detection, log_path)
    return IsomerRepresentativeFileWorkflowResult(
        detection=detection,
        output_xmolout_isomers=output_xmolout,
        isomer_dir=isomer_dir,
        log_path=log_path,
    )


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Build the parser for isomer representative detection."""
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS, aliases=COMMAND_ALIASES)
    parser.set_defaults(command=canonical)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Detect coarse target-formula isomer representatives using canonical ReaxKit data.\n"
        "For this file workflow, ReaxFF `fort.7` and `xmolout` are first loaded through ReaxFFAdapter\n"
        "as ConnectivityTrajectoryData, then the engine-independent analyzer is executed.\n"
        "The target formula, prefix, and legacy folder behavior are read from `control_params`.\n\n"
        "Examples:\n"
        "   reaxkit detect-isomer-representatives --fort7 fort.7 --xmolout xmolout --control control_params --output-dir isomer_outputs\n"
        "   reaxkit detect-isomer-representatives --output-dir isomer_outputs --max-representatives 10"
    )
    parser.add_argument("--fort7", default="fort.7", help="Input ReaxFF fort.7 file.")
    parser.add_argument("--xmolout", default="xmolout", help="Input ReaxFF xmolout file.")
    parser.add_argument("--control", default="control_params", help="Input legacy control_params file.")
    parser.add_argument(
        "--output-dir",
        default="isomer_outputs",
        help="Output directory for xmolout_isomers, isomer_run_log.txt, and optional isomer folders.",
    )
    parser.add_argument(
        "--max-representatives",
        type=int,
        default=None,
        help="Optional cap on representatives, limiting downstream Jaguar jobs.",
    )
    folder_group = parser.add_mutually_exclusive_group()
    folder_group.add_argument(
        "--write-isomer-dirs",
        action="store_true",
        help="Force writing per-isomer folders under output-dir/isomers.",
    )
    folder_group.add_argument(
        "--no-isomer-dirs",
        action="store_true",
        help="Disable per-isomer folders even when control_params has isomer_run=2.",
    )
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run isomer representative detection."""
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS, aliases=COMMAND_ALIASES)
    write_isomer_dirs = None
    if bool(getattr(args, "write_isomer_dirs", False)):
        write_isomer_dirs = True
    if bool(getattr(args, "no_isomer_dirs", False)):
        write_isomer_dirs = False

    result = detect_isomer_representatives_from_reaxff_files(
        fort7_path=args.fort7,
        xmolout_path=args.xmolout,
        control_path=args.control,
        output_dir=args.output_dir,
        write_isomer_dirs=write_isomer_dirs,
        max_representatives=getattr(args, "max_representatives", None),
    )
    print(
        f"{canonical}: detected {len(result.detection.records)} isomer representatives; "
        f"wrote {result.output_xmolout_isomers}"
    )
    if result.isomer_dir is not None:
        print(f"Per-isomer xmolout folders: {result.isomer_dir}")
    print(f"Log: {result.log_path}")
    return 0


__all__ = [
    "ALL_COMMANDS",
    "ALL_LEGACY_COMMANDS",
    "IsomerRepresentativeFileWorkflowResult",
    "LegacyIsomerRepresentativeControl",
    "build_parser",
    "detect_isomer_representatives_from_reaxff_files",
    "parse_legacy_isomer_representative_control",
    "run_main",
]
