"""Split a force-field optimization GEO collection into individual structures."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from reaxkit.core.resolve.command_alias_resolver import resolve_command_name

ALL_COMMANDS = ("get_force_field_opt_geo_files",)
ALL_LEGACY_COMMANDS = ("get-force-field-opt-geo-files",)
COMMAND_ALIASES = {ALL_COMMANDS[0]: list(ALL_LEGACY_COMMANDS)}

_BLOCK_HEADERS = {"BIOGRF", "XTLGRF"}
_INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


@dataclass(frozen=True)
class ExtractedGeometry:
    """Paths and basic metadata for one extracted geometry."""

    identifier: str
    directory: Path
    geo_path: Path
    xyz_path: Path
    atom_count: int
    occurrence: int = 1


def _canonical_command(command: str) -> str:
    return resolve_command_name(
        command,
        task_names=ALL_COMMANDS,
        aliases=COMMAND_ALIASES,
    )


def _iter_geo_blocks(geo_path: Path) -> Iterator[list[str]]:
    """Yield complete BIOGRF/XTLGRF blocks without loading the whole file."""
    block: list[str] = []
    start_line = 0

    with geo_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            keyword = stripped.split(maxsplit=1)[0].upper() if stripped else ""

            if keyword in _BLOCK_HEADERS:
                if block:
                    raise ValueError(
                        f"Geometry beginning on line {start_line} has no END record "
                        f"before the next geometry on line {line_number}."
                    )
                block = [line]
                start_line = line_number
                continue

            if not block:
                if stripped:
                    raise ValueError(
                        f"Unexpected content before the first BIOGRF/XTLGRF record "
                        f"on line {line_number}: {stripped!r}"
                    )
                continue

            block.append(line)
            if stripped.upper() == "END":
                yield block
                block = []
                start_line = 0

    if block:
        raise ValueError(
            f"Geometry beginning on line {start_line} is incomplete: no END record found."
        )


def _descriptor(block: Sequence[str]) -> str:
    for line in block:
        if line.lstrip().upper().startswith("DESCRP"):
            identifier = line.lstrip()[6:].strip()
            if identifier:
                return identifier
    raise ValueError("A geometry block has no non-empty DESCRP identifier.")


def _atoms(block: Sequence[str], identifier: str) -> list[tuple[str, float, float, float]]:
    atoms: list[tuple[str, float, float, float]] = []
    for line in block:
        stripped = line.lstrip()
        if not (stripped.startswith("ATOM") or stripped.startswith("HETATM")):
            continue
        parts = stripped.split()
        if len(parts) < 6:
            raise ValueError(f"Malformed atom record in geometry {identifier!r}: {line.rstrip()!r}")
        try:
            coordinates = tuple(float(value) for value in parts[3:6])
        except ValueError as exc:
            raise ValueError(
                f"Invalid atom coordinates in geometry {identifier!r}: {line.rstrip()!r}"
            ) from exc
        atoms.append((parts[2], *coordinates))

    if not atoms:
        raise ValueError(f"Geometry {identifier!r} contains no ATOM/HETATM records.")
    return atoms


def _safe_directory_name(identifier: str) -> str:
    name = _INVALID_PATH_CHARS.sub("_", identifier).strip().rstrip(". ")
    if not name or name in {".", ".."}:
        raise ValueError(f"Geometry identifier {identifier!r} cannot be used as a directory name.")
    if name.upper() in _WINDOWS_RESERVED_NAMES:
        name = f"_{name}"
    return name


def _block_text(block: Sequence[str]) -> str:
    text = "".join(block)
    return text if text.endswith(("\n", "\r")) else f"{text}\n"


def _xyz_text(
        identifier: str,
        atoms: Sequence[tuple[str, float, float, float]],
) -> str:
    lines = [str(len(atoms)), identifier]
    lines.extend(
        f"{atom_type:<8} {x: .10f} {y: .10f} {z: .10f}"
        for atom_type, x, y, z in atoms
    )
    return "\n".join(lines) + "\n"


def extract_force_field_opt_geometries(
        geo_path: str | Path,
        output_dir: str | Path,
        *,
        identifiers: Sequence[str] | None = None,
        overwrite: bool = False,
) -> list[ExtractedGeometry]:
    """Extract selected (or all) structures from a multi-geometry GEO file."""
    source = Path(geo_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"GEO file not found: {source}")

    destination = Path(output_dir).expanduser()
    requested = set(identifiers or ())
    found: set[str] = set()
    occurrence_counts: dict[str, int] = {}
    directory_names: dict[str, str] = {}
    results: list[ExtractedGeometry] = []

    for block in _iter_geo_blocks(source):
        identifier = _descriptor(block)
        if requested and identifier not in requested:
            continue
        occurrence = occurrence_counts.get(identifier, 0) + 1
        occurrence_counts[identifier] = occurrence
        base_directory_name = _safe_directory_name(identifier)
        directory_name = (
            base_directory_name
            if occurrence == 1
            else f"{base_directory_name}__{occurrence}"
        )
        previous = directory_names.get(directory_name.casefold())
        if previous is not None:
            raise ValueError(
                f"Geometry identifiers {previous!r} and {identifier!r} map to the same output directory."
            )
        directory_names[directory_name.casefold()] = identifier
        found.add(identifier)

        atoms = _atoms(block, identifier)
        geometry_dir = destination / directory_name
        geo_output = geometry_dir / f"{directory_name}.geo"
        xyz_output = geometry_dir / f"{directory_name}.xyz"
        existing = [path for path in (geo_output, xyz_output) if path.exists()]
        if existing and not overwrite:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Output already exists (use --overwrite to replace it): {joined}")

        geometry_dir.mkdir(parents=True, exist_ok=True)
        geo_output.write_text(_block_text(block), encoding="utf-8", newline="")
        xyz_output.write_text(_xyz_text(identifier, atoms), encoding="utf-8", newline="\n")
        results.append(
            ExtractedGeometry(
                identifier=identifier,
                directory=geometry_dir,
                geo_path=geo_output,
                xyz_path=xyz_output,
                atom_count=len(atoms),
                occurrence=occurrence,
            )
        )

    missing = requested - found
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Requested geometry identifier(s) not found in {source}: {names}")
    if not results:
        raise ValueError(f"No geometry blocks were found in {source}.")
    return results


def build_parser(
        parser: argparse.ArgumentParser,
        *,
        command: str,
) -> argparse.ArgumentParser:
    """Configure the ``get_force_field_opt_geo_files`` command parser."""
    canonical = _canonical_command(command)
    parser.set_defaults(command=canonical)
    parser.description = (
        "Split a force-field optimization GEO file into one folder per DESCRP identifier.\n"
        "Each folder contains <identifier>.geo and <identifier>.xyz.\n\n"
        "Examples:\n"
        "  reaxkit get_force_field_opt_geo_files --geo geo\n"
        "  reaxkit get_force_field_opt_geo_files --geo geo --identifier bulk_e3_mp_2604\n"
        "  reaxkit get_force_field_opt_geo_files --geo geo --output extracted --overwrite"
    )
    parser.add_argument("--geo", default="geo", help="Multi-geometry GEO input file (default: geo)")
    parser.add_argument(
        "--output",
        "--outdir",
        dest="output",
        default="force_field_opt_geo_files",
        help="Output root containing one folder per identifier",
    )
    parser.add_argument(
        "--identifier",
        "--iden",
        action="append",
        dest="identifiers",
        default=[],
        help="Extract only this DESCRP identifier (repeatable; default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace GEO/XYZ files that already exist",
    )
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run the geometry extraction command."""
    _canonical_command(command)
    results = extract_force_field_opt_geometries(
        args.geo,
        args.output,
        identifiers=args.identifiers,
        overwrite=bool(args.overwrite),
    )
    print(f"[Done] Extracted {len(results)} geometries from {Path(args.geo)}")
    print(f"Results saved in:\n  {Path(args.output)}")
    return 0


__all__ = [
    "ALL_COMMANDS",
    "ALL_LEGACY_COMMANDS",
    "ExtractedGeometry",
    "build_parser",
    "extract_force_field_opt_geometries",
    "run_main",
]
