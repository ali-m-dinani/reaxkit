"""Create ReaxFF trainset inputs from completed isomer jobs.

This helper extracts final DFT energies and optimized geometries from completed
isomer representative jobs and writes the legacy ``geo``, ``trainset.in``,
``composition.txt``, and log files used by the downstream ReaxFF training
workflow. The current parser supports Jaguar ``hf.out`` outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re


HARTREE_TO_KCAL_PER_MOL = 627.509
DEFAULT_REFERENCE_COMPOSITION = 100.0
TOTAL_ENERGY_RE = re.compile(r"^\s*Total energy:\s*(\S+)\s+hartrees\s*$")
FINAL_GEOMETRY_RE = re.compile(r"^\s*final geometry:\s*$")
ATOM_LINE_RE = re.compile(
    r"^\s*([A-Za-z]+)\d*\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*$"
)


@dataclass(frozen=True)
class IsomerTrainsetAtom:
    """One atom parsed from a supported final geometry block."""

    element: str
    x: str
    y: str
    z: str


@dataclass(frozen=True)
class IsomerTrainsetRecord:
    """One completed isomer output included in trainset generation."""

    structure_name: str
    hf_output: Path
    energy_hartree: float
    energy_kcal: float
    atoms: tuple[IsomerTrainsetAtom, ...]
    order: int
    composition: float = 0.0


@dataclass(frozen=True)
class IsomerTrainsetSkippedRecord:
    """One isomer output skipped during trainset generation."""

    structure_name: str
    hf_output: Path
    reason: str


@dataclass(frozen=True)
class IsomerTrainsetResult:
    """Result for Isomer trainset generation."""

    records: list[IsomerTrainsetRecord]
    skipped: list[IsomerTrainsetSkippedRecord]
    geo_path: Path
    trainset_path: Path
    composition_path: Path
    log_path: Path


def _structure_order(structure_name: str) -> int:
    """Return the integer suffix used by the legacy isomer folder names."""
    suffix = structure_name.rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return 0


def parse_isomer_hf_output(path: str | Path, *, structure_name: str | None = None) -> IsomerTrainsetRecord:
    """Parse total energy and final geometry from one supported ``hf.out`` file."""
    hf_output = Path(path)
    name = structure_name or hf_output.parent.name
    if not hf_output.is_file():
        raise FileNotFoundError(f"hf.out file not found: {hf_output}")

    energy_hartree: float | None = None
    atoms: list[IsomerTrainsetAtom] = []
    in_final_geometry = False

    for raw_line in hf_output.read_text(encoding="utf-8", errors="replace").splitlines():
        if in_final_geometry:
            if not raw_line.strip():
                if atoms:
                    break
                continue
            atom_match = ATOM_LINE_RE.match(raw_line)
            if atom_match:
                atoms.append(
                    IsomerTrainsetAtom(
                        element=atom_match.group(1),
                        x=atom_match.group(2),
                        y=atom_match.group(3),
                        z=atom_match.group(4),
                    )
                )
            continue

        energy_match = TOTAL_ENERGY_RE.match(raw_line)
        if energy_match:
            energy_hartree = float(energy_match.group(1))
            continue
        if FINAL_GEOMETRY_RE.match(raw_line):
            in_final_geometry = True

    if not in_final_geometry or not atoms:
        raise ValueError(f"{name} does not have final geometry")
    if energy_hartree is None:
        raise ValueError(f"{name} does not have a Total energy before final geometry")

    return IsomerTrainsetRecord(
        structure_name=name,
        hf_output=hf_output,
        energy_hartree=energy_hartree,
        energy_kcal=energy_hartree * HARTREE_TO_KCAL_PER_MOL,
        atoms=tuple(atoms),
        order=_structure_order(name),
    )


def discover_isomer_hf_outputs(job_dir: str | Path, *, hf_output_name: str = "hf.out") -> list[tuple[str, Path]]:
    """Return sorted ``(structure_name, hf_output_path)`` pairs from isomer job folders."""
    root = Path(job_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"isomer job directory not found: {root}")
    subdirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not subdirs:
        raise ValueError(f"isomer job directory contains no isomer subfolders: {root}")
    return [(path.name, path / hf_output_name) for path in subdirs]


def _format_composition(value: float) -> str:
    """Format relative composition values compactly."""
    return f"{value:.6g}"


def _with_compositions(
    records_by_energy: list[IsomerTrainsetRecord],
    *,
    reference_composition: float,
    temperature: float,
    gas_constant: float,
) -> list[IsomerTrainsetRecord]:
    """Return records with legacy relative composition values assigned."""
    reference_energy = records_by_energy[0].energy_kcal
    with_composition: list[IsomerTrainsetRecord] = []
    for index, record in enumerate(records_by_energy):
        if index == 0:
            composition = reference_composition
        else:
            composition = math.exp((-1.0 * (record.energy_kcal - reference_energy) * 1000.0) / (temperature * gas_constant))
        with_composition.append(
            IsomerTrainsetRecord(
                structure_name=record.structure_name,
                hf_output=record.hf_output,
                energy_hartree=record.energy_hartree,
                energy_kcal=record.energy_kcal,
                atoms=record.atoms,
                order=record.order,
                composition=composition,
            )
        )
    return with_composition


def render_geo(records: list[IsomerTrainsetRecord]) -> str:
    """Render a concatenated ReaxFF ``geo`` file from parsed final geometries."""
    lines: list[str] = []
    for record in records:
        lines.append(str(len(record.atoms)))
        lines.append(record.structure_name)
        for atom in record.atoms:
            lines.append(f"{atom.element} {atom.x} {atom.y} {atom.z} ")
        lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


def render_trainset(records_by_energy: list[IsomerTrainsetRecord], *, weight: float = 1.0) -> str:
    """Render the ENERGY block for ReaxFF trainset input."""
    if len(records_by_energy) < 2:
        return ""
    reference = records_by_energy[0]
    lines = ["ENERGY"]
    for record in records_by_energy[1:]:
        delta = record.energy_kcal - reference.energy_kcal
        lines.append(f" {weight:.1f}  + {record.structure_name}/1    - {reference.structure_name}/1    {delta:.6f}")
    lines.append("ENDENERGY")
    return "\n".join(lines) + "\n"


def render_composition(records_by_composition: list[IsomerTrainsetRecord]) -> str:
    """Render relative composition values."""
    return "".join(f"{record.structure_name} {_format_composition(record.composition)}\n" for record in records_by_composition)


def render_log(records: list[IsomerTrainsetRecord], skipped: list[IsomerTrainsetSkippedRecord]) -> str:
    """Render the trainset-generation log."""
    lines = [f"{record.structure_name} {record.energy_kcal:.6f}" for record in records]
    lines.extend(f"{record.structure_name} {record.reason}" for record in skipped)
    lines.append(f"{len(records)} structures are present in trainset file.")
    return "\n".join(lines) + "\n"


def create_isomer_trainset(
    *,
    job_dir: str | Path,
    output_dir: str | Path,
    hf_output_name: str = "hf.out",
    geo_filename: str = "geo",
    trainset_filename: str = "trainset.in",
    composition_filename: str = "composition.txt",
    log_filename: str = "out_trainset_log.txt",
    weight: float = 1.0,
    reference_composition: float = DEFAULT_REFERENCE_COMPOSITION,
    temperature: float = 273.0,
    gas_constant: float = 1.987,
    require_all_complete: bool = False,
    force: bool = False,
) -> IsomerTrainsetResult:
    """Create trainset artifacts from completed isomer ``hf.out`` files."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if gas_constant <= 0:
        raise ValueError("gas_constant must be positive.")

    pairs = discover_isomer_hf_outputs(job_dir, hf_output_name=hf_output_name)
    out_dir = Path(output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory exists and is non-empty: {out_dir}. Use force=True to overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[IsomerTrainsetRecord] = []
    skipped: list[IsomerTrainsetSkippedRecord] = []
    for structure_name, hf_output in pairs:
        try:
            records.append(parse_isomer_hf_output(hf_output, structure_name=structure_name))
        except (FileNotFoundError, ValueError) as exc:
            skipped.append(
                IsomerTrainsetSkippedRecord(
                    structure_name=structure_name,
                    hf_output=hf_output,
                    reason=str(exc),
                )
            )

    if require_all_complete and skipped:
        names = [record.structure_name for record in skipped]
        raise ValueError(f"not all isomer jobs have completed supported hf.out files: {names}")
    if len(records) < 2:
        raise ValueError("at least two completed supported hf.out files are required to create a trainset.")

    records_by_energy = sorted(records, key=lambda record: record.energy_kcal)
    records_with_composition = _with_compositions(
        records_by_energy,
        reference_composition=reference_composition,
        temperature=temperature,
        gas_constant=gas_constant,
    )
    records_by_composition = sorted(records_with_composition, key=lambda record: record.composition)
    composition_by_name = {record.structure_name: record for record in records_with_composition}
    records_for_geo_and_log = [composition_by_name[record.structure_name] for record in records]

    geo_path = out_dir / geo_filename
    trainset_path = out_dir / trainset_filename
    composition_path = out_dir / composition_filename
    log_path = out_dir / log_filename

    geo_path.write_text(render_geo(records_for_geo_and_log), encoding="utf-8")
    trainset_path.write_text(render_trainset(records_by_energy, weight=weight), encoding="utf-8")
    composition_path.write_text(render_composition(records_by_composition), encoding="utf-8")
    log_path.write_text(render_log(records_for_geo_and_log, skipped), encoding="utf-8")

    return IsomerTrainsetResult(
        records=records_for_geo_and_log,
        skipped=skipped,
        geo_path=geo_path,
        trainset_path=trainset_path,
        composition_path=composition_path,
        log_path=log_path,
    )


__all__ = [
    "DEFAULT_REFERENCE_COMPOSITION",
    "HARTREE_TO_KCAL_PER_MOL",
    "IsomerTrainsetAtom",
    "IsomerTrainsetRecord",
    "IsomerTrainsetResult",
    "IsomerTrainsetSkippedRecord",
    "create_isomer_trainset",
    "discover_isomer_hf_outputs",
    "parse_isomer_hf_output",
    "render_composition",
    "render_geo",
    "render_log",
    "render_trainset",
]
