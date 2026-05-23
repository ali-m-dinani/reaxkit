"""Direct command workflow for merging ReaxFF ``ffield`` files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from reaxkit.core.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.engine.reaxff.generators.ffielld_generator import merge_ffields


def _parse_csv_items(value: str) -> list[str]:
    return [token.strip() for token in str(value).replace(";", ",").split(",") if token.strip()]


def _write_snapshot_text(path: Path, title: str, labels_by_field: dict[str, list[str]], blocks_by_field: dict[str, list[str]]) -> Path:
    names = {
        "atom": "Atom",
        "bond": "Bond",
        "off_diagonal": "Off-diagonal",
        "angle": "Angle",
        "torsion": "Torsion",
        "hbond": "H-bond",
    }
    lines: list[str] = [title, ""]
    for field in ("atom", "bond", "off_diagonal", "angle", "torsion", "hbond"):
        lines.append(f"{names[field]}:")
        lines.append("|")
        labels = labels_by_field.get(field, [])
        lines.append((" | ".join(labels) + " |") if labels else "Not Added")
        lines.append("")
    lines.append("Formatted ffield blocks:")
    lines.append("")
    for field in ("atom", "bond", "off_diagonal", "angle", "torsion", "hbond"):
        lines.append(f"{names[field]}:")
        lines.append("|")
        blocks = blocks_by_field.get(field, [])
        if not blocks:
            lines.append("Not Added")
        else:
            for block in blocks:
                lines.append(block)
                lines.append("")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _write_snapshot_field_csvs(root: Path, labels_by_field: dict[str, list[str]], blocks_by_field: dict[str, list[str]]) -> list[Path]:
    files: list[Path] = []
    for field in ("atom", "bond", "off_diagonal", "angle", "torsion", "hbond"):
        out = root / f"{field}_entries.csv"
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["entry_index", "label", "formatted_block"])
            labels = labels_by_field.get(field, [])
            blocks = blocks_by_field.get(field, [])
            n = max(len(labels), len(blocks))
            for i in range(n):
                label = labels[i] if i < len(labels) else ""
                block = blocks[i] if i < len(blocks) else ""
                writer.writerow([i + 1, label, block])
        files.append(out)
    return files


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Merge selected atom-type parameter blocks from one ffield into another.\n\n"
        "Examples:\n"
        "  reaxkit merge-ffield --src ffield_src --dest ffield_dst --output ffield_merged --atom-types W\n"
        "  reaxkit merge-ffield --source f_src --destination f_dst --output merged --atom-types W,Mo --fields atom,bond,angle,torsion"
    )
    parser.add_argument("--source", "--src", required=True, dest="source", help="Source ffield path")
    parser.add_argument("--destination", "--dest", required=True, dest="destination", help="Destination ffield path")
    parser.add_argument("--output", default="ffield_merged", help="Output merged ffield path")
    parser.add_argument(
        "--atom-types",
        required=True,
        help="Comma-separated source atom symbols to merge, for example: W or W,Mo",
    )
    parser.add_argument(
        "--fields",
        default="atom,bond,off_diagonal,angle,torsion,hbond",
        help="Comma-separated fields to merge: atom,bond,off_diagonal,angle,torsion,hbond",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace destination rows when the same atom tuple already exists.",
    )
    parser.add_argument(
        "--disallow-torsion-wildcard",
        action="store_true",
        help="Reject torsion rows containing atom index 0 wildcard.",
    )
    parser.add_argument(
        "--report-format",
        choices=["none", "txt", "csv", "both"],
        default="both",
        help="Write merge-detail report files next to output ffield.",
    )
    parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")
    add_storage_cli_arguments(parser)
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    out_path, layout = prepare_generator_output(args, command=command, output_value=str(args.output))

    atom_types = _parse_csv_items(args.atom_types)
    fields = _parse_csv_items(args.fields)

    summary = merge_ffields(
        source=args.source,
        destination=args.destination,
        output=out_path,
        atom_types=atom_types,
        fields=fields,
        replace_existing=bool(args.replace_existing),
        allow_torsion_wildcard=not bool(args.disallow_torsion_wildcard),
    )

    print(f"[Done] Merged ffield written to {summary.output_path}")
    print(f"  atom types: {', '.join(summary.atom_types_merged)}")
    print(f"  fields: {', '.join(summary.fields)}")
    print(f"  appended: {summary.appended}")
    if any(summary.updated.values()):
        print(f"  updated: {summary.updated}")
    if any(summary.skipped_existing.values()):
        print(f"  skipped_existing: {summary.skipped_existing}")
    if any(summary.skipped_incompatible.values()):
        print(f"  skipped_incompatible: {summary.skipped_incompatible}")

    report_paths: list[Path] = []
    base = Path(summary.output_path)
    report_root = base.parent / f"{base.stem}_merge_audit"
    source_root = report_root / "source_snapshot"
    destination_root = report_root / "destination_projection"
    source_root.mkdir(parents=True, exist_ok=True)
    destination_root.mkdir(parents=True, exist_ok=True)

    if args.report_format in {"txt", "both"}:
        report_paths.append(
            _write_snapshot_text(
                source_root / "source_snapshot_summary.txt",
                "Source snapshot of transferred entries",
                summary.source_labels,
                summary.source_blocks,
            )
        )
        report_paths.append(
            _write_snapshot_text(
                destination_root / "destination_projection_summary.txt",
                "Destination projection of transferred entries",
                summary.destination_labels,
                summary.destination_blocks,
            )
        )
    if args.report_format in {"csv", "both"}:
        report_paths.extend(_write_snapshot_field_csvs(source_root, summary.source_labels, summary.source_blocks))
        report_paths.extend(_write_snapshot_field_csvs(destination_root, summary.destination_labels, summary.destination_blocks))
    for report_path in report_paths:
        print(f"[Done] Merge report written to {report_path}")

    persist_generator_metadata(
        args,
        command=command,
        output_path=summary.output_path,
        layout=layout,
        extra={
            "source": str(args.source),
            "destination": str(args.destination),
            "atom_types": list(summary.atom_types_merged),
            "fields": list(summary.fields),
            "report_format": str(args.report_format),
            "report_paths": [str(p) for p in report_paths],
            "appended": summary.appended,
            "updated": summary.updated,
            "skipped_existing": summary.skipped_existing,
            "skipped_incompatible": summary.skipped_incompatible,
        },
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
    )
    copied = maybe_copy_output_to_dot(summary.output_path, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [summary.output_path.parent]
    if copied is not None:
        dirs.append(copied.parent)
    if bool(getattr(args, "copy_to_dot", False)):
        for report_path in report_paths:
            copied_report = maybe_copy_output_to_dot(report_path, enabled=True)
            if copied_report is not None:
                dirs.append(copied_report.parent)
    print_saved_dirs(dirs)
    return 0
