"""Direct command workflows for ReaxFF ``ffield`` tools."""

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
from reaxkit.engine.reaxff.generators.ffield_generator import (
    add_element_to_ffield,
    add_term_to_ffield,
    merge_ffields,
)


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


def _write_similarity_summary(path: Path, details: dict[str, object]) -> Path:
    lines: list[str] = ["Similarity selection", ""]
    lines.append(f"Mode: {details.get('mode', '')}")
    target = details.get("target", {})
    chosen = details.get("chosen", {})
    if isinstance(target, dict):
        lines.append(f"Target: {target.get('symbol', '')}")
        lines.append(f"  group: {target.get('group', '')}")
        lines.append(f"  family: {target.get('family', '')}")
    if isinstance(chosen, dict):
        lines.append(f"Chosen template: {chosen.get('symbol', '')}")
        lines.append(f"  group: {chosen.get('group', '')}")
        lines.append(f"  family: {chosen.get('family', '')}")
    lines.append("")
    lines.append("Candidates:")
    candidates = details.get("candidates", [])
    if isinstance(candidates, list) and candidates:
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            selected = "*" if bool(candidate.get("selected")) else " "
            lines.append(
                f"{selected} {candidate.get('symbol', '')}: "
                f"group={candidate.get('group', '')}, "
                f"family={candidate.get('family', '')}, "
                f"radius_distance={candidate.get('radius_distance', '')}"
            )
        lines.append("")
        lines.append(
            "radius_distance = mean absolute difference across the selected radius metrics "
            "(metrics with missing values are ignored; if all are missing, distance is infinite)."
        )
    else:
        lines.append("No candidate diagnostics available.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _write_similarity_summaries(path: Path, details_by_atom: dict[str, dict[str, object]]) -> Path:
    lines: list[str] = ["Similarity selection (merge template fill)", ""]
    if not details_by_atom:
        lines.append("No similarity details available.")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return path
    for atom in sorted(details_by_atom.keys()):
        details = details_by_atom.get(atom, {})
        lines.append(f"Target atom: {atom}")
        lines.append(f"Mode: {details.get('mode', '')}")
        target = details.get("target", {})
        chosen = details.get("chosen", {})
        if isinstance(target, dict):
            lines.append(f"  target symbol: {target.get('symbol', '')}")
            lines.append(f"  target group: {target.get('group', '')}")
            lines.append(f"  target family: {target.get('family', '')}")
        if isinstance(chosen, dict):
            lines.append(f"  chosen template: {chosen.get('symbol', '')}")
            lines.append(f"  chosen group: {chosen.get('group', '')}")
            lines.append(f"  chosen family: {chosen.get('family', '')}")
        lines.append("  candidates:")
        candidates = details.get("candidates", [])
        if isinstance(candidates, list) and candidates:
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                selected = "*" if bool(candidate.get("selected")) else " "
                lines.append(
                    f"  {selected} {candidate.get('symbol', '')}: "
                    f"group={candidate.get('group', '')}, "
                    f"family={candidate.get('family', '')}, "
                    f"radius_distance={candidate.get('radius_distance', '')}"
                )
        else:
            lines.append("  No candidate diagnostics available.")
        lines.append("")
    lines.append(
        "radius_distance = mean absolute difference across the selected radius metrics "
        "(metrics with missing values are ignored; if all are missing, distance is infinite)."
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.set_defaults(command=command)

    if command in {"add-element-to-ffield", "add_element_to_ffield"}:
        parser.description = (
            "Add one atom type to an existing ffield and assigning parameters of a very similar atom in the "
            "ffield as a template. This is intended as a quick way to expand coverage of an existing ffield "
            "by cloning the most similar existing atom terms.\n\n"
            "Examples:\n"
            "  reaxkit add-element-to-ffield --dest ffield --element Al --output ffield_al\n"
            "  reaxkit add-element-to-ffield --dest ffield --element Al --similarity group --closest-atom B --fields atom,bond,angle"
        )
        parser.add_argument("--destination", "--dest", required=True, dest="destination", help="Destination ffield path")
        parser.add_argument("--output", default="ffield_with_element", help="Output expanded ffield path")
        parser.add_argument("--element", required=True, help="Element symbol to add, for example: Al")
        parser.add_argument(
            "--similarity",
            default="group",
            choices=["family", "group", "radius"],
            help=(
                "Similarity rule for template-atom selection: "
                "'family' = chemical family match "
                "(transition_metal, lanthanoid, actinoid, alkali_metal, alkaline_earth_metal, "
                "halogen, noble_gas, metalloid, post_transition_metal, other), "
                "'group' = same periodic-table group number (column), "
                "'radius' = closest by atomic/covalent-proxy/van-der-Waals radii distance."
            ),
        )
        parser.add_argument(
            "--radius-metrics",
            default="atomic_radius,covalent_radius,van_der_waals_radius",
            help=(
                "Comma-separated radius metrics used when --similarity radius is selected. "
                "Use 'all' to include every supported metric. "
                "Options: atomic_radius (empirical neutral-atom radius), "
                "covalent_radius (mapped to pymatgen atomic_radius_calculated proxy), "
                "van_der_waals_radius (non-bonded contact radius), "
                "atomic_radius_calculated (theoretical neutral-atom radius), "
                "average_ionic_radius (mean ionic radius over known oxidation states), "
                "average_cationic_radius (mean radius over positive oxidation states), "
                "average_anionic_radius (mean radius over negative oxidation states)."
            ),
        )
        parser.add_argument(
            "--closest-atom",
            default=None,
            help="Override automatic selection and force template atom symbol, for example: B",
        )
        parser.add_argument(
            "--replace-existing",
            action="store_true",
            help="Replace destination rows when the same atom tuple already exists.",
        )
    elif command in {"add-term-to-ffield", "add_term_to_ffield"}:
        parser.description = (
            "Add one specific missing term (bond/off_diagonal/angle/torsion/hbond) to an existing ffield by "
            "copying parameters from a similar existing template term.\n\n"
            "Examples:\n"
            "  reaxkit add-term-to-ffield --dest ffield --field angle --term Al-N-Al --output ffield_with_term\n"
            "  reaxkit add-term-to-ffield --dest ffield --field angle --term Al-N-Al --template-map Al:B"
        )
        parser.add_argument("--destination", "--dest", required=True, dest="destination", help="Destination ffield path")
        parser.add_argument("--output", default="ffield_with_term", help="Output expanded ffield path")
        parser.add_argument(
            "--field",
            required=True,
            choices=["bond", "off_diagonal", "angle", "torsion", "hbond"],
            help="Target section for the term.",
        )
        parser.add_argument(
            "--term",
            required=True,
            help="Hyphen-separated atom symbols, for example: Al-N-Al",
        )
        parser.add_argument(
            "--template-map",
            default="",
            help="Optional per-atom manual template mapping CSV, for example: Al:B,N:N",
        )
        parser.add_argument(
            "--similarity",
            default="group",
            choices=["family", "group", "radius"],
            help="Automatic similarity rule for template-atom selection.",
        )
        parser.add_argument(
            "--radius-metrics",
            default="atomic_radius,covalent_radius,van_der_waals_radius",
            help=(
                "Comma-separated radius metrics used when --similarity radius is selected. "
                "Use 'all' to include every supported metric."
            ),
        )
        parser.add_argument(
            "--replace-existing",
            action="store_true",
            help="Replace destination row when the same atom tuple already exists.",
        )
    else:
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
            "--replace-existing",
            action="store_true",
            help="Replace destination rows when the same atom tuple already exists.",
        )
        parser.add_argument(
            "--fill-missing-with-template",
            action="store_true",
            help="After direct merge, fill missing terms for merged atom-types by templating from the most similar atom in destination.",
        )
        parser.add_argument(
            "--template-similarity",
            default="group",
            choices=["family", "group", "radius"],
            help="Similarity rule for selecting destination template atom when --fill-missing-with-template is enabled.",
        )
        parser.add_argument(
            "--template-closest-atom",
            default=None,
            help="Manual destination template atom override for --fill-missing-with-template, for example: B",
        )
        parser.add_argument(
            "--template-radius-metrics",
            default="atomic_radius,covalent_radius,van_der_waals_radius",
            help=(
                "Radius metrics for template selection when --template-similarity radius. "
                "Use 'all' or a CSV subset of: atomic_radius,covalent_radius,van_der_waals_radius,"
                "atomic_radius_calculated,average_ionic_radius,average_cationic_radius,average_anionic_radius."
            ),
        )
    parser.add_argument(
        "--fields",
        default="atom,bond,off_diagonal,angle,torsion,hbond",
        help="Comma-separated fields to process: atom,bond,off_diagonal,angle,torsion,hbond",
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
    fields = _parse_csv_items(args.fields)

    report_paths: list[Path] = []
    if command in {"add-element-to-ffield", "add_element_to_ffield"}:
        summary = add_element_to_ffield(
            destination=args.destination,
            output=out_path,
            element=args.element,
            fields=fields,
            similarity_mode=str(args.similarity),
            closest_atom=args.closest_atom,
            radius_metrics=_parse_csv_items(args.radius_metrics),
            replace_existing=bool(args.replace_existing),
            allow_torsion_wildcard=not bool(args.disallow_torsion_wildcard),
        )

        print(f"[Done] Expanded ffield written to {summary.output_path}")
        print(f"  element: {summary.element}")
        print(f"  template atom: {summary.template_atom}")
        print(f"  similarity: {summary.similarity_mode}")
        print(f"  fields: {', '.join(summary.fields)}")
        print(f"  appended: {summary.appended}")
        if any(summary.updated.values()):
            print(f"  updated: {summary.updated}")
        if any(summary.skipped_existing.values()):
            print(f"  skipped_existing: {summary.skipped_existing}")
        if any(summary.skipped_incompatible.values()):
            print(f"  skipped_incompatible: {summary.skipped_incompatible}")

        base = Path(summary.output_path)
        report_root = base.parent / f"{base.stem}_add_element_audit"
        template_root = report_root / "template_snapshot"
        destination_root = report_root / "destination_projection"
        template_root.mkdir(parents=True, exist_ok=True)
        destination_root.mkdir(parents=True, exist_ok=True)

        if args.report_format in {"txt", "both"}:
            report_paths.append(_write_similarity_summary(report_root / "similarity_summary.txt", summary.similarity_details))
            report_paths.append(
                _write_snapshot_text(
                    template_root / "template_snapshot_summary.txt",
                    f"Template snapshot of entries projected from {summary.template_atom}",
                    summary.template_labels,
                    summary.template_blocks,
                )
            )
            report_paths.append(
                _write_snapshot_text(
                    destination_root / "destination_projection_summary.txt",
                    f"Destination projection of entries added for {summary.element}",
                    summary.destination_labels,
                    summary.destination_blocks,
                )
            )
        if args.report_format in {"csv", "both"}:
            report_paths.extend(_write_snapshot_field_csvs(template_root, summary.template_labels, summary.template_blocks))
            report_paths.extend(_write_snapshot_field_csvs(destination_root, summary.destination_labels, summary.destination_blocks))
        for report_path in report_paths:
            print(f"[Done] Add-element report written to {report_path}")

        persist_generator_metadata(
            args,
            command=command,
            output_path=summary.output_path,
            layout=layout,
            extra={
                "destination": str(args.destination),
                "element": summary.element,
                "template_atom": summary.template_atom,
                "similarity_mode": summary.similarity_mode,
                "similarity_details": summary.similarity_details,
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
    elif command in {"add-term-to-ffield", "add_term_to_ffield"}:
        template_map: dict[str, str] = {}
        for item in _parse_csv_items(getattr(args, "template_map", "")):
            if ":" not in item:
                continue
            key, value = item.split(":", 1)
            k = key.strip()
            v = value.strip()
            if k and v:
                template_map[k] = v

        summary = add_term_to_ffield(
            destination=args.destination,
            output=out_path,
            field=str(args.field),
            term=str(args.term),
            template_atom_map=template_map if template_map else None,
            similarity_mode=str(args.similarity),
            radius_metrics=_parse_csv_items(args.radius_metrics),
            replace_existing=bool(args.replace_existing),
        )
        print(f"[Done] Updated ffield written to {summary.output_path}")
        print(f"  field: {summary.field}")
        print(f"  term: {summary.term}")
        print(f"  template term: {summary.template_term}")
        print(f"  similarity: {summary.similarity_mode}")
        print(f"  template atoms: {summary.template_atoms}")
        print(f"  appended: {summary.appended}")
        if summary.updated:
            print(f"  updated: {summary.updated}")
        if summary.skipped_existing:
            print(f"  skipped_existing: {summary.skipped_existing}")

        persist_generator_metadata(
            args,
            command=command,
            output_path=summary.output_path,
            layout=layout,
            extra={
                "destination": str(args.destination),
                "field": summary.field,
                "term": summary.term,
                "template_term": summary.template_term,
                "template_atoms": summary.template_atoms,
                "similarity_mode": summary.similarity_mode,
                "similarity_details": summary.similarity_details,
                "appended": summary.appended,
                "updated": summary.updated,
                "skipped_existing": summary.skipped_existing,
            },
            copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
        )
    else:
        atom_types = _parse_csv_items(args.atom_types)
        summary = merge_ffields(
            source=args.source,
            destination=args.destination,
            output=out_path,
            atom_types=atom_types,
            fields=fields,
            replace_existing=bool(args.replace_existing),
            allow_torsion_wildcard=not bool(args.disallow_torsion_wildcard),
            fill_missing_with_template=bool(getattr(args, "fill_missing_with_template", False)),
            template_similarity_mode=str(getattr(args, "template_similarity", "group")),
            template_closest_atom=getattr(args, "template_closest_atom", None),
            template_radius_metrics=_parse_csv_items(getattr(args, "template_radius_metrics", "")),
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
        if any(summary.template_generated.values()):
            print(f"  template_generated: {summary.template_generated}")
        if summary.template_choices:
            print(f"  template_choices: {summary.template_choices}")

        base = Path(summary.output_path)
        report_root = base.parent / f"{base.stem}_merge_audit"
        source_root = report_root / "source_snapshot"
        template_root = report_root / "template_snapshot"
        skipped_root = report_root / "skipped_existing"
        destination_root = report_root / "destination_projection"
        source_root.mkdir(parents=True, exist_ok=True)
        template_root.mkdir(parents=True, exist_ok=True)
        skipped_root.mkdir(parents=True, exist_ok=True)
        destination_root.mkdir(parents=True, exist_ok=True)

        if args.report_format in {"txt", "both"}:
            if bool(getattr(args, "fill_missing_with_template", False)):
                report_paths.append(
                    _write_similarity_summaries(
                        report_root / "similarity_summary.txt",
                        summary.template_similarity_details,
                    )
                )
            report_paths.append(
                _write_snapshot_text(
                    source_root / "source_snapshot_summary.txt",
                    "Source snapshot of transferred entries",
                    summary.source_labels,
                    summary.source_blocks,
                )
            )
            if bool(getattr(args, "fill_missing_with_template", False)):
                report_paths.append(
                    _write_snapshot_text(
                        template_root / "template_snapshot_summary.txt",
                        "Template snapshot of destination entries used for template generation",
                        summary.template_labels,
                        summary.template_blocks,
                    )
                )
            if any(summary.skipped_existing.values()):
                report_paths.append(
                    _write_snapshot_text(
                        skipped_root / "skipped_existing_summary.txt",
                        "Entries skipped because matching rows already existed in destination",
                        summary.skipped_existing_labels,
                        summary.skipped_existing_blocks,
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
            if bool(getattr(args, "fill_missing_with_template", False)):
                report_paths.extend(_write_snapshot_field_csvs(template_root, summary.template_labels, summary.template_blocks))
            if any(summary.skipped_existing.values()):
                report_paths.extend(_write_snapshot_field_csvs(skipped_root, summary.skipped_existing_labels, summary.skipped_existing_blocks))
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
                "template_generated": summary.template_generated,
                "template_choices": summary.template_choices,
                "template_similarity_details": summary.template_similarity_details,
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
