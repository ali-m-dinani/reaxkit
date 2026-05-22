"""Direct command workflows for GEO and structure file utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from reaxkit.core.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.engine.common.io.geo_io import read_structure, write_structure
from reaxkit.engine.common.geo_transforms import (
    build_surface,
    make_supercell,
    orthogonalize_hexagonal_cell,
    place2,
)
from reaxkit.engine.reaxff.generators.geo_generator import (
    add_restraints_to_geo,
    sort_geo,
    xtob,
)

GEO_FILE_TOOL_COMMANDS = (
    "xtob",
    "make-geo",
    "sort_geo",
    "orthogonalize-geo",
    "place-geo",
    "add_restraints_to_geo",
    "add-geo-restraint",
)


def _parse_csv_floats(value: str, expected: int, name: str) -> list[float]:
    parts = [v.strip() for v in value.split(",") if v.strip()]
    if len(parts) != expected:
        raise argparse.ArgumentTypeError(
            f"{name} must contain exactly {expected} comma-separated values, got {len(parts)} from {value!r}."
        )
    try:
        return [float(v) for v in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be numeric, could not parse {value!r}.") from exc


def _parse_csv_ints(value: str, expected: int, name: str) -> list[int]:
    parts = [v.strip() for v in value.split(",") if v.strip()]
    if len(parts) != expected:
        raise argparse.ArgumentTypeError(
            f"{name} must contain exactly {expected} comma-separated values, got {len(parts)} from {value!r}."
        )
    try:
        return [int(v) for v in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be integers, could not parse {value!r}.") from exc


def _run_xtob(args: argparse.Namespace) -> int:
    xyz_path = Path(args.file)
    if not xyz_path.is_file():
        raise FileNotFoundError(f"Input XYZ file not found: {xyz_path}")

    xtob(
        xyz_file=xyz_path,
        geo_file=args.output,
        box_lengths=_parse_csv_floats(args.dims, expected=3, name="--dims"),
        box_angles=_parse_csv_floats(args.angles, expected=3, name="--angles"),
        sort_by=args.sort,
        ascending=not args.descending,
    )
    print(f"[Done] Converted {xyz_path} to {args.output}")
    return 0


def _run_make_geo(args: argparse.Namespace) -> int:
    in_path = Path(args.file)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input structure file not found: {in_path}")

    miller = _parse_csv_ints(args.surface, expected=3, name="--surface")
    expand = _parse_csv_ints(args.expand, expected=3, name="--expand")
    nx, ny, layers = expand

    bulk = read_structure(in_path)
    slab = build_surface(
        bulk,
        miller=tuple(miller),
        layers=layers,
        vacuum=float(args.vacuum),
        center=True,
    )
    slab_expanded = make_supercell(slab, (nx, ny, 1))
    write_structure(slab_expanded, args.output)
    print(f"[Done] Built surface {tuple(miller)} with layers={layers}, expanded ({nx}, {ny}, 1) and wrote {args.output}")
    return 0


def _run_sort_geo(args: argparse.Namespace) -> int:
    out_path = sort_geo(
        input_geo=args.file,
        output_geo=args.output,
        sort_by=args.sort,
        descending=bool(args.descending),
    )
    in_path = Path(args.file)
    direction = "descending" if args.descending else "ascending"
    sort_label_map = {"m": "atom index", "x": "x-coordinate", "y": "y-coordinate", "z": "z-coordinate", "atom_type": "atom type"}
    sort_label = sort_label_map[args.sort]
    print(f"[Done] Sorted {in_path} by {sort_label} ({direction}) to {out_path}")
    return 0


def _run_orthogonalize_geo(args: argparse.Namespace) -> int:
    in_path = Path(args.file)
    out_path = Path(args.output)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input structure file not found: {in_path}")

    atoms = read_structure(in_path)
    ortho_atoms = orthogonalize_hexagonal_cell(atoms)
    write_structure(ortho_atoms, out_path)
    print(f"[Done] Converted hexagonal to orthorhombic: {in_path} to {out_path}")
    return 0


def _run_place_geo(args: argparse.Namespace) -> int:
    insert_path = Path(args.insert)
    if not insert_path.is_file():
        raise FileNotFoundError(f"Insert molecule not found: {insert_path}")

    base_path = None
    if args.base is not None:
        base_path = Path(args.base)
        if not base_path.is_file():
            raise FileNotFoundError(f"Base structure not found: {base_path}")

    dims = _parse_csv_floats(args.dims, expected=3, name="--dims")
    angles = _parse_csv_floats(args.angles, expected=3, name="--angles")
    a, b, c = dims
    alpha, beta, gamma = angles

    atoms = place2(
        insert_molecule=insert_path,
        base_structure=base_path,
        n_copies=int(args.ncopy),
        box_length_x=a,
        box_length_y=b,
        box_length_z=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        min_interatomic_distance=float(args.mindist),
        base_structure_placement_mode=args.baseplace,
        max_placement_attempts_per_copy=int(args.maxattempt),
        random_seed=None if args.randomseed is None else int(args.randomseed),
    )

    out_path = Path(args.output)
    if out_path.suffix.lower() == ".xyz":
        write_structure(atoms, out_path)
        print(f"[Done] Placed {args.ncopy} copies into box to {out_path}")
        return 0

    tmp_xyz = out_path.parent / "place2_output.xyz"
    write_structure(atoms, tmp_xyz)
    xtob(
        xyz_file=tmp_xyz,
        geo_file=out_path,
        box_lengths=dims,
        box_angles=angles,
        sort_by=None,
        ascending=True,
    )
    print(f"[Done] Placed {args.ncopy} copies into box to {tmp_xyz} and converted to {out_path}")
    return 0


def _run_add_geo_restraint(args: argparse.Namespace) -> int:
    in_path = Path(args.file)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input GEO file not found: {in_path}")

    out_path = Path(args.output) if args.output else (in_path.parent / f"{in_path.name}_with_restraints")
    params = {}
    kinds = []
    for key, label in (("bond", "BOND"), ("angle", "ANGLE"), ("torsion", "TORSION"), ("mascen", "MASCEN")):
        value = getattr(args, key)
        if value is not None:
            kinds.append(label)
            params[label] = value.strip()
    if not kinds:
        raise ValueError("No restraints requested. Provide at least one of: --bond, --angle, --torsion, --mascen")

    out_written = add_restraints_to_geo(in_path, out_file=out_path, kinds=kinds, params=params)
    print(f"[Done] Added restraints to {in_path} and exported {out_written}")
    return 0


RUNNERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "xtob": _run_xtob,
    "make-geo": _run_make_geo,
    "sort_geo": _run_sort_geo,
    "orthogonalize-geo": _run_orthogonalize_geo,
    "place-geo": _run_place_geo,
    "add_restraints_to_geo": _run_add_geo_restraint,
    "add-geo-restraint": _run_add_geo_restraint,
}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    parser.set_defaults(command=command)
    parser.formatter_class = argparse.RawTextHelpFormatter

    if command == "xtob":
        parser.description = (
            "Convert an XYZ file to ReaxFF GEO format.\n\n"
            "Examples:\n"
            "  reaxkit xtob --file slab.xyz --dims 11.0,12.0,100.0 --angles 90,90,90 --output geo\n"
            "  reaxkit xtob --file slab.xyz --dims 11,12,100 --angles 90,90,90 --sort z --output slab_geo"
        )
        parser.add_argument("--file", required=True, help="Input XYZ file")
        parser.add_argument("--dims", required=True, help="Box dimensions a,b,c")
        parser.add_argument("--angles", required=True, help="Box angles alpha,beta,gamma")
        parser.add_argument("--output", default="geo", help="Output GEO file")
        parser.add_argument("--sort", choices=["x", "y", "z", "atom_type"], help="Sort atoms before writing")
        parser.add_argument("--descending", action="store_true", help="Sort in descending order")
    elif command == "make-geo":
        parser.description = (
            "Build a surface slab from a bulk structure and write it to an ASE-supported format.\n\n"
            "Examples:\n"
            "  reaxkit make-geo --file AlN.cif --output slab.xyz --surface 1,0,0 --expand 4,4,6 --vacuum 15\n"
        )
        parser.add_argument("--file", required=True, help="Input bulk structure file")
        parser.add_argument("--output", required=True, help="Output file")
        parser.add_argument("--surface", required=True, help="Miller indices h,k,l")
        parser.add_argument("--expand", required=True, help="Supercell and layers nx,ny,layers")
        parser.add_argument("--vacuum", required=True, help="Vacuum thickness in angstrom")
    elif command == "sort_geo":
        parser.description = (
            "Sort atoms in a GEO file and write a new GEO file.\n\n"
            "Examples:\n"
            "  reaxkit sort_geo --file geo --output sorted_geo --sort x\n"
            "  reaxkit sort_geo --file geo --output sorted_geo --sort atom_type --descending"
        )
        parser.add_argument("--file", required=True, help="Input GEO file")
        parser.add_argument("--output", required=True, help="Output GEO file")
        parser.add_argument("--sort", required=True, choices=["m", "x", "y", "z", "atom_type"], help="Sort key")
        parser.add_argument("--descending", action="store_true", help="Sort in descending order")
    elif command == "orthogonalize-geo":
        parser.description = (
            "Convert a hexagonal cell into an orthorhombic cell.\n\n"
            "Examples:\n"
            "  reaxkit orthogonalize-geo --file AlN.cif --output AlN_ortho.cif"
        )
        parser.add_argument("--file", required=True, help="Input structure file")
        parser.add_argument("--output", required=True, help="Output structure file")
    elif command == "place-geo":
        parser.description = (
            "Randomly place copies of a molecule into a box, optionally around a base structure.\n\n"
            "Examples:\n"
            "  reaxkit place-geo --insert template.xyz --ncopy 40 --dims 28.8,33.27,60 --angles 90,90,90 --output placed.xyz\n"
            "  reaxkit place-geo --insert template.xyz --ncopy 40 --dims 28.8,33.27,60 --angles 90,90,90 --base slab.xyz --output placed_geo"
        )
        parser.add_argument("--insert", required=True, help="Insert molecule")
        parser.add_argument("--ncopy", required=True, help="Number of copies to place")
        parser.add_argument("--dims", required=True, help="Box dimensions a,b,c")
        parser.add_argument("--angles", required=True, help="Box angles alpha,beta,gamma")
        parser.add_argument("--output", required=True, help="Output file")
        parser.add_argument("--base", help="Optional base structure")
        parser.add_argument("--mindist", default=2.0, help="Minimum interatomic distance")
        parser.add_argument("--baseplace", default="as-is", choices=["as-is", "center", "origin"], help="Base placement mode")
        parser.add_argument("--maxattempt", default=50000, help="Maximum placement attempts per copy")
        parser.add_argument("--randomseed", default=None, help="Random seed")
    elif command in {"add_restraints_to_geo", "add-geo-restraint"}:
        parser.description = (
            "Insert sample or explicit restraint blocks into a GEO file.\n\n"
            "Examples:\n"
            "  reaxkit add_restraints_to_geo --file geo --bond --output geo_r\n"
            "  reaxkit add_restraints_to_geo --file geo --angle '1 2 3 109.5000 600.00 0.25000 0.0000000' --output geo_r"
        )
        parser.add_argument("--file", default="geo", help="Input GEO file")
        parser.add_argument("--output", default=None, help="Output GEO file")
        parser.add_argument("--bond", nargs="?", const="", default=None, help="Add one bond restraint")
        parser.add_argument("--angle", nargs="?", const="", default=None, help="Add one angle restraint")
        parser.add_argument("--torsion", nargs="?", const="", default=None, help="Add one torsion restraint")
        parser.add_argument("--mascen", nargs="?", const="", default=None, help="Add one mass-center restraint")
    else:
        raise KeyError(f"Unsupported GEO file-tool command {command!r}.")

    parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")
    add_storage_cli_arguments(parser)
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    output_value = getattr(args, "output", None)
    if command in {"add_restraints_to_geo", "add-geo-restraint"} and not output_value:
        output_value = f"{Path(args.file).name}_with_restraints"
    if not output_value:
        output_value = command
    out_path, layout = prepare_generator_output(args, command=command, output_value=str(output_value))
    args.output = str(out_path)
    code = RUNNERS[command](args)
    if code != 0:
        return code
    persist_generator_metadata(
        args,
        command=command,
        output_path=out_path,
        layout=layout,
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
    )
    copied = maybe_copy_output_to_dot(out_path, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [out_path.parent]
    if copied is not None:
        dirs.append(copied.parent)
    print_saved_dirs(dirs)
    return 0
