"""Workflow for GEO-related tasks (xtob: XYZ→GEO, make: build slabs)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from reaxkit.io.geo_handler import GeoHandler
from reaxkit.io.geo_generator import (
    xtob,
    read_structure,
    build_surface,
    make_supercell,
    write_structure,
    _format_crystx,
    _format_hetatm_line,
    orthogonalize_hexagonal_cell,
    place2,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _parse_csv_floats(value: str, expected: int, name: str) -> List[float]:
    """
    Parse a comma-separated string of floats and validate length.

    Example: "1,2,3" -> [1.0, 2.0, 3.0]
    """
    parts = [v.strip() for v in value.split(",") if v.strip()]
    if len(parts) != expected:
        raise argparse.ArgumentTypeError(
            f"{name} must contain exactly {expected} comma-separated values, "
            f"got {len(parts)} from {value!r}."
        )
    try:
        return [float(v) for v in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be numeric, could not parse {value!r}."
        ) from exc


def _parse_csv_ints(value: str, expected: int, name: str) -> List[int]:
    """
    Parse a comma-separated string of ints and validate length.

    Example: "1,0,0" -> [1, 0, 0]
    """
    parts = [v.strip() for v in value.split(",") if v.strip()]
    if len(parts) != expected:
        raise argparse.ArgumentTypeError(
            f"{name} must contain exactly {expected} comma-separated values, "
            f"got {len(parts)} from {value!r}."
        )
    try:
        return [int(v) for v in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be integers, could not parse {value!r}."
        ) from exc


# ----------------------------------------------------------------------
# Task 1: xyz -> geo (xtob)
# ----------------------------------------------------------------------

def xtob_task(args: argparse.Namespace) -> int:
    """
    Convert XYZ → GEO (XTLGRF) using reaxkit.io.geo_generator.xtob.
    """
    xyz_path = Path(args.file)

    if not xyz_path.is_file():
        raise FileNotFoundError(f"Input XYZ file not found: {xyz_path}")

    dims = _parse_csv_floats(args.dims, expected=3, name="--dims")
    angles = _parse_csv_floats(args.angles, expected=3, name="--angles")
    output = args.output or "geo"

    xtob(
        xyz_file=xyz_path,
        geo_file=output,
        box_lengths=dims,
        box_angles=angles,
        sort_by=args.sort,
        ascending=not args.descending,
    )

    print(f"[Done] Converted {xyz_path} → {output}")
    return 0


# ----------------------------------------------------------------------
# Task 2: build slab from CIF/POSCAR and write XYZ (make)
# ----------------------------------------------------------------------

def make_task(args: argparse.Namespace) -> int:
    """
    Build a surface slab from a bulk structure and write it to XYZ (or any ASE format).

    CLI:
        reaxkit geo make --file X.cif --output Y.xyz \
            --surface 1,0,0 --expand 4,4,6 --vacuum 15

    Mapping:
        --surface h,k,l     -> miller = (h, k, l)
        --expand nx,ny,l    -> layers = l
                               repetition = (nx, ny, 1)
        --vacuum v          -> vacuum thickness in Å
    """
    in_path = Path(args.file)

    if not in_path.is_file():
        raise FileNotFoundError(f"Input structure file not found: {in_path}")

    # Parse Miller indices and expansion
    miller = _parse_csv_ints(args.surface, expected=3, name="--surface")
    expand = _parse_csv_ints(args.expand, expected=3, name="--expand")
    nx, ny, layers = expand

    vacuum = float(args.vacuum)
    output = args.output or "slab.xyz"

    # 1) Read bulk
    bulk = read_structure(in_path)

    # 2) Build surface slab
    slab = build_surface(
        bulk,
        miller=tuple(miller),
        layers=layers,
        vacuum=vacuum,
        center=True,
    )

    # 3) Expand in-plane only: (nx, ny, 1)
    slab_expanded = make_supercell(slab, (nx, ny, 1))

    # 4) Write output (ASE will infer format from extension, e.g. .xyz)
    write_structure(slab_expanded, output)

    print(
        f"[Done] Built surface {tuple(miller)} with layers={layers}, "
        f"expanded ({nx}, {ny}, 1) and wrote {output}"
    )
    return 0

# ----------------------------------------------------------------------
# Task 3: sort a geo file
# ----------------------------------------------------------------------

def sort_task(args: argparse.Namespace) -> int:
    """
    Sort atoms in a GEO file and write a new GEO file.

    CLI example:
        reaxkit geo sort --file X.geo --output Y.geo --sort m --descending

    where:
        --sort m            → sort by atom index (atom_id)
        --sort x|y|z        → sort by coordinate
        --sort atom_type    → sort by atom type
    """
    in_path = Path(args.file)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input GEO file not found: {in_path}")

    handler = GeoHandler(in_path)
    df = handler.dataframe().copy()
    meta = handler.metadata()

    sort_map = {"m": "atom_id", "x": "x", "y": "y", "z": "z", "atom_type": "atom_type"}
    if args.sort not in sort_map:
        raise ValueError(f"--sort must be one of {list(sort_map.keys())}, got {args.sort!r}")

    sort_col = sort_map[args.sort]
    ascending = not args.descending

    df_sorted = df.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)
    df_sorted["atom_id"] = df_sorted.index + 1  # renumber sequentially

    descriptor = meta.get("descriptor") or ""
    remark = meta.get("remark")
    cell_lengths = meta.get("cell_lengths")
    cell_angles = meta.get("cell_angles")

    out_path = Path(args.output)
    direction = "descending" if args.descending else "ascending"
    sort_label_map = {"m": "atom index", "x": "x-coordinate", "y": "y-coordinate", "z": "z-coordinate", "atom_type": "atom type"}
    sort_label = sort_label_map[args.sort]

    with out_path.open("w") as fh:
        fh.write("XTLGRF 200\n")
        if descriptor:
            fh.write(f"DESCRP  {descriptor}\n")
        if remark:
            fh.write(f"REMARK {remark}\n")
        fh.write(f"REMARK Structure sorted by {sort_label} ({direction})\n")

        if cell_lengths and cell_angles:
            try:
                lengths = [cell_lengths.get(k) for k in ("a", "b", "c")]
                angles = [cell_angles.get(k) for k in ("alpha", "beta", "gamma")]
                if all(v is not None for v in lengths + angles):
                    fh.write(_format_crystx(lengths, angles) + "\n")
            except Exception:
                # If anything is weird, just skip CRYSTX instead of crashing
                pass

        fh.write("FORMAT ATOM   (a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5)\n")
        for row in df_sorted.itertuples(index=False):
            line = _format_hetatm_line(row.atom_id, row.atom_type, row.x, row.y, row.z)
            fh.write(line + "\n")
        fh.write("END\n")

    print(f"[Done] Sorted {in_path} by {sort_label} ({direction}) → {out_path}")
    return 0

# ----------------------------------------------------------------------
# Task 4: convert a hexagonal cell (90°, 90°, 120°) into an orthorhombic (90°, 90°, 90°) cell
# ----------------------------------------------------------------------

def ortho_task(args: argparse.Namespace) -> int:
    """
    Orthogonalize a hexagonal (90,90,120) cell into an orthorhombic (90,90,90) cell.

    CLI example:
        reaxkit geo ortho --file AlN_hex.cif --output AlN_ortho.cif
    """
    in_path = Path(args.file)
    out_path = Path(args.output)

    if not in_path.is_file():
        raise FileNotFoundError(f"Input structure file not found: {in_path}")

    # 1. Read structure using ASE
    atoms = read_structure(in_path)

    # 2. Apply orthogonalization transform
    ortho_atoms = orthogonalize_hexagonal_cell(atoms)

    # 3. Write output (ASE infers format from extension)
    write_structure(ortho_atoms, out_path)

    print(f"[Done] Converted hexagonal → orthorhombic: {in_path} → {out_path}")
    return 0

# ----------------------------------------------------------------------
# task 5: place2 algorithm for placing n instances of a structure into a simulation box or within another structure
# ----------------------------------------------------------------------
def place2_task(args: argparse.Namespace) -> int:
    """
    Randomly place copies of an insert molecule into a simulation box,
    optionally around/within a base structure.

    CLI example:
        reaxkit geo place2 \
            --insert X.xyz \
            --ncopy 40 \
            --dims 1,2,3 \
            --angles 90,90,90 \
            --output Y.xyz \
            [--base base.xyz] \
            [--mindist 3.0] \
            [--baseplace as-is|center|origin] \
            [--maxattempt 50000] \
            [--randomseed 1234]

    If output is:
        - *.xyz → write XYZ directly.
        - geo or *.bgf or anything non-*.xyz →
            1) write place2_output.xyz
            2) run xtob on that to generate requested output.
    """
    insert_path = Path(args.insert)
    if not insert_path.is_file():
        raise FileNotFoundError(f"Insert molecule not found: {insert_path}")

    base_path = None
    if args.base is not None:
        base_path = Path(args.base)
        if not base_path.is_file():
            raise FileNotFoundError(f"Base structure not found: {base_path}")

    # Parse box dimensions and angles
    dims = _parse_csv_floats(args.dims, expected=3, name="--dims")
    angles = _parse_csv_floats(args.angles, expected=3, name="--angles")
    a, b, c = dims
    alpha, beta, gamma = angles

    # Optional parameters
    min_dist = float(args.mindist)
    baseplace = args.baseplace
    max_attempt = int(args.maxattempt)
    random_seed = None if args.randomseed is None else int(args.randomseed)

    # Run the placement algorithm
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
        min_interatomic_distance=min_dist,
        base_structure_placement_mode=baseplace,
        max_placement_attempts_per_copy=max_attempt,
        random_seed=random_seed,
    )

    # Handle output
    out_path = Path(args.output)
    ext = out_path.suffix.lower()

    if ext == ".xyz":
        # Direct XYZ write
        write_structure(atoms, out_path)
        print(f"[Done] Placed {args.ncopy} copies into box → {out_path}")
    else:
        # Intermediate XYZ then xtob → GEO/BGF/etc.
        tmp_xyz = Path("place2_output.xyz")
        write_structure(atoms, tmp_xyz)
        xtob(
            xyz_file=tmp_xyz,
            geo_file=out_path,
            box_lengths=dims,
            box_angles=angles,
            sort_by=None,
            ascending=True,
        )
        print(
            f"[Done] Placed {args.ncopy} copies into box → {tmp_xyz} "
            f"→ converted to {out_path} via xtob"
        )

    return 0


# ----------------------------------------------------------------------
# CLI registration
# ----------------------------------------------------------------------

def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    Register GEO workflow tasks under the `geo` command.
    Commands:
        reaxkit geo xtob ...
        reaxkit geo make ...
        reaxkit geo sort ...
    """

    # ---- xtob ----
    p_xtob = subparsers.add_parser("xtob",
                                   help="Convert an XYZ file to GEO (XTLGRF) format || "
                                        "reaxkit geo xtob --file slab.xyz --dims 11.0,12.0,100.0 --angles 90,90,90 --output geo")
    p_xtob.add_argument("--file", required=True, help="Input XYZ file (X.xyz)")
    p_xtob.add_argument("--dims", required=True, help="Box dimensions a,b,c (e.g., 11.0,12.0,100.0)")
    p_xtob.add_argument("--angles", required=True, help="Box angles alpha,beta,gamma (e.g., 90,90,90)")
    p_xtob.add_argument("--output", default="geo", help="Output GEO file name (default: geo)")
    p_xtob.add_argument("--sort", choices=["x", "y", "z", "atom_type"], help="Sort atoms before writing GEO")
    p_xtob.add_argument("--descending", action="store_true", help="Sort in descending order")
    p_xtob.set_defaults(_run=xtob_task)

    # ---- make ----
    p_make = subparsers.add_parser("make",
                                   help="Build a surface slab from bulk and write XYZ/CIF || "
                                        "reaxkit geo make --file bulk.cif --output slab.xyz --surface 1,0,0 --expand 4,4,6 --vacuum 15")
    p_make.add_argument("--file", required=True, help="Input bulk file (CIF, POSCAR, etc.)")
    p_make.add_argument("--output", required=True, help="Output file (XYZ, CIF, etc.)")
    p_make.add_argument("--surface", required=True, help="Miller indices h,k,l (e.g., 1,0,0)")
    p_make.add_argument("--expand", required=True, help="Supercell and layers nx,ny,layers (e.g., 4,4,6)")
    p_make.add_argument("--vacuum", required=True, help="Vacuum thickness in Å (e.g., 15)")
    p_make.set_defaults(_run=make_task)

    # ---- sort ----
    p_sort = subparsers.add_parser("sort", help="Sort atoms in a GEO file and write a new GEO || "
                                                "reaxkit geo sort --file geo --output sorted_geo --sort x")
    p_sort.add_argument("--file", required=True, help="Input GEO file (X.geo)")
    p_sort.add_argument("--output", required=True, help="Output GEO file (Y.geo)")
    p_sort.add_argument("--sort", required=True, choices=["m", "x", "y", "z", "atom_type"], help="Sort key: m=atom index, x/y/z=coordinates, atom_type=element")
    p_sort.add_argument("--descending", action="store_true", help="Sort in descending order")
    p_sort.set_defaults(_run=sort_task)

    # ---- ortho (orthogonalize) ----
    p_ortho = subparsers.add_parser("ortho",
                                    help="Convert hexagonal (90,90,120) cell to orthorhombic (90,90,90) || "
                                         "reaxkit geo ortho --file AlN_hex.cif --output AlN_ortho.cif")
    p_ortho.add_argument("--file", required=True, help="Input CIF/POSCAR/GEO file to orthogonalize")
    p_ortho.add_argument("--output", required=True, help="Output file (e.g., AlN_ortho.cif)")
    p_ortho.set_defaults(_run=ortho_task)

    # ---- place2 ----
    p_place2 = subparsers.add_parser("place2",
        help="Randomly place copies of a molecule into a box and optionally around a base structure || "
             "reaxkit geo place2 --insert template.xyz --ncopy 40 --dims 28.8,33.27,60 --angles 90,90,90 --output place2_no_base.xyz || "
             "reaxkit geo place2 --insert template.xyz --ncopy 40 --dims 28.8,33.27,60 --angles 90,90,90 --output place2_with_base.xyz --base base.xyz"
             "reaxkit geo place2 --insert template.xyz --ncopy 40 --dims 28.8,33.27,60 --angles 90,90,90 --output place2_geo --base base.xyz"
    )
    p_place2.add_argument("--insert", required=True,
        help="Insert molecule (XYZ or any ASE-readable format, e.g., X.xyz)",
    )
    p_place2.add_argument("--ncopy", required=True, help="Number of copies of the insert molecule to place")
    p_place2.add_argument("--dims", required=True, help="Box dimensions a,b,c (e.g., 30,30,60)")
    p_place2.add_argument("--angles", required=True, help="Box angles alpha,beta,gamma (e.g., 90,90,90)")
    p_place2.add_argument("--output", required=True, help="Output file: Y.xyz, Y.bgf, or 'geo'")
    p_place2.add_argument("--base",help="Optional base structure (e.g., slab.xyz) to place molecules around")
    p_place2.add_argument("--mindist", default=2.0,
        help="Minimum interatomic distance between insert copies and base/system (Å), default=2.0",
    )
    p_place2.add_argument("--baseplace", default="as-is", choices=["as-is", "center", "origin"],
        help="How to place the base structure: as-is, center, or origin (default: as-is)"
    )
    p_place2.add_argument("--maxattempt", default=50000, help="Maximum placement attempts per copy (default: 50000)")
    p_place2.add_argument("--randomseed", default=None, help="Random seed for reproducible placement (optional)")
    p_place2.set_defaults(_run=place2_task)



