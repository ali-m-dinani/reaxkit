"""
Geometry and structure generators for ReaxFF simulations.

This module provides high-level utilities for creating, converting,
and manipulating atomic structures used in ReaxFF workflows.

Capabilities include:
---

- Conversion of XYZ files to GEO/XTLGRF format with full ReaxFF-compliant
  formatting.
- Structure I/O wrappers based on ASE (CIF, POSCAR, VASP, XYZ, etc.).
- Surface slab construction from bulk crystals.
- Supercell generation via repetition or full 3×3 transformation matrices.
- Hexagonal → orthorhombic cell transformations.
- A Python reimplementation of the Fortran `place2` algorithm for random
  molecular packing with periodic boundary conditions.
- Programmatic insertion of sample restraint blocks into GEO files.

Generators in this module:
---

- operate deterministically on input structures
- write files or return ASE Atoms objects
- do not perform simulation, analysis, or parsing of ReaxFF output
"""


from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional, Tuple, List, Dict, Any, Sequence

import pandas as pd
import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.io import read, write
from ase.build import surface as ase_surface, make_supercell as ase_make_supercell

SortKey = Literal["x", "y", "z", "atom_type"]
TerminationSide = Literal["top", "bottom"]
Axis = Literal[0, 1, 2]

# ---------------------------------------------------------------------------
# Core reader for XYZ
# ---------------------------------------------------------------------------

def _read_xyz(xyz_path: str | Path) -> Tuple[str, pd.DataFrame]:
    """
    Read a simple XYZ file and return (descriptor, atoms_df).

    Parameters
    ----------
    xyz_path : str or Path
        Path to the .xyz file.

    Returns
    -------
    descriptor : str
        Descriptor derived from the second line (first token).
    atoms_df : pandas.DataFrame
        Columns: ["atom_type", "x", "y", "z"]
    """
    xyz_path = Path(xyz_path)

    with xyz_path.open("r") as fh:
        # First non-empty line: number of atoms
        first = ""
        while first == "":
            first = fh.readline()
            if not first:
                raise ValueError(f"❌ {xyz_path} appears to be empty.")
            first = first.strip()

        try:
            nat_expected = int(first.split()[0])
        except ValueError:
            raise ValueError(f"❌ First line of {xyz_path} is not a valid atom count: {first!r}")

        # Second non-empty line: descriptor line (first token used)
        second = ""
        while second == "":
            second = fh.readline()
            if not second:
                raise ValueError(f"❌ {xyz_path} ended before descriptor line.")
            second = second.strip()

        descriptor_tokens = second.split()
        descriptor = descriptor_tokens[0] if descriptor_tokens else ""

        # Remaining lines: atoms
        records: List[Dict[str, Any]] = []
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 4:
                continue

            symbol = parts[0]

            # NEW: skip non-atom lines (VEC1, VEC2, etc.)
            if symbol.upper().startswith("VEC"):
                continue

            # You can also skip any non-alphabetic symbol:
            if not symbol[0].isalpha():
                continue

            try:
                x, y, z = map(float, parts[1:4])
            except ValueError:
                continue

            records.append({
                "atom_type": symbol,
                "x": x,
                "y": y,
                "z": z,
            })

    atoms_df = pd.DataFrame(records, columns=["atom_type", "x", "y", "z"])

    if len(atoms_df) != nat_expected:
        raise ValueError(
            f"❌ Number of atoms in XYZ header ({nat_expected}) "
            f"does not match coordinate lines found ({len(atoms_df)})."
        )

    return descriptor, atoms_df


# ---------------------------------------------------------------------------
# Sorting helper
# ---------------------------------------------------------------------------

def _sort_atoms(
    atoms: pd.DataFrame,
    sort_by: Optional[SortKey] = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Optionally sort atoms by coordinate or atom type.

    Parameters
    ----------
    atoms : DataFrame
        Columns ["atom_type", "x", "y", "z"].
    sort_by : {"x", "y", "z", "atom_type"} or None, optional
        Which column to sort by. If None, no sorting.
    ascending : bool, default True
        Sort direction.

    Returns
    -------
    DataFrame
        Sorted (or unchanged) DataFrame.
    """
    if sort_by is None:
        return atoms

    if sort_by not in atoms.columns:
        raise ValueError(f"❌ sort_by must be one of {list(atoms.columns)!r}, got {sort_by!r}.")

    return atoms.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Writer for GEO / XTLGRF
# ---------------------------------------------------------------------------

def _format_crystx(
    box_lengths: Iterable[float],
    box_angles: Iterable[float],
) -> str:
    """
    Format the CRYSTX line:

        CRYSTX    a        b        c      alpha     beta     gamma

    with each value as f11.5.
    """
    a, b, c = list(box_lengths)
    alpha, beta, gamma = list(box_angles)

    nums = (a, b, c, alpha, beta, gamma)
    return "CRYSTX" + "".join(f"{v:11.5f}" for v in nums)


def _format_hetatm_line(atom_id: int, atom_type: str, x: float, y: float, z: float) -> str:
    at2 = atom_type.strip()[:2]      # a2 field
    at5 = atom_type.strip()[:5]      # a5 field

    return (
        "HETATM"                    # literal
        f" {atom_id:5d}"            # 1x, i5
        f" {at2:2s}"                # 1x, a2
        "   "                       # 3x
        " "                         # 1x
        "   "                       # 3x
        " "                         # 1x
        " "                         # 1x
        " "                         # 1x
        "     "                     # 5x
        f"{x:10.5f}{y:10.5f}{z:10.5f}"  # 3f10.5
        f" {at5:5s}"                # 1x, a5
        f"{0:3d}{0:2d}"             # i3, i2  (0 0)
        f" {0.0:8.5f}"              # 1x, f8.5 (0.00000)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def xtob(
    xyz_file: str | Path,
    geo_file: str | Path = "geo",
    box_lengths: Iterable[float] = (1.0, 1.0, 1.0),
    box_angles: Iterable[float] = (90.0, 90.0, 90.0),
    sort_by: Optional[SortKey] = None,
    ascending: bool = True,
) -> None:
    """
    Convert an XYZ file to ReaxFF GEO/XTLGRF format.

    This function reads a simple XYZ structure, optionally sorts atoms,
    assigns sequential atom IDs, and writes a fully formatted GEO file
    compatible with ReaxFF.

    Works on
    ---
    XYZ structure files

    Parameters
    ----------
    xyz_file : str | Path
        Input XYZ file.
    geo_file : str | Path, optional
        Output GEO file path (default: ``"geo"``).
    box_lengths : iterable of float, optional
        Periodic cell lengths (a, b, c) in Å.
    box_angles : iterable of float, optional
        Periodic cell angles (alpha, beta, gamma) in degrees.
    sort_by : {"x","y","z","atom_type"} or None, optional
        If provided, sort atoms before writing.
    ascending : bool, default True
        Sort direction.

    Returns
    -------
    None
        Writes the GEO file to disk.

    Examples
    ---
    >>> xtob("structure.xyz", "geo", box_lengths=(10,10,10))
    """
    xyz_file = Path(xyz_file)
    geo_file = Path(geo_file)

    descriptor, atoms = _read_xyz(xyz_file)

    # Normalize box inputs
    box_lengths = list(box_lengths)
    box_angles = list(box_angles)

    if len(box_lengths) != 3 or len(box_angles) != 3:
        raise ValueError(
            "❌ box_lengths and box_angles must each contain exactly 3 values "
            "(a, b, c) and (alpha, beta, gamma)."
        )

    # Sort if requested
    atoms_sorted = _sort_atoms(atoms, sort_by=sort_by, ascending=ascending)

    # Reassign atom IDs after sorting
    atoms_sorted = atoms_sorted.reset_index(drop=True)
    atoms_sorted["atom_id"] = atoms_sorted.index + 1

    # Prepare REMARK lines
    sort_remark = None
    if sort_by is not None:
        direction = "ascending" if ascending else "descending"
        if sort_by == "atom_type":
            coord_label = "atom type"
        else:
            coord_label = f"{sort_by}-coordinate"
        sort_remark = f"REMARK Structure sorted by {coord_label} ({direction})"

    # Write GEO file
    with geo_file.open("w") as fh:
        # Header
        fh.write("XTLGRF 200\n")
        fh.write(f"DESCRP  {descriptor}\n")
        fh.write("REMARK .bgf-file generated by xtob-python\n")
        if sort_remark:
            fh.write(f"{sort_remark}\n")
        fh.write(_format_crystx(box_lengths, box_angles) + "\n")
        fh.write(
            "FORMAT ATOM   (a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5)\n"
        )

        # Atoms
        for row in atoms_sorted.itertuples(index=False):
            line = _format_hetatm_line(
                atom_id=row.atom_id,
                atom_type=row.atom_type,
                x=row.x,
                y=row.y,
                z=row.z,
            )
            fh.write(line + "\n")

        # Footer
        fh.write("END\n")


__all__ = ["xtob"]


# ---------------------------------------------------------------------------
# 1) Read structure (CIF, POSCAR, etc.)
# ---------------------------------------------------------------------------

def read_structure(
    path: str | Path,
    format: Optional[str] = None,
    index: int | str = 0,
) -> Atoms:
    """
    Read a structure file using ASE.

    Works on
    ---
    ASE-supported structure formats (CIF, POSCAR, XYZ, etc.)

    Parameters
    ----------
    path : str | Path
        Structure file path (CIF, POSCAR, XYZ, etc.).
    format : str, optional
        ASE format string. If None, ASE infers from file extension.
    index : int | str, default 0
        Image index if the file contains multiple structures.

    Returns
    -------
    ase.Atoms
        Loaded structure.

    Examples
    ---
    >>> atoms = read_structure("AlN.cif")
    """
    path = Path(path)
    return read(path, format=format, index=index)


# ---------------------------------------------------------------------------
# 2) Generate a surface slab
# ---------------------------------------------------------------------------

def build_surface(
    bulk: Atoms,
    miller: Tuple[int, int, int],
    layers: int,
    vacuum: float = 15.0,
    center: bool = True,
) -> Atoms:
    """
    Build a surface slab from a bulk structure using Miller indices.

    Works on
    ---
    ASE Atoms bulk structures

    Parameters
    ----------
    bulk : ase.Atoms
        Bulk crystal structure.
    miller : (h, k, l)
        Miller indices defining the surface orientation.
    layers : int
        Number of atomic layers.
    vacuum : float, default 15.0
        Vacuum thickness along the surface normal (Å).
    center : bool, default True
        Center the slab along the surface-normal direction.

    Returns
    -------
    ase.Atoms
        Surface slab structure.

    Examples
    ---
    >>> slab = build_surface(bulk, (0,0,1), layers=6)
    """
    slab = ase_surface(bulk, miller, layers=layers, vacuum=vacuum)
    if center:
        # Surface normal is along cell c (axis 2) by ASE convention
        slab.center(axis=2)
    return slab


# ---------------------------------------------------------------------------
# 3) Transform: generate supercells / apply transformation matrix
# ---------------------------------------------------------------------------

def make_supercell(
    atoms: Atoms,
    transform: Iterable[Iterable[int]] | Tuple[int, int, int],
) -> Atoms:
    """
    Generate a supercell or apply a lattice transformation.

    Works on
    ---
    ASE Atoms structures

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure.
    transform : (nx, ny, nz) tuple or 3×3 integer matrix
        Repetition factors or full transformation matrix.

    Returns
    -------
    ase.Atoms
        Transformed structure.

    Examples
    ---
    >>> sc = make_supercell(atoms, (2,2,1))
    """
    # Simple repetition (e.g., (4, 4, 1))
    if isinstance(transform, tuple) and len(transform) == 3:
        return atoms * transform  # ASE overloads * for repetition

    # Full 3×3 integer matrix
    P = np.array(transform, dtype=int)
    if P.shape != (3, 3):
        raise ValueError(
            "transform must be either (nx, ny, nz) or a 3x3 integer matrix."
        )
    return ase_make_supercell(atoms, P)


# ---------------------------------------------------------------------------
# 5) Convert / write to other formats (e.g., XYZ)
# ---------------------------------------------------------------------------

def write_structure(
    atoms: Atoms,
    path: str | Path,
    format: Optional[str] = None,
    comment: Optional[str] = None,
) -> None:
    """
    Write a structure to file in any ASE-supported format.

    Works on
    ---
    ASE Atoms objects

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to write.
    path : str or Path
        Output file path. Extension is used to guess format if `format` is None.
    format : str, optional
        This is based on ASE format support as explained here:
        https://ase-lib.org/ase/io/io.html#ase.io.write
        ASE format string (e.g., "xyz", "cif", "vasp", "xsf"). If None,
        ASE guesses from the file extension.

    Returns
    -------
    None

    Examples
    ---
    >>> write_structure(atoms, "out.xyz")
    """
    path = Path(path)
    write(path, atoms, format=format, comment=comment)

# ---------------------------------------------------------------------------
# convert a hexagonal cell (90°, 90°, 120°) into an orthorhombic (90°, 90°, 90°) cell
# ---------------------------------------------------------------------------

def orthogonalize_hexagonal_cell(atoms: Atoms) -> Atoms:
    """
    Convert a hexagonal unit cell into an orthorhombic cell.

    Convert a hexagonal cell (90°, 90°, 120°) into an orthorhombic cell (90°, 90°, 90°)
    using the standard a-b, a+b transformation.

    Works on
    ---
    ASE Atoms structures with hexagonal lattice geometry

    Parameters
    ---
    atoms : ase.Atoms
        Hexagonal structure.

    Returns
    ---
    ase.Atoms
        Orthorhombic structure.

    Examples
    ---
    >>> ortho = orthogonalize_hexagonal_cell(hex_atoms)
    """
    import numpy as np
    from ase.build.supercells import make_supercell

    # Transformation matrix: new vectors = old vectors * T
    T = np.array([
        [1, -1, 0],
        [1,  1, 0],
        [0,  0, 1]
    ])

    new_atoms = make_supercell(atoms, T)
    return new_atoms

# ---------------------------------------------------------------------------
# place2 algorithm for placing n instances of a structure into a simulation box or within another structure
# ---------------------------------------------------------------------------

def place2(
    insert_molecule: str | Path,
    base_structure: Optional[str | Path] = None,
    *,
    n_copies: int,
    box_length_x: float,
    box_length_y: float,
    box_length_z: float,
    alpha: float,
    beta: float,
    gamma: float,
    min_interatomic_distance: float,
    base_structure_placement_mode: str = "as-is",   # "as-is", "center", "origin"
    max_placement_attempts_per_copy: int = 50000,
    random_seed: int | None = None,
) -> Atoms:
    """
    Randomly place multiple copies of a molecule into a simulation cell.

    Python reimplementation of the Fortran `place2` algorithm

    Works on
    ---
    Molecular structure files and simulation boxes

    Parameters
    ---
    insert_molecule : str or pathlib.Path
        Structure to be duplicated.
    base_structure : str or pathlib.Path or None, optional
        Initial structure to insert into.
    n_copies : int
        Number of copies to place.
    box_length_x, box_length_y, box_length_z : float
        Simulation box lengths (Å).
    alpha, beta, gamma : float
        Cell angles (degrees).
    min_interatomic_distance : float
        Minimum allowed atomic separation.
    base_structure_placement_mode : str, optional
        Base structure positioning mode.
    max_placement_attempts_per_copy : int, optional
        Maximum placement attempts.
    random_seed : int or None, optional
        Random seed for reproducibility.

    Returns
    ---
    ase.Atoms
        Combined atomic system.

    Examples
    ---
    >>> atoms = place2("H2O.xyz", n_copies=100, box_length_x=30,
    ...                box_length_y=30, box_length_z=30,
    ...                alpha=90, beta=90, gamma=90,
    ...                min_interatomic_distance=1.5)
    """

    rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # 1) Build triclinic cell matrix
    # ------------------------------------------------------------------
    cell = cellpar_to_cell(
        [box_length_x, box_length_y, box_length_z, alpha, beta, gamma]
    )
    inv_cell = np.linalg.inv(cell)

    # ------------------------------------------------------------------
    # 2) Read and center the insert_molecule at origin
    # ------------------------------------------------------------------
    insert_atoms = read_structure(insert_molecule)
    insert_symbols = insert_atoms.get_chemical_symbols()
    insert_coords = insert_atoms.get_positions()

    insert_center = insert_coords.mean(axis=0)
    insert_coords_centered = insert_coords - insert_center

    # ------------------------------------------------------------------
    # 3) Initialize with base_structure (if given)
    # ------------------------------------------------------------------
    system_symbols: List[str] = []
    system_coords = np.empty((0, 3), dtype=float)

    if base_structure is not None:
        base_atoms = read_structure(base_structure)
        base_coords = base_atoms.get_positions()
        base_symbols = base_atoms.get_chemical_symbols()

        if base_structure_placement_mode not in {"as-is", "center", "origin"}:
            raise ValueError(
                "base_structure_placement_mode must be one of "
                '{"as-is", "center", "origin"}'
            )

        if base_structure_placement_mode in {"center", "origin"}:
            base_center = base_coords.mean(axis=0)
            if base_structure_placement_mode == "center":
                box_center = np.array([
                    box_length_x / 2,
                    box_length_y / 2,
                    box_length_z / 2,
                ])
                shift = box_center - base_center
            else:
                # "origin"
                shift = -base_center
            base_coords = base_coords + shift

        system_symbols.extend(base_symbols)
        system_coords = np.vstack([system_coords, base_coords])

    # ------------------------------------------------------------------
    # Helper: random rotation matrix
    # ------------------------------------------------------------------
    def random_rotation_matrix() -> np.ndarray:
        phi = 2 * np.pi * rng.random()
        theta = 2 * np.pi * rng.random()
        psi = 2 * np.pi * rng.random()

        c1, s1 = np.cos(phi), np.sin(phi)
        c2, s2 = np.cos(theta), np.sin(theta)
        c3, s3 = np.cos(psi), np.sin(psi)

        Rz1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
        Ry  = np.array([[c2,  0, s2], [0, 1, 0], [-s2, 0, c2]])
        Rz2 = np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])

        return Rz2 @ Ry @ Rz1

    # ------------------------------------------------------------------
    # Helper: minimum distance to existing system with PBC
    # ------------------------------------------------------------------
    def min_distance(new: np.ndarray, existing: np.ndarray) -> float:
        if existing.size == 0:
            return float("inf")

        diff = new[:, None, :] - existing[None, :, :]
        frac = diff @ inv_cell
        frac -= np.round(frac)
        cart = frac @ cell
        d2 = np.sum(cart ** 2, axis=-1)
        return float(np.sqrt(d2.min()))

    # ------------------------------------------------------------------
    # 4) Insert N random copies
    # ------------------------------------------------------------------
    for n in range(n_copies):
        attempts = 0
        while True:
            attempts += 1
            if attempts > max_placement_attempts_per_copy:
                raise RuntimeError(
                    f"Could not place copy {n+1} after "
                    f"{max_placement_attempts_per_copy} attempts."
                )

            R = random_rotation_matrix()
            rotated = insert_coords_centered @ R.T

            translate = np.array([
                box_length_x * rng.random(),
                box_length_y * rng.random(),
                box_length_z * rng.random(),
            ])
            new_coords = rotated + translate

            if min_distance(new_coords, system_coords) > min_interatomic_distance:
                system_coords = np.vstack([system_coords, new_coords])
                system_symbols.extend(insert_symbols)
                break
            # else: try again

    # ------------------------------------------------------------------
    # 5) Build final ASE atoms
    # ------------------------------------------------------------------
    final_atoms = Atoms(
        symbols=system_symbols,
        positions=system_coords,
        cell=cell,
        pbc=True,
    )
    return final_atoms

# ---------------------------------------------------------------------------
# adding a restraint of any type (bond, angle, torsion, mascen) to a geo file
# ---------------------------------------------------------------------------

def add_restraints_to_geo(
    geo_file: str | Path,
    *,
    out_file: str | Path | None = None,
    kinds: Sequence[str],
    params: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Insert sample restraint blocks into a GEO/XTLGRF file.

    Insert BEFORE first line starting with any of:
      CRYSTX, FORMAT ATOM, HETATM, ATOM
    else before END, else EOF.

    For each kind write exactly 3 lines (NO blank lines):
      FORMAT ...
      # (guide)
      <KIND> RESTRAINT ...   (fields LEFT-aligned under guide headers)

    Works on
    ---
    ReaxFF GEO/XTLGRF structure files

    Parameters
    ---
    geo_file : str or pathlib.Path
        Input GEO file.
    out_file : str or pathlib.Path or None, optional
        Output file path.
    kinds : sequence of str
        Restraint kinds to insert (e.g. BOND, ANGLE).
    params : dict or None, optional
        Custom restraint parameter strings.

    Returns
    ---
    pathlib.Path
        Path to the modified GEO file.

    Examples
    ---
    >>> add_restraints_to_geo("geo", kinds=["BOND", "ANGLE"])
    """
    geo_file = Path(geo_file)
    if not geo_file.is_file():
        raise FileNotFoundError(f"Input GEO file not found: {geo_file}")

    out_path = Path(out_file) if out_file is not None else geo_file
    params = params or {}

    wanted = [str(k).strip().upper() for k in kinds if str(k).strip()]
    if not wanted:
        raise ValueError("No restraint kinds provided (kinds is empty).")

    order = ["BOND", "ANGLE", "TORSION", "MASCEN"]
    wanted_sorted = [k for k in order if k in set(wanted)]

    default_params: Dict[str, str] = {
        "BOND":    "1   2  1.0900 7500.00 0.25000 0.0000000",
        "ANGLE":   "1   2   3 109.5000 600.00 0.25000 0.0000000",
        "TORSION": "1   2   3   4 180.0000 100.00 0.25000 0.0000000",
        "MASCEN":  "1   0.0000 0.0000 0.0000 500.00 0.25000 0.0000000",
    }

    format_lines: Dict[str, str] = {
        "BOND":    "FORMAT BOND RESTRAINT (15x,2i4,f8.4,f8.2,f8.5,f10.7)",
        "ANGLE":   "FORMAT ANGLE RESTRAINT (15x,3i4,f8.3,f8.2,f8.5,f10.7)",
        "TORSION": "FORMAT TORSION RESTRAINT (15x,4i4,f8.3,f8.2,f8.5,f10.7)",
        "MASCEN":  "FORMAT MASCEN RESTRAINT (15x,i4,3f10.4,f8.2,f8.5,f10.7)",
    }

    # Keep these exactly like your working example style: a short "#", then spaces, then headers.
    guide_lines: Dict[str, str] = {
        "BOND":    "#                                 At1 At2  R12     Force1  Force2  dR12/dIteration(MD only)",
        "ANGLE":   "#                                 At1 At2 At3  A123    Force1  Force2  dA123/dIteration(MD only)",
        "TORSION": "#                                 At1 At2 At3 At4  T1234   Force1  Force2  dT1234/dIteration(MD only)",
        "MASCEN":  "#                                 At1    X          Y          Z          Force1  Force2  dR/dIteration(MD only)",
    }

    # We will align under the *actual token starts* in the guide line (left-aligned).
    # Each spec gives: token_name, formatter(kind-specific), min_width (fallback)
    token_layout: Dict[str, List[Tuple[str, str, int]]] = {
        "BOND": [
            ("At1", "i", 3),
            ("At2", "i", 3),
            ("R12", "f4", 7),
            ("Force1", "f2", 7),
            ("Force2", "f5", 7),
            ("dR12/dIteration(MD only)", "f7", 10),
        ],
        "ANGLE": [
            ("At1", "i", 3),
            ("At2", "i", 3),
            ("At3", "i", 3),
            ("A123", "f3", 7),
            ("Force1", "f2", 7),
            ("Force2", "f5", 7),
            ("dA123/dIteration(MD only)", "f7", 10),
        ],
        "TORSION": [
            ("At1", "i", 3),
            ("At2", "i", 3),
            ("At3", "i", 3),
            ("At4", "i", 3),
            ("T1234", "f3", 7),
            ("Force1", "f2", 7),
            ("Force2", "f5", 7),
            ("dT1234/dIteration(MD only)", "f7", 10),
        ],
        "MASCEN": [
            ("At1", "i", 3),
            ("X", "f4_10", 10),
            ("Y", "f4_10", 10),
            ("Z", "f4_10", 10),
            ("Force1", "f2", 7),
            ("Force2", "f5", 7),
            ("dR/dIteration(MD only)", "f7", 10),
        ],
    }

    def _format_value(tok: str, fmt: str) -> str:
        # ints
        if fmt == "i":
            return str(int(float(tok)))
        # floats
        if fmt == "f2":
            return f"{float(tok):.2f}"
        if fmt == "f3":
            return f"{float(tok):.3f}"
        if fmt == "f4":
            return f"{float(tok):.4f}"
        if fmt == "f5":
            return f"{float(tok):.5f}"
        if fmt == "f7":
            return f"{float(tok):.7f}"
        if fmt == "f4_10":  # for X/Y/Z use wider 10.4 style
            return f"{float(tok):.4f}"
        return tok

    def _token_starts(guide: str, names: List[str]) -> List[int]:
        """
        Find start indices of each token name in the guide line.
        Uses find() from left to right; if missing, falls back to spaced layout.
        """
        starts: List[int] = []
        cursor = 0
        for nm in names:
            j = guide.find(nm, cursor)
            if j < 0:
                # fallback: place after last start + 4
                j = (starts[-1] + 4) if starts else guide.find("#") + 2
            starts.append(j)
            cursor = j + len(nm)
        return starts

    def _build_aligned_data_line(kind: str, param_str: str) -> str:
        """
        Build:
            "BOND RESTRAINT" + spaces + values
        where each value is LEFT-aligned to the same column as the header token in guide line.
        """
        guide = guide_lines[kind]
        layout = token_layout[kind]
        names = [x[0] for x in layout]
        starts = _token_starts(guide, names)

        toks = [t for t in (param_str or "").split() if t.strip()]

        # Basic validation: ensure enough tokens
        need = len(layout)
        if len(toks) < need:
            raise ValueError(f"{kind} params need {need} tokens but got {len(toks)}: {toks}")
        toks = toks[:need]

        label = f"{kind} RESTRAINT"
        # Start with label, then pad up to the first header start
        s = label
        if len(s) < starts[0]:
            s += " " * (starts[0] - len(s))
        else:
            s += " "

        # Place each field at exact start column; LEFT-aligned, no right-justification
        for (nm, fmt, minw), start, tok in zip(layout, starts, toks):
            val = _format_value(tok, fmt)

            if len(s) < start:
                s += " " * (start - len(s))
            # left align within a reasonable width so next field doesn't collide
            # width = distance to next start (or minw for last token)
            idx = names.index(nm)
            if idx < len(starts) - 1:
                width = max(minw, starts[idx + 1] - start - 1)
            else:
                width = max(minw, len(val))
            s += val.ljust(width) + " "

        return s.rstrip()

    # Read file
    lines = geo_file.read_text().splitlines()

    # Insert BEFORE first CRYSTX / FORMAT ATOM / HETATM / ATOM
    insert_idx: Optional[int] = None
    triggers = ("CRYSTX", "FORMAT ATOM", "HETATM", "ATOM")
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if any(s.startswith(t) for t in triggers):
            insert_idx = i
            break
    if insert_idx is None:
        for i, ln in enumerate(lines):
            if ln.strip() == "END":
                insert_idx = i
                break
    if insert_idx is None:
        insert_idx = len(lines)

    block: list[str] = []
    block.append("REMARK Restraints added by ReaxKit (sample lines; edit as needed)")

    for k in wanted_sorted:
        block.append(format_lines[k])
        block.append(guide_lines[k])

        p = (params.get(k) or "").strip()
        if not p:
            p = default_params[k]

        block.append(_build_aligned_data_line(k, p))

    new_lines = lines[:insert_idx] + block + lines[insert_idx:]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(new_lines) + "\n")
    return out_path



