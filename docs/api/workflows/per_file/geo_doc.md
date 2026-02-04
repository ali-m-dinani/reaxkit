# Geo Workflow

CLI namespace: `reaxkit geo <task> [flags]`

Geometry (GEO) manipulation workflow for ReaxKit.

This workflow provides a collection of utilities for creating, transforming,
and modifying atomic geometry files used in ReaxFF simulations, with a focus
on the GEO (XTLGRF) format and ASE-compatible structure files.

It supports:
- Converting XYZ structures to GEO format with explicit cell dimensions
  and angles.
- Building surface slabs from bulk structures (CIF, POSCAR, etc.), including
  Miller-index selection, supercell expansion, and vacuum padding.
- Sorting atoms in GEO files by index, coordinate, or atom type.
- Orthogonalizing hexagonal unit cells (90°, 90°, 120°) into orthorhombic
  representations (90°, 90°, 90°).
- Randomly placing multiple copies of a molecule into a simulation box or
  around an existing structure using a placement algorithm.
- Inserting sample or user-defined restraint blocks (bond, angle, torsion,
  mass-center) into GEO files for constrained simulations.

The workflow is designed to streamline preparation of ReaxFF input geometries
and to support reproducible, scriptable structure generation from the command line.

## Available tasks

### `add-restraint`

#### Examples

- `reaxkit geo add-restraint --bond`
- `reaxkit geo add-restraint --file geo --output geo_r --angle '1   2   3 109.5000 600.00 0.25000 0.0000000'`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Input GEO file (e.g., geo) |
| `--output OUTPUT` | Output GEO file (default: <input>_with_restraints) |
| `--bond [BOND]` | Add ONE BOND restraint (optional params string; empty => default sample). |
| `--angle [ANGLE]` | Add ONE ANGLE restraint (optional params string; empty => default sample). |
| `--torsion [TORSION]` | Add ONE TORSION restraint (optional params string; empty => default sample). |
| `--mascen [MASCEN]` | Add ONE MASCEN restraint (optional params string; empty => default sample). |

### `make`

#### Examples

- `reaxkit geo make --file AlN.cif --output slab_from_AlN_cif.xyz --surface 1,0,0 --expand 4,4,6 --vacuum 15`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Input bulk file (CIF, POSCAR, etc.) |
| `--output OUTPUT` | Output file (XYZ, CIF, etc.) |
| `--surface SURFACE` | Miller indices h,k,l (e.g., 1,0,0) |
| `--expand EXPAND` | Supercell and layers nx,ny,layers (e.g., 4,4,6) |
| `--vacuum VACUUM` | Vacuum thickness in Å (e.g., 15) |

### `ortho`

#### Examples

- `reaxkit geo ortho --file AlN.cif --output AlN_ortho_from_hex.cif`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Input CIF/POSCAR/GEO file to orthogonalize |
| `--output OUTPUT` | Output file (e.g., AlN_ortho.cif) |

### `place2`

#### Examples

- `reaxkit geo place2 --insert template.xyz --ncopy 40 --dims 28.8,33.27,60 --angles 90,90,90 --output place2_on_template_xyz_with_no_base.xyz`
- `reaxkit geo place2 --insert template.xyz --ncopy 40 --dims 28.8,33.27,60 --angles 90,90,90 --output place2__on_template_xyz_with_base.xyz --base base.xyz`
- `reaxkit geo place2 --insert template.xyz --ncopy 40 --dims 28.8,33.27,60 --angles 90,90,90 --output place2_geo_from_template_xyz --base base.xyz`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--insert INSERT` | Insert molecule (XYZ or any ASE-readable format, e.g., X.xyz) |
| `--ncopy NCOPY` | Number of copies of the insert molecule to place |
| `--dims DIMS` | Box dimensions a,b,c (e.g., 30,30,60) |
| `--angles ANGLES` | Box angles alpha,beta,gamma (e.g., 90,90,90) |
| `--output OUTPUT` | Output file: Y.xyz, Y.bgf, or 'geo' |
| `--base BASE` | Optional base structure (e.g., slab.xyz) to place molecules around |
| `--mindist MINDIST` | Minimum interatomic distance between insert copies and base/system (Å), default=2.0 |
| `--baseplace {as-is,center,origin}` | How to place the base structure: as-is, center, or origin (default: as-is) |
| `--maxattempt MAXATTEMPT` | Maximum placement attempts per copy (default: 50000) |
| `--randomseed RANDOMSEED` | Random seed for reproducible placement (optional) |

### `sort`

#### Examples

- `reaxkit geo sort --file geo --output sorted_geo --sort x`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Input GEO file (X.geo) |
| `--output OUTPUT` | Output GEO file (Y.geo) |
| `--sort {m,x,y,z,atom_type}` | Sort key: m=atom index, x/y/z=coordinates, atom_type=element |
| `--descending` | Sort in descending order |

### `xtob`

#### Examples

- `reaxkit geo xtob --file slab.xyz --dims 11.0,12.0,100.0 --angles 90,90,90 --output slab_geo_from_xyz`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Input XYZ file (X.xyz) |
| `--dims DIMS` | Box dimensions a,b,c (e.g., 11.0,12.0,100.0) |
| `--angles ANGLES` | Box angles alpha,beta,gamma (e.g., 90,90,90) |
| `--output OUTPUT` | Output GEO file name (default: geo) |
| `--sort {x,y,z,atom_type}` | Sort atoms before writing GEO |
| `--descending` | Sort in descending order |
