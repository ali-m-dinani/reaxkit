"""Read complete LAMMPS text dump frames for geometry export."""

from io import StringIO

import numpy as np
from ase.data import atomic_numbers
from ase.io import read
from ase.io.lammpsrun import read_lammps_dump_text


def read_frame(source, selected, *, type_map=None, cell=None, units="metal"):
    """Select a complete native or XYZ frame without loading the full trajectory.

    Native boxes, including triclinic boxes, are decoded by ASE. Numeric types
    are never interpreted as atomic numbers. Cartesian positions are shifted
    to the box origin before the shared GEO orientation step.
    """
    if type_map and any(atomic_numbers.get(symbol, 0) == 0 for symbol in type_map):
        raise ValueError("--type-map must contain valid element symbols in type-number order.")
    if units not in {"metal", "real"}:
        raise ValueError("LAMMPS export supports metal or real units (angstrom coordinates).")
    chosen = None
    with open(source, encoding="utf-8") as stream:
        source_index = 0
        while True:
            first = stream.readline()
            if not first:
                break
            if not first.strip():
                continue
            native = first.startswith("ITEM: TIMESTEP")
            if native:
                header = [first] + [stream.readline() for position in range(8)]
                if any(not line for line in header):
                    break
                if not header[2].startswith("ITEM: NUMBER OF ATOMS") or not header[4].startswith("ITEM: BOX BOUNDS") or not header[8].startswith("ITEM: ATOMS"):
                    raise ValueError(f"Malformed LAMMPS dump header at frame {source_index}.")
                count = int(header[3])
                columns = header[8].split()[2:]
            else:
                count = int(first)
                header = [first, stream.readline()]
                if not header[1]:
                    break
                columns = None
            if count <= 0:
                raise ValueError(f"Invalid atom count at frame {source_index}.")
            rows = []
            for atom_index in range(count):
                line = stream.readline()
                if not line:
                    break
                fields = line.split()
                if len(fields) < (len(columns) if native else 4):
                    if not line.endswith("\n"):
                        break
                    raise ValueError(f"Malformed atom row at frame {source_index}.")
                if not line.endswith("\n"):
                    numeric_fields = (
                        [value for column, value in zip(columns, fields) if column in {
                            "id", "type", "x", "y", "z", "xu", "yu", "zu",
                            "xs", "ys", "zs", "xsu", "ysu", "zsu",
                        }]
                        if native else fields[1:4]
                    )
                    try:
                        [float(value) for value in numeric_fields]
                    except ValueError:
                        break
                rows.append(line)
            if len(rows) != count:
                break
            chosen = (source_index, native, header, rows, columns)
            if source_index == selected:
                break
            source_index += 1
    if chosen is None or (selected >= 0 and chosen[0] != selected):
        raise ValueError(f"No complete frame {selected!r} found in {source}.")
    source_index, native, header, rows, columns = chosen
    if native:
        if "element" not in columns:
            if "type" not in columns or not type_map:
                raise ValueError("LAMMPS dump needs an element column or --type-map (e.g. --type-map Al N).")
            type_column = columns.index("type")
            for row in rows:
                atom_type = int(row.split()[type_column])
                if not 1 <= atom_type <= len(type_map):
                    raise ValueError(f"LAMMPS type {atom_type} is missing from --type-map.")
            if "mass" in columns:
                columns = ["element" if column == "mass" else column for column in columns]
                mass_column = header[8].split()[2:].index("mass")
                mapped_rows = []
                for row in rows:
                    fields = row.split()
                    fields[mass_column] = type_map[int(fields[type_column]) - 1]
                    mapped_rows.append(" ".join(fields) + "\n")
                rows = mapped_rows
                header[8] = "ITEM: ATOMS " + " ".join(columns) + "\n"
        atoms = read_lammps_dump_text(StringIO("".join(header + rows)), index=0, order=False, specorder=type_map, units=units)
        if any(column in columns for column in ("x", "xu")):
            atoms.positions -= np.asarray(atoms.get_celldisp()).reshape(3)
    else:
        atoms = read(StringIO("".join(header + rows)), format="extxyz")
        if cell is not None:
            atoms.set_cell(cell)
        if atoms.cell.rank != 3:
            raise ValueError("XYZ dump has no 3D cell; supply --cell a b c alpha beta gamma.")
    atoms.info["source_frame_index"] = source_index
    return atoms
