"""Export individual trajectory frames from supported simulation engines."""

from pathlib import Path

import numpy as np
from ase.data import atomic_numbers

from reaxkit.engine.reaxff.generators.geo_generator import orient_structure_for_reaxff, xtob


def frame_index(frame: int | str) -> int:
    """Normalize zero-based frame selectors, using -1 for the last frame."""
    try:
        selected = -1 if str(frame).lower() == "last" else int(str(frame))
    except ValueError as exc:
        raise ValueError("frame must be a zero-based integer, -1, or 'last'.") from exc
    if selected < -1:
        raise ValueError("frame must be nonnegative, -1, or 'last'.")
    return selected


def extract_frame(trajectory, *, engine="reaxff", frame="last", xyz_file="frame.xyz",
                  geo_file="frame.geo", type_map=None, cell=None, units="metal"):
    """Write XYZ and ReaxFF GEO, orienting non-ReaxFF cells and coordinates together.

    ``type_map`` lists elements for LAMMPS types 1, 2, ... . ``cell`` supplies
    six cell parameters for XYZ dumps without lattice metadata (angstrom/degrees).
    """
    source, xyz_path, geo_path = Path(trajectory), Path(xyz_file), Path(geo_file)
    if len({source.resolve(), xyz_path.resolve(), geo_path.resolve()}) != 3:
        raise ValueError("Trajectory, XYZ output and GEO output must be different files.")
    selected = frame_index(frame)
    if engine == "reaxff":
        from reaxkit.engine.reaxff.generators.trajectory_frame import extract_frame as extract_reaxff

        return extract_reaxff(source, frame=selected, xyz_file=xyz_path, geo_file=geo_path)
    if engine == "ams":
        from reaxkit.engine.ams.trajectory_frame import read_frame

        atoms = read_frame(source, selected)
    elif engine == "lammps":
        from reaxkit.engine.lammps.trajectory_frame import read_frame

        atoms = read_frame(source, selected, type_map=type_map, cell=cell, units=units)
    else:
        raise ValueError(f"Unsupported trajectory engine: {engine}")
    if not len(atoms) or not np.isfinite(atoms.positions).all():
        raise ValueError("Selected frame has empty or non-finite coordinates.")
    if any(atomic_numbers.get(symbol, 0) == 0 for symbol in atoms.get_chemical_symbols()):
        raise ValueError("Selected frame requires valid element symbols for GEO export.")
    atoms = orient_structure_for_reaxff(atoms)
    lines = [str(len(atoms)), f"{engine} frame {atoms.info['source_frame_index']}; ReaxFF Cartesian coordinates"]
    lines.extend(
        symbol + " " + " ".join(format(value, ".16g") for value in position)
        for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions)
    )
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    xyz_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    xtob(xyz_path, geo_path, box_lengths=atoms.cell.lengths(), box_angles=atoms.cell.angles())
    return xyz_path, geo_path
