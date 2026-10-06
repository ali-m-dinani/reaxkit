"""Extract complete ReaxFF trajectory frames as XYZ and GEO structures."""

from __future__ import annotations

from pathlib import Path
import math

from ase.data import atomic_numbers

from reaxkit.engine.reaxff.generators.geo_generator import xtob
from reaxkit.engine.reaxff.io.xmolout_handler import _parse_xmolout_header


def extract_frame(
    trajectory: str | Path,
    *,
    frame: int | str = "last",
    xyz_file: str | Path = "frame.xyz",
    geo_file: str | Path = "frame.geo",
) -> tuple[Path, Path]:
    """Write a zero-based frame (or ``last``/``-1``) from an xmolout.

    Scan without caching or loading the full trajectory. An EOF inside a frame
    leaves the preceding complete frame available for ``last``. Explicit frame
    selection never falls back to a different frame. Coordinates and atom order
    are preserved; additional per-atom columns are omitted. These are geometry
    files, not a restart containing velocities or thermostat state.
    """
    if str(frame).lower() == "last":
        selected_index = -1
    else:
        try:
            selected_index = int(str(frame))
        except ValueError as exc:
            raise ValueError("frame must be a zero-based integer, -1, or 'last'.") from exc
    if selected_index < -1:
        raise ValueError("frame must be nonnegative, -1, or 'last'.")

    source = Path(trajectory)
    xyz_path, geo_path = Path(xyz_file), Path(geo_file)
    if len({source.resolve(), xyz_path.resolve(), geo_path.resolve()}) != 3:
        raise ValueError("Trajectory, XYZ output and GEO output must be different files.")

    selected = None
    frame_index = 0
    with source.open(encoding="utf-8") as stream:
        while True:
            count_line = stream.readline()
            if not count_line:
                break
            if not count_line.strip():
                continue
            try:
                atom_count = int(count_line.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid atom count at frame {frame_index}.") from exc
            if atom_count <= 0:
                raise ValueError(f"Atom count must be positive at frame {frame_index}.")
            header = stream.readline()
            if not header:
                break
            coordinate_offset = stream.tell()
            complete = True
            for atom_index in range(atom_count):
                atom_line = stream.readline()
                if not atom_line:
                    complete = False
                    break
                if not atom_line.endswith("\n"):
                    fields = atom_line.split()
                    try:
                        complete = len(fields) >= 4 and all(math.isfinite(float(value)) for value in fields[1:4])
                    except ValueError:
                        complete = False
                    if not complete:
                        break
            if not complete:
                break
            selected = (frame_index, atom_count, header, coordinate_offset)
            if frame_index == selected_index:
                break
            frame_index += 1

        if selected is None or (selected_index >= 0 and selected[0] != selected_index):
            raise ValueError(f"No complete frame {frame!r} found in {source}.")
        frame_index, atom_count, header, coordinate_offset = selected
        _, _, numeric_values = _parse_xmolout_header(
            header, path=source, frame_index=frame_index, line_number=None,
        )
        cell = numeric_values[1:]
        if not all(math.isfinite(value) for value in cell):
            raise ValueError(f"Non-finite cell parameters at frame {frame_index}.")
        if any(value <= 0 for value in cell[:3]) or any(not 0 < value < 180 for value in cell[3:]):
            raise ValueError(f"Invalid cell lengths or angles at frame {frame_index}.")
        cos_alpha, cos_beta, cos_gamma = [math.cos(math.radians(value)) for value in cell[3:]]
        volume_factor = 1 + 2 * cos_alpha * cos_beta * cos_gamma - cos_alpha**2 - cos_beta**2 - cos_gamma**2
        if volume_factor <= 0:
            raise ValueError(f"Degenerate cell at frame {frame_index}.")
        stream.seek(coordinate_offset)
        coordinates = []
        for atom_index in range(atom_count):
            fields = stream.readline().split()
            try:
                valid = len(fields) >= 4 and fields[0] in atomic_numbers and all(
                    math.isfinite(float(value)) for value in fields[1:4]
                )
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f"Invalid coordinates at frame {frame_index}, atom {atom_index + 1}.")
            coordinates.append(" ".join(fields[:4]))

    xyz_text = f"{atom_count}\n{header.strip()}\n" + "\n".join(coordinates) + "\n"
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    xyz_path.write_text(xyz_text, encoding="utf-8")
    xtob(xyz_path, geo_path, box_lengths=cell[:3], box_angles=cell[3:])
    return xyz_path, geo_path
