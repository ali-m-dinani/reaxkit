"""Read individual complete AMS KF/RKF geometry frames."""

import numpy as np
from ase import Atoms

from reaxkit.engine.ams.adapter import AMSAdapter
from reaxkit.engine.ams.rkf_handler import RKFHandler


def read_frame(source, selected):
    """Read coordinates and full cell vectors with the native History units."""
    adapter = AMSAdapter()
    kf = RKFHandler(source).kf()
    count = adapter._history_frame_count(kf)
    steps = adapter._step_numbers(kf)
    candidates = range(count - 1, -1, -1) if selected == -1 else [selected]
    for source_index in candidates:
        if source_index >= count:
            break
        step = int(steps[source_index]) if source_index < len(steps) else None
        layout = {}

        def read_variable(prefix):
            return adapter._read_history_source_variable(
                kf, prefix, source_index, step_number=step, layout_cache=layout,
            )

        raw = None
        for prefix in ("Coords", "Coordinates"):
            raw = read_variable(prefix)
            if raw is not None:
                coordinate_factor = adapter._history_length_factor(layout.get(prefix))
                break
        if raw is None:
            continue
        positions = np.asarray(raw, dtype=float).ravel()
        names = read_variable("Atom names")
        names = adapter._extract_atom_names_raw(kf, {} if names is None else {"Atom names": names})
        if names is None:
            raise ValueError("AMS frame has no element names for GEO export.")
        symbols = adapter._parse_atom_names(names)
        if positions.size != len(symbols) * 3 or positions.size == 0:
            continue
        axes = None
        for prefix in ("Unit cell axes", "LatticeVectors", "Lattice vectors"):
            raw_axes = read_variable(prefix)
            if raw_axes is not None:
                values = np.asarray(raw_axes, dtype=float).ravel()
                factor = adapter._history_length_factor(layout.get(prefix))
                if values.size == 9:
                    axes = values.reshape(3, 3) * factor
                elif values.size == 3:
                    angles = read_variable("Unit cell angles")
                    if angles is None or not np.allclose(angles, 90):
                        raise ValueError("AMS nonorthogonal export requires full lattice vectors.")
                    axes = np.diag(values * factor)
                break
        else:
            first_step = int(steps[0]) if len(steps) else None
            dynamic_cell = source_index > 0 and any(
                adapter._read_history_source_variable(kf, prefix, 0, step_number=first_step) is not None
                for prefix in ("Unit cell axes", "LatticeVectors", "Lattice vectors")
            )
            if dynamic_cell:
                continue
            axes = adapter._molecule_lattice_vectors(kf)
        if axes is None:
            if selected == -1 and source_index == count - 1:
                continue
            raise ValueError("AMS frame requires a complete 3D lattice for GEO export.")
        atoms = Atoms(symbols, positions=positions.reshape(-1, 3) * coordinate_factor, cell=axes, pbc=True)
        atoms.info["source_frame_index"] = source_index
        return atoms
    raise ValueError(f"No complete frame {selected!r} found in {source}.")
