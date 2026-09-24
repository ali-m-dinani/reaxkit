"""Single-pass molecular population input, retaining one iteration."""

import numpy as np
import pandas as pd
from reaxkit.domain.data_models import MolecularAnalysisData


def iter_molecular_data(path, selected=None, reporter=None):
    wanted = None if selected is None else set(selected)
    current = None
    rows, totals = [], {}
    source = 0

    def payload():
        result = MolecularAnalysisData(iterations=np.array([current]),
            molecular_species=pd.DataFrame(rows, columns=["iter", "molecular_formula", "freq", "molecular_mass"]),
            totals=pd.DataFrame([{"iter": current, **totals}]) if totals else pd.DataFrame())
        result.source_frame_indices = np.array([source])
        return result

    with open(path, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            parts = line.split()
            if len(parts) >= 5 and "x" in parts:
                try:
                    iteration, frequency, mass = int(parts[0]), int(parts[1]), float(parts[-1])
                    formula = parts[parts.index("x") + 1]
                except (ValueError, IndexError):
                    continue
                if current is not None and iteration != current:
                    if iteration < current:
                        raise ValueError("Molecular iterations must be monotonically increasing for streaming.")
                    if wanted is None or source in wanted:
                        yield payload()
                    source += 1
                    rows, totals = [], {}
                    if wanted and source > max(wanted):
                        return
                current = iteration
                rows.append(dict(iter=iteration, molecular_formula=formula, freq=frequency, molecular_mass=mass))
            elif line.startswith("Total number of molecules"):
                totals["total_molecules"] = int(parts[-1])
            elif line.startswith("Total number of atoms"):
                totals["total_atoms"] = int(parts[-1])
            elif line.startswith("Total system"):
                totals["total_molecular_mass"] = float(parts[-1])
        if current is not None and (wanted is None or source in wanted):
            yield payload()
