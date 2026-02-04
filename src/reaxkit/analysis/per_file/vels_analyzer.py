"""
vels / moldyn.vel / molsav analysis utilities.

This module provides a unified interface for accessing atomic coordinates,
velocities, and accelerations stored in ReaxFF velocity output files via
``VelsHandler``.

Typical use cases include:

- retrieving atomic coordinates or velocities for selected atoms
- extracting acceleration histories for diagnostics or plotting
- accessing file metadata (timestep, atom count, sections present)
"""


from __future__ import annotations

from typing import Literal, Sequence

import pandas as pd

from reaxkit.io.handlers.vels_handler import VelsHandler


VelsKey = Literal[
    "metadata",
    "coordinates",
    "velocities",
    "accelerations",
    "prev_accelerations",
]


def get_vels_data(
    handler: VelsHandler,
    key: VelsKey,
    *,
    atoms: Sequence[int] | None = None,
) -> pd.DataFrame | dict:
    """
    Retrieve metadata or a selected atomic table from a velocity output file.

    Works on
    --------
    VelsHandler — ``vels`` / ``moldyn.vel`` / ``molsav``

    Parameters
    ----------
    handler : VelsHandler
        Parsed velocity file handler.
    key : {"metadata", "coordinates", "velocities", "accelerations", "prev_accelerations"}
        Section to retrieve:
        - ``metadata``: file-level information (returned as a dict)
        - ``coordinates``: atomic positions
        - ``velocities``: atomic velocities
        - ``accelerations``: atomic accelerations
        - ``prev_accelerations``: accelerations from the previous step
    atoms : sequence of int, optional
        1-based atom indices to include. If None, all atoms are returned.
        Ignored when ``key="metadata"``.

    Returns
    -------
    pandas.DataFrame or dict
        If ``key="metadata"``, returns a metadata dictionary.
        Otherwise, returns a DataFrame with one row per atom containing the
        requested quantities and an ``atom_index`` column.

    Examples
    --------
    >>> from reaxkit.io.handlers.vels_handler import VelsHandler
    >>> from reaxkit.analysis.per_file.vels_analyzer import get_vels_data
    >>> h = VelsHandler("moldyn.vel")
    >>> v = get_vels_data(h, "velocities", atoms=[1, 2, 3])
    >>> meta = get_vels_data(h, "metadata")
    """
    if key == "metadata":
        return handler.metadata()

    if key == "coordinates":
        df = handler.section_df(handler.SECTION_COORDS).copy()
    elif key == "velocities":
        df = handler.section_df(handler.SECTION_VELS).copy()
    elif key == "accelerations":
        df = handler.section_df(handler.SECTION_ACCELS).copy()
    elif key == "prev_accelerations":
        df = handler.section_df(handler.SECTION_PREV_ACCELS).copy()
    else:
        raise ValueError(f"Unknown vels key: {key}")

    if atoms:
        # 1-based indices expected (matches how handler stores atom_index)
        df = df[df["atom_index"].isin(list(atoms))].copy()

    return df.reset_index(drop=True)
