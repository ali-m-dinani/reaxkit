"""this is an analyzer for vels, moldyn.vel, and molsav files"""

from __future__ import annotations

from typing import Literal, Sequence

import pandas as pd

from reaxkit.io.vels_handler import VelsHandler


VelsKey = Literal[
    "metadata",
    "coordinates",
    "velocities",
    "accelerations",
    "prev_accelerations",
]


def vels_get(
    handler: VelsHandler,
    key: VelsKey,
    *,
    atoms: Sequence[int] | None = None,
) -> pd.DataFrame | dict:
    """
    Get metadata or a specific dataframe from a VelsHandler.

    key:
      - "metadata"
      - "coordinates"
      - "velocities"
      - "accelerations"
      - "prev_accelerations"

    atoms:
      Optional list of 1-based atom indices to slice the returned dataframe.
      (Ignored for key="metadata".)
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
