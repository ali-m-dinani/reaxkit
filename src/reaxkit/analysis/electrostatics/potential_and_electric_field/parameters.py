"""Force-field parameters required by the local electrostatics analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from reaxkit.engine.common.io.ffield_handler import FFieldHandler

from .physics import build_taper_coefficients


@dataclass(frozen=True)
class ReaxFFCoulombParameters:
    lower_taper_radius: float
    upper_taper_radius: float
    gamma_by_symbol: Mapping[str, float]
    source: str = ""

    @property
    def taper_coefficients(self) -> tuple[float, ...]:
        return build_taper_coefficients(self.lower_taper_radius, self.upper_taper_radius)

    @classmethod
    def from_ffield(cls, path: str | Path) -> "ReaxFFCoulombParameters":
        handler = FFieldHandler(path)
        general = handler.section_df(handler.SECTION_GENERAL)
        atoms = handler.section_df(handler.SECTION_ATOM).sort_index()
        def value(name: str) -> float:
            matches = general.loc[general["name"].eq(name), "value"]
            if len(matches) != 1:
                raise ValueError(f"Expected one ffield general parameter named {name!r}.")
            return float(matches.iloc[0])
        gammas = {
            str(row.symbol).strip().casefold(): float(row.gammaEEM)
            for row in atoms.itertuples()
        }
        return cls(value("taper_radius_lower"), value("taper_radius_upper"), gammas, str(Path(path)))

    def gamma_values(self, labels: Sequence[object]) -> np.ndarray:
        try:
            values = [self.gamma_by_symbol[str(label).strip().casefold()] for label in labels]
        except KeyError as exc:
            raise ValueError(f"No gammaEEM parameter for atom/probe label {exc.args[0]!r}.") from None
        values = np.asarray(values, dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("All gammaEEM values must be finite and positive.")
        return values
