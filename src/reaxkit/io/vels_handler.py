# reaxkit/io/vels_handler.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from reaxkit.io.template_handler import TemplateHandler


class VelsHandler(TemplateHandler):
    SECTION_COORDS = "Atom coordinates"
    SECTION_VELS = "Atom velocities"
    SECTION_ACCELS = "Atom accelerations"
    SECTION_PREV_ACCELS = "Previous atom accelerations"

    def __init__(self, file_path: str | Path = "vels") -> None:
        super().__init__(file_path)
        self._sections: Dict[str, pd.DataFrame] = {}

    @property
    def sections(self) -> Dict[str, pd.DataFrame]:
        if not self._parsed:
            self.parse()
        return self._sections

    def section_df(self, name: str) -> pd.DataFrame:
        if not self._parsed:
            self.parse()
        return self._sections[name]

    def _parse(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        lines = self.path.read_text().splitlines()
        meta: Dict[str, Any] = {}
        sections: Dict[str, pd.DataFrame] = {}

        def next_nonempty(idx: int) -> int:
            while idx < len(lines) and not lines[idx].strip():
                idx += 1
            return idx

        def floats_from_line(s: str, n: int) -> list[float]:
            # IMPORTANT: handle Fortran D exponents like 0.20D+01
            s = s.replace("D", "E").replace("d", "E")
            out: list[float] = []
            for tok in s.replace(",", " ").split():
                tok2 = tok.replace("D", "E").replace("d", "E")
                try:
                    out.append(float(tok2))
                except ValueError:
                    pass
                if len(out) == n:
                    break
            return out

        def first_int_in_line(s: str) -> int | None:
            for tok in s.split():
                try:
                    return int(tok)
                except ValueError:
                    continue
            return None

        def parse_lattice(idx: int) -> tuple[dict[str, float], int]:
            idx = next_nonempty(idx)
            abc = floats_from_line(lines[idx], 3)

            idx += 1
            idx = next_nonempty(idx)
            ang = floats_from_line(lines[idx], 3)

            if len(abc) != 3 or len(ang) != 3:
                raise ValueError("Could not parse lattice parameters (expected two lines: 3 + 3 floats).")

            lat = {"a": abc[0], "b": abc[1], "c": abc[2], "alpha": ang[0], "beta": ang[1], "gamma": ang[2]}
            return lat, idx + 1

        def parse_coords(idx: int, n_atoms: int, section_name: str) -> tuple[pd.DataFrame, int]:
            idx = next_nonempty(idx)
            rows = []
            for a in range(1, n_atoms + 1):
                idx = next_nonempty(idx)
                if idx >= len(lines):
                    raise ValueError(
                        f"[vels] Truncated section: '{section_name}'. "
                        f"Expected {n_atoms} atom lines, but file ended at atom {a - 1}."
                    )

                s = lines[idx].strip()
                low = s.lower()
                if (
                        "md-temperature" in low
                        or "atom velocities" in low
                        or ("atom accelerations" in low)
                        or "lattice parameters" in low
                ):
                    raise ValueError(
                        f"[vels] Truncated section: '{section_name}'. "
                        f"Expected {n_atoms} atom lines, but only found {a - 1}. "
                        f"Next header encountered early at line {idx + 1}: {s!r}"
                    )

                xyz = floats_from_line(s, 3)
                if len(xyz) != 3:
                    raise ValueError(
                        f"[vels] Bad numeric line in section '{section_name}' at atom {a}. "
                        f"Line {idx + 1}: {s!r}"
                    )

                symbol = s.split()[-1] if s.split() else ""
                rows.append({"atom_index": a, "x": xyz[0], "y": xyz[1], "z": xyz[2], "symbol": symbol})
                idx += 1

            return pd.DataFrame(rows), idx

        def parse_xyz3(idx: int, n_atoms: int, c1: str, c2: str, c3: str, section_name: str) -> tuple[
            pd.DataFrame, int]:
            idx = next_nonempty(idx)
            rows = []

            for a in range(1, n_atoms + 1):
                idx = next_nonempty(idx)
                if idx >= len(lines):
                    raise ValueError(
                        f"[vels] Truncated section: '{section_name}'. "
                        f"Expected {n_atoms} atom lines, but file ended at atom {a - 1}."
                    )

                s = lines[idx].strip()
                low = s.lower()

                # If we accidentally hit the next header, the section is shortened/truncated.
                if (
                        "md-temperature" in low
                        or "atom coordinates" in low
                        or "atom velocities" in low
                        or ("atom accelerations" in low)
                        or "lattice parameters" in low
                ):
                    raise ValueError(
                        f"[vels] Truncated section: '{section_name}'. "
                        f"Expected {n_atoms} atom lines, but only found {a - 1}. "
                        f"Next header encountered early at line {idx + 1}: {s!r}"
                    )

                v = floats_from_line(s, 3)
                if len(v) != 3:
                    raise ValueError(
                        f"[vels] Bad numeric line in section '{section_name}' at atom {a}. "
                        f"Line {idx + 1}: {s!r}"
                    )

                rows.append({"atom_index": a, c1: v[0], c2: v[1], c3: v[2]})
                idx += 1

            return pd.DataFrame(rows), idx

        def parse_temperature(idx: int) -> tuple[float, int]:
            idx = next_nonempty(idx)
            v = floats_from_line(lines[idx], 1)
            if not v:
                raise ValueError(f"Could not parse MD-temperature from line: {lines[idx]!r}")
            return float(v[0]), idx + 1

        i = 0
        n_atoms: int | None = None
        prev_acc_present = False

        while i < len(lines):
            s = lines[i].strip()
            low = s.lower()

            if not s:
                i += 1
                continue

            if "lattice parameters" in low:
                lat, i = parse_lattice(i + 1)
                meta["lattice_parameters"] = lat
                continue

            if "atom coordinates" in low:
                n_atoms = first_int_in_line(s)
                if n_atoms is None:
                    raise ValueError("Could not read number of atoms from 'Atom coordinates' header.")
                df, i = parse_coords(i + 1, n_atoms, self.SECTION_COORDS)
                sections[self.SECTION_COORDS] = df
                continue

            if "atom velocities" in low:
                if n_atoms is None:
                    raise ValueError("Found velocities before coordinates; cannot infer number of atoms.")
                df, i = parse_xyz3(i + 1, n_atoms, "vx", "vy", "vz", self.SECTION_VELS)
                sections[self.SECTION_VELS] = df
                continue

            if "atom accelerations" in low and "previous" not in low:
                if n_atoms is None:
                    raise ValueError("Found accelerations before coordinates; cannot infer number of atoms.")
                df, i = parse_xyz3(i + 1, n_atoms, "ax", "ay", "az", self.SECTION_ACCELS)
                sections[self.SECTION_ACCELS] = df
                continue

            if "previous atom accelerations" in low:
                if n_atoms is None:
                    raise ValueError("Found previous accelerations before coordinates; cannot infer number of atoms.")
                df, i = parse_xyz3(i + 1, n_atoms, "ax", "ay", "az", self.SECTION_PREV_ACCELS)
                sections[self.SECTION_PREV_ACCELS] = df
                prev_acc_present = True
                continue

            if "md-temperature" in low or "md temperature" in low:
                t, i = parse_temperature(i + 1)
                meta["md_temperature_K"] = t
                continue

            i += 1

        if not prev_acc_present:
            sections[self.SECTION_PREV_ACCELS] = pd.DataFrame(columns=["atom_index", "ax", "ay", "az"])

        self._sections = sections
        return pd.DataFrame(), meta
