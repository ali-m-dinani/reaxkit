"""
ReaxFF force-field parameter (ffield) handler.

This module provides a handler for parsing ReaxFF ``ffield`` files,
which define all force-field parameters used in ReaxFF simulations.

Unlike most handlers, ``ffield`` data is inherently sectional rather
than tabular and is therefore exposed through per-section tables
instead of a single summary DataFrame.
"""


from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

from reaxkit.io.base_handler import BaseHandler


class FFieldHandler(BaseHandler):
    """
    Parser for ReaxFF force-field parameter files (``ffield``).

    This class parses ReaxFF ``ffield`` files and exposes all force-field
    parameters as structured, section-specific tables suitable for
    inspection, modification, and analysis.

    Parsed Data
    -----------
    Summary table
        The main ``dataframe()`` is intentionally empty for ``ffield`` files.
        All meaningful data is stored in per-section DataFrames.

    Section tables
        Accessible via ``sections`` or ``section_df(name)``, with one table
        per force-field section:

        - ``general``:
          Global ReaxFF parameters (39 fixed parameters), columns:
          ["index", "name", "value", "raw_comment"]

        - ``atom``:
          Atom-type parameters, indexed by atom number, with columns:
          ["symbol", <atom parameter names>]

        - ``bond``:
          Bond parameters, indexed by bond index, with columns:
          ["i", "j", <bond parameter names>]

        - ``off_diagonal``:
          Off-diagonal interaction parameters, indexed by entry number, with columns:
          ["i", "j", <off-diagonal parameter names>]

        - ``angle``:
          Angle parameters, indexed by angle index, with columns:
          ["i", "j", "k", <angle parameter names>]

        - ``torsion``:
          Torsion parameters, indexed by torsion index, with columns:
          ["i", "j", "k", "l", <torsion parameter names>]

        - ``hbond``:
          Hydrogen-bond parameters, indexed by hbond index, with columns:
          ["i", "j", "k", <hbond parameter names>]

    Metadata
        Returned by ``metadata()``, containing counts per section:
        ["n_general_params", "n_atoms", "n_bonds", "n_off_diagonal",
         "n_angles", "n_torsions", "n_hbonds"]

    Notes
    -----
    - Parameter names follow canonical ReaxFF ordering and numbering.
    - Unused parameters are labeled ``n.u.`` with numeric suffixes.
    - Inline comments in the original file are preserved where applicable.
    - Section headers and ordering are detected automatically.
    """

    SECTION_GENERAL = "general"
    SECTION_ATOM = "atom"
    SECTION_BOND = "bond"
    SECTION_OFF_DIAGONAL = "off_diagonal"
    SECTION_ANGLE = "angle"
    SECTION_TORSION = "torsion"
    SECTION_HBOND = "hbond"

    # ---------------- General parameter names --------------------
    # Fixed order, 39 parameters. "Not used" ones are tagged by
    # their 1-based line number in the general section.
    _GENERAL_PARAM_NAMES: List[str] = [
        "overcoord_1",                   # 1
        "overcoord_2",                   # 2
        "valency_angle_conj_1",          # 3
        "triple_bond_stab_1",            # 4
        "triple_bond_stab_2",            # 5
        "not_used_line_num_6",           # 6
        "undercoord_1",                  # 7
        "triple_bond_stab_3",            # 8
        "undercoord_2",                  # 9
        "undercoord_3",                  # 10
        "triple_bond_stab_energy",       # 11
        "taper_radius_lower",            # 12
        "taper_radius_upper",            # 13
        "not_used_line_num_14",          # 14
        "valency_undercoord",            # 15
        "valency_angle_lonepair",        # 16
        "valency_angle",                 # 17
        "valency_angle_param",           # 18
        "not_used_line_num_19",          # 19
        "double_bond_angle",             # 20
        "double_bond_angle_overcoord_1", # 21
        "double_bond_angle_overcoord_2", # 22
        "not_used_line_num_23",          # 23
        "torsion_bo",                    # 24
        "torsion_overcoord_1",           # 25
        "torsion_overcoord_2",           # 26
        "conj_0_not_used",               # 27
        "conj",                          # 28
        "vdw_shielding",                 # 29
        "bo_cutoff_scaled",              # 30
        "valency_angle_conj_2",          # 31
        "overcoord_3",                   # 32
        "overcoord_4",                   # 33
        "valency_lonepair",              # 34
        "not_used_line_num_35",          # 35
        "not_used_line_num_36",          # 36
        "molecular_energy_1_not_used",   # 37
        "molecular_energy_2_not_used",   # 38
        "valency_angle_conj_3",          # 39
    ]

    # ---------------- parameter name file_templates --------------------
    # Atom: 4 × 8 parameters
    _ATOM_PARAM_NAMES_BASE: List[str] = [
        # line 1
        "cov.r", "valency", "a.m", "Rvdw", "Evdw", "gammaEEM", "cov.r2", "#el",
        # line 2
        "alfa", "gammavdW", "valency(2)", "Eunder", "n.u.", "chiEEM", "etaEEM", "n.u.",
        # line 3
        "cov r3", "Elp", "Heat inc.", "13BO1", "13BO2", "13BO3", "n.u.", "n.u.",
        # line 4
        "ov/un", "val1", "n.u.", "val3", "vval4", "n.u.", "n.u.", "n.u.",
    ]

    # Bond: 2 × 8 parameters
    _BOND_PARAM_NAMES_BASE: List[str] = [
        # line 1
        "Edis1", "Edis2", "Edis3", "pbe1", "pbo5", "13corr", "pbo6", "kov",
        # line 2
        "pbe2", "pbo3", "pbo4", "n.u.", "pbo1", "pbo2", "ovcorr", "n.u.",
    ]

    # Off-diagonal
    _OFF_DIAGONAL_PARAM_NAMES: List[str] = [
        "Evdw", "Rvdw", "alfa", "cov.r", "cov.r2", "cov.r3",
    ]

    # Angle
    _ANGLE_PARAM_NAMES: List[str] = [
        "Theta0", "ka", "kb", "pconj", "pv2", "kpenal", "pv3",
    ]

    # Torsion
    _TORSION_PARAM_NAMES_BASE: List[str] = [
        "V1", "V2", "V3", "V2(BO)", "vconj", "n.u.", "n.u.",
    ]

    # H-bond
    _HBOND_PARAM_NAMES: List[str] = [
        "Rhb", "Dehb", "vhb1", "vhb2",
    ]

    # ---------------- init / public API ---------------------------
    def __init__(self, file_path: str | Path = "ffield") -> None:
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

    # ---------------- core parsing -------------------------------
    def _parse(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        path = self.path
        lines = path.read_text().splitlines()

        meta: Dict[str, Any] = {}

        # description: first non-empty line
        for ln in lines:
            if ln.strip():
                meta["description"] = ln.strip()
                break

        sections: Dict[str, pd.DataFrame] = {}

        i = 0
        n_lines = len(lines)

        while i < n_lines:
            line = lines[i]
            lower = line.lower()

            if not line.strip():
                i += 1
                continue

            # --- detect section headers ---
            if "general" in lower and "parameter" in lower:
                n = self._first_int_in_line(line)
                if n is None:
                    i += 1
                    continue
                general_df, i = self._parse_general_section(lines, i + 1, n)
                sections[self.SECTION_GENERAL] = general_df
                meta["n_general_params"] = len(general_df)
                continue

            if "atom" in lower:
                n = self._first_int_in_line(line)
                if n is None:
                    i += 1
                    continue
                atom_df, i = self._parse_atom_section(lines, i + 1, n)
                sections[self.SECTION_ATOM] = atom_df
                meta["n_atoms"] = len(atom_df)
                continue

            if "bond" in lower and "off" not in lower and "hydrogen" not in lower:
                n = self._first_int_in_line(line)
                if n is None:
                    i += 1
                    continue
                bond_df, i = self._parse_bond_section(lines, i + 1, n)
                sections[self.SECTION_BOND] = bond_df
                meta["n_bonds"] = len(bond_df)
                continue

            if "off-diagonal" in lower or "off diagonal" in lower:
                n = self._first_int_in_line(line)
                if n is None:
                    i += 1
                    continue
                off_df, i = self._parse_off_diagonal_section(lines, i + 1, n)
                sections[self.SECTION_OFF_DIAGONAL] = off_df
                meta["n_off_diagonal"] = len(off_df)
                continue

            if "angle" in lower:
                n = self._first_int_in_line(line)
                if n is None:
                    i += 1
                    continue
                angle_df, i = self._parse_angle_section(lines, i + 1, n)
                sections[self.SECTION_ANGLE] = angle_df
                meta["n_angles"] = len(angle_df)
                continue

            if "torsion" in lower:
                n = self._first_int_in_line(line)
                if n is None:
                    i += 1
                    continue
                torsion_df, i = self._parse_torsion_section(lines, i + 1, n)
                sections[self.SECTION_TORSION] = torsion_df
                meta["n_torsions"] = len(torsion_df)
                continue

            if "hydrogen" in lower and "bond" in lower:
                n = self._first_int_in_line(line)
                if n is None:
                    i += 1
                    continue
                hbond_df, i = self._parse_hbond_section(lines, i + 1, n)
                sections[self.SECTION_HBOND] = hbond_df
                meta["n_hbonds"] = len(hbond_df)
                continue

            i += 1

        self._sections = sections

        # Summary DataFrame for ffield is intentionally empty
        df = pd.DataFrame()
        return df, meta

    # ---------------- section parsers ----------------------------
    def _parse_general_section(
        self, lines: List[str], start: int, n_params: int
    ) -> Tuple[pd.DataFrame, int]:
        """Parse General parameters using fixed names, not inline comments."""
        expected = len(self._GENERAL_PARAM_NAMES)
        if n_params == expected:
            print("[FFieldHandler Check] Number of general parameters is 39 (expected).")
        else:
            print(
                f"[FFieldHandler Check] WARNING: expected {expected} general parameters, "
                f"but header says {n_params}."
            )

        records: List[Dict[str, Any]] = []

        for idx in range(n_params):
            if start + idx >= len(lines):
                break
            raw_line = lines[start + idx]

            if "!" in raw_line:
                left, comment = raw_line.split("!", 1)
                raw_comment = comment.strip()
            else:
                left = raw_line
                raw_comment = ""

            tokens = left.split()
            value = float(tokens[0]) if tokens else float("nan")

            if idx < expected:
                name = self._GENERAL_PARAM_NAMES[idx]
            else:
                name = f"general_param_{idx + 1}"

            records.append(
                {
                    "index": idx + 1,
                    "name": name,
                    "value": value,
                    "raw_comment": raw_comment,
                }
            )

        df = pd.DataFrame.from_records(records).set_index("index")
        end = start + n_params
        return df, end

    def _parse_atom_section(
        self, lines: List[str], start: int, n_atoms: int
    ) -> Tuple[pd.DataFrame, int]:
        names = self._number_unused_titles(self._ATOM_PARAM_NAMES_BASE)
        n_per_atom = len(names)

        records: List[Dict[str, Any]] = []
        n_lines = len(lines)

        # Skip the 4 description lines after the atom header
        i = min(start + 3, n_lines)

        for atom_idx in range(1, n_atoms + 1):
            values: List[float] = []
            atom_no: Optional[int] = None
            symbol: Optional[str] = None
            first_line = True

            while len(values) < n_per_atom and i < n_lines:
                line = lines[i]
                i += 1

                data_part = line.split("!", 1)[0]
                tokens = data_part.split()
                if not tokens:
                    continue

                if first_line:
                    j = 0
                    try:
                        atom_no = int(tokens[0])
                        j = 1
                    except ValueError:
                        atom_no = atom_idx
                        j = 0

                    if j < len(tokens):
                        try:
                            float(tokens[j])
                        except ValueError:
                            symbol = tokens[j]
                            j += 1

                    for tok in tokens[j:]:
                        try:
                            values.append(float(tok))
                        except ValueError:
                            continue

                    first_line = False
                else:
                    for tok in tokens:
                        try:
                            values.append(float(tok))
                        except ValueError:
                            continue

            if len(values) < n_per_atom:
                values.extend([float("nan")] * (n_per_atom - len(values)))

            record: Dict[str, Any] = {
                "atom_index": atom_no if atom_no is not None else atom_idx,
                "symbol": symbol,
            }
            record.update({name: val for name, val in zip(names, values)})
            records.append(record)

        df = pd.DataFrame.from_records(records).set_index("atom_index")
        return df, i

    def _parse_bond_section(
        self, lines: List[str], start: int, n_bonds: int
    ) -> Tuple[pd.DataFrame, int]:
        names = self._number_unused_titles(self._BOND_PARAM_NAMES_BASE)
        n_per_bond = len(names)

        records: List[Dict[str, Any]] = []
        n_lines = len(lines)

        # Skip the 2 description lines after the bond header
        i = min(start + 1, n_lines)

        for bond_idx in range(1, n_bonds + 1):
            values: List[float] = []
            at_i: Optional[int] = None
            at_j: Optional[int] = None
            first_line = True

            while len(values) < n_per_bond and i < n_lines:
                line = lines[i]
                i += 1

                data_part = line.split("!", 1)[0]
                tokens = data_part.split()
                if not tokens:
                    continue

                if first_line:
                    j = 0
                    if len(tokens) >= 1:
                        try:
                            at_i = int(tokens[0])
                            j = 1
                        except ValueError:
                            j = 0
                    if len(tokens) >= 2 and j == 1:
                        try:
                            at_j = int(tokens[1])
                            j = 2
                        except ValueError:
                            pass

                    for tok in tokens[j:]:
                        try:
                            values.append(float(tok))
                        except ValueError:
                            continue

                    first_line = False
                else:
                    for tok in tokens:
                        try:
                            values.append(float(tok))
                        except ValueError:
                            continue

            if len(values) < n_per_bond:
                values.extend([float("nan")] * (n_per_bond - len(values)))

            record: Dict[str, Any] = {
                "bond_index": bond_idx,
                "i": at_i,
                "j": at_j,
            }
            record.update({name: val for name, val in zip(names, values)})
            records.append(record)

        df = pd.DataFrame.from_records(records).set_index("bond_index")
        return df, i

    def _parse_off_diagonal_section(
        self, lines: List[str], start: int, n_entries: int
    ) -> Tuple[pd.DataFrame, int]:
        names = list(self._OFF_DIAGONAL_PARAM_NAMES)
        n_per = len(names)

        records: List[Dict[str, Any]] = []
        i = start
        n_lines = len(lines)

        for idx in range(1, n_entries + 1):
            values: List[float] = []
            at_i: Optional[int] = None
            at_j: Optional[int] = None
            first_line = True

            while len(values) < n_per and i < n_lines:
                line = lines[i]
                i += 1

                data_part = line.split("!", 1)[0]
                tokens = data_part.split()
                if not tokens:
                    continue

                if first_line:
                    j = 0
                    if len(tokens) >= 1:
                        try:
                            at_i = int(tokens[0])
                            j = 1
                        except ValueError:
                            j = 0
                    if len(tokens) >= 2 and j == 1:
                        try:
                            at_j = int(tokens[1])
                            j = 2
                        except ValueError:
                            pass

                    for tok in tokens[j:]:
                        try:
                            values.append(float(tok))
                        except ValueError:
                            continue

                    first_line = False
                else:
                    for tok in tokens:
                        try:
                            values.append(float(tok))
                        except ValueError:
                            continue

            if len(values) < n_per:
                values.extend([float("nan")] * (n_per - len(values)))

            record: Dict[str, Any] = {
                "offdiag_index": idx,
                "i": at_i,
                "j": at_j,
            }
            record.update({name: val for name, val in zip(names, values)})
            records.append(record)

        df = pd.DataFrame.from_records(records).set_index("offdiag_index")
        return df, i

    def _parse_angle_section(
        self, lines: List[str], start: int, n_angles: int
    ) -> Tuple[pd.DataFrame, int]:
        names = list(self._ANGLE_PARAM_NAMES)
        n_per = len(names)

        records: List[Dict[str, Any]] = []
        i = start
        n_lines = len(lines)

        for idx in range(1, n_angles + 1):
            values: List[float] = []
            at_i: Optional[int] = None
            at_j: Optional[int] = None
            at_k: Optional[int] = None
            first_line = True

            while len(values) < n_per and i < n_lines:
                line = lines[i]
                i += 1

                data_part = line.split("!", 1)[0]
                tokens = data_part.split()
                if not tokens:
                    continue

                if first_line:
                    j = 0
                    if len(tokens) >= 1:
                        try:
                            at_i = int(tokens[0])
                            j = 1
                        except ValueError:
                            j = 0
                    if len(tokens) >= 2 and j == 1:
                        try:
                            at_j = int(tokens[1])
                            j = 2
                        except ValueError:
                            pass
                    if len(tokens) >= 3 and j == 2:
                        try:
                            at_k = int(tokens[2])
                            j = 3
                        except ValueError:
                            pass

                    for tok in tokens[j:]:
                        try:
                            values.append(float(tok))
                        except ValueError:
                            continue

                    first_line = False
                else:
                    for tok in tokens:
                        try:
                            values.append(float(tok))
                        except ValueError:
                            continue

            if len(values) < n_per:
                values.extend([float("nan")] * (n_per - len(values)))

            record: Dict[str, Any] = {
                "angle_index": idx,
                "i": at_i,
                "j": at_j,
                "k": at_k,
            }
            record.update({name: val for name, val in zip(names, values)})
            records.append(record)

        df = pd.DataFrame.from_records(records).set_index("angle_index")
        return df, i

    def _parse_torsion_section(
        self, lines: List[str], start: int, n_torsions: int
    ) -> Tuple[pd.DataFrame, int]:
        names = self._number_unused_titles(self._TORSION_PARAM_NAMES_BASE)
        n_per = len(names)

        records: List[Dict[str, Any]] = []
        i = start
        n_lines = len(lines)

        for idx in range(1, n_torsions + 1):
            values: List[float] = []
            at_i: Optional[int] = None
            at_j: Optional[int] = None
            at_k: Optional[int] = None
            at_l: Optional[int] = None
            first_line = True

            while len(values) < n_per and i < n_lines:
                line = lines[i]
                i += 1

                data_part = line.split("!", 1)[0]
                tokens = data_part.split()
                if not tokens:
                    continue

                if first_line:
                    j = 0
                    if len(tokens) >= 1:
                        try:
                            at_i = int(tokens[0])
                            j = 1
                        except ValueError:
                            j = 0
                    if len(tokens) >= 2 and j == 1:
                        try:
                            at_j = int(tokens[1])
                            j = 2
                        except ValueError:
                            pass
                    if len(tokens) >= 3 and j == 2:
                        try:
                            at_k = int(tokens[2])
                            j = 3
                        except ValueError:
                            pass
                    if len(tokens) >= 4 and j == 3:
                        try:
                            at_l = int(tokens[3])
                            j = 4
                        except ValueError:
                            pass

                    for tok in tokens[j:]:
                        try:
                            values.append(float(tok))
                        except ValueError:
                            continue

                    first_line = False
                else:
                    for tok in tokens:
                        try:
                            values.append(float(tok))
                        except ValueError:
                            continue

            if len(values) < n_per:
                values.extend([float("nan")] * (n_per - len(values)))

            record: Dict[str, Any] = {
                "torsion_index": idx,
                "i": at_i,
                "j": at_j,
                "k": at_k,
                "l": at_l,
            }
            record.update({name: val for name, val in zip(names, values)})
            records.append(record)

        df = pd.DataFrame.from_records(records).set_index("torsion_index")
        return df, i

    def _parse_hbond_section(
        self, lines: List[str], start: int, n_hbonds: int
    ) -> Tuple[pd.DataFrame, int]:
        names = list(self._HBOND_PARAM_NAMES)
        n_per = len(names)

        records: List[Dict[str, Any]] = []
        i = start
        n_lines = len(lines)

        for idx in range(1, n_hbonds + 1):
            values: List[float] = []
            at_i: Optional[int] = None
            at_j: Optional[int] = None
            at_k: Optional[int] = None
            first_line = True

            while len(values) < n_per and i < n_lines:
                line = lines[i]
                i += 1

                data_part = line.split("!", 1)[0]
                tokens = data_part.split()
                if not tokens:
                    continue

                if first_line:
                    j = 0
                    if len(tokens) >= 1:
                        try:
                            at_i = int(tokens[0])
                            j = 1
                        except ValueError:
                            j = 0
                    if len(tokens) >= 2 and j == 1:
                        try:
                            at_j = int(tokens[1])
                            j = 2
                        except ValueError:
                            pass
                    if len(tokens) >= 3 and j == 2:
                        try:
                            at_k = int(tokens[2])
                            j = 3
                        except ValueError:
                            pass

                    for tok in tokens[j:]:
                        try:
                            values.append(float(tok))
                        except ValueError:
                            continue

                    first_line = False
                else:
                    for tok in tokens:
                        try:
                            values.append(float(tok))
                        except ValueError:
                            continue

            if len(values) < n_per:
                values.extend([float("nan")] * (n_per - len(values)))

            record: Dict[str, Any] = {
                "hbond_index": idx,
                "i": at_i,
                "j": at_j,
                "k": at_k,
            }
            record.update({name: val for name, val in zip(names, values)})
            records.append(record)

        df = pd.DataFrame.from_records(records).set_index("hbond_index")
        return df, i

    # ---------------- helpers ------------------------------------
    @staticmethod
    def _first_int_in_line(line: str) -> Optional[int]:
        for tok in line.split():
            try:
                return int(tok)
            except ValueError:
                continue
        return None

    @staticmethod
    def _number_unused_titles(
        names: List[str],
        label: str = "n.u.",
    ) -> List[str]:
        result: List[str] = []
        counter = 0
        for name in names:
            if name == label:
                counter += 1
                result.append(f"{label}{counter}")
            else:
                result.append(name)
        return result
