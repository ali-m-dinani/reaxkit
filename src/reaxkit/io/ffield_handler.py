from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

from reaxkit.io.template_handler import TemplateHandler


class FFieldHandler(TemplateHandler):
    """
    Handler for ReaxFF `ffield` files.

    Design:
    - The main DataFrame (`dataframe()`) is intentionally empty.
      All useful data lives in per-section DataFrames exposed via
      `sections` or convenience properties (general_df, atom_df, ...).
    - Sections currently supported:
        * general
        * atom
        * bond
        * off_diagonal
        * angle
        * torsion
        * hbond
    """

    SECTION_GENERAL = "general"
    SECTION_ATOM = "atom"
    SECTION_BOND = "bond"
    SECTION_OFF_DIAGONAL = "off_diagonal"
    SECTION_ANGLE = "angle"
    SECTION_TORSION = "torsion"
    SECTION_HBOND = "hbond"

    # ---------------- parameter name templates --------------------
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
        # per-section dataframes, filled in _parse()
        self._sections: Dict[str, pd.DataFrame] = {}

    # we can use TemplateHandler/FileHandler.dataframe() directly;
    # it will call parse() -> _parse() and return self._df.

    @property
    def sections(self) -> Dict[str, pd.DataFrame]:
        """Mapping of section name -> section DataFrame."""
        if not self._parsed:
            self.parse()
        return self._sections

    def section_df(self, name: str) -> pd.DataFrame:
        """Convenience accessor for a single section by name."""
        if not self._parsed:
            self.parse()
        return self._sections[name]

    # Shorthands
    @property
    def general_df(self) -> pd.DataFrame:
        return self.section_df(self.SECTION_GENERAL)

    @property
    def atom_df(self) -> pd.DataFrame:
        return self.section_df(self.SECTION_ATOM)

    @property
    def bond_df(self) -> pd.DataFrame:
        return self.section_df(self.SECTION_BOND)

    @property
    def off_diagonal_df(self) -> pd.DataFrame:
        return self.section_df(self.SECTION_OFF_DIAGONAL)

    @property
    def angle_df(self) -> pd.DataFrame:
        return self.section_df(self.SECTION_ANGLE)

    @property
    def torsion_df(self) -> pd.DataFrame:
        return self.section_df(self.SECTION_TORSION)

    @property
    def hbond_df(self) -> pd.DataFrame:
        return self.section_df(self.SECTION_HBOND)

    # ---------------- core parsing -------------------------------
    def _parse(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parse the ffield file into section-specific DataFrames.

        Returns
        -------
        df : pd.DataFrame
            Empty summary DataFrame (no single unified table for ffield).
        meta : dict
            Metadata dictionary; includes at least:
              - "description": first non-empty line (ffield description).
        """
        path = self.path
        lines = path.read_text().splitlines()

        meta: Dict[str, Any] = {}

        # Capture ffield description: first non-empty line
        for ln in lines:
            if ln.strip():
                meta["description"] = ln.strip()
                break

        # reset containers
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
                continue

            # Atom header: e.g. " 19    ! Nr of atoms; cov.r; ..."
            if "atom" in lower:
                n = self._first_int_in_line(line)
                if n is None:
                    i += 1
                    continue
                atom_df, i = self._parse_atom_section(lines, i + 1, n)
                sections[self.SECTION_ATOM] = atom_df
                continue

            # Bonds (but not off-diagonal or hydrogen bonds)
            if "bond" in lower and "off" not in lower and "hydrogen" not in lower:
                n = self._first_int_in_line(line)
                if n is None:
                    i += 1
                    continue
                bond_df, i = self._parse_bond_section(lines, i + 1, n)
                sections[self.SECTION_BOND] = bond_df
                continue

            if "off-diagonal" in lower or "off diagonal" in lower:
                n = self._first_int_in_line(line)
                if n is None:
                    i += 1
                    continue
                off_df, i = self._parse_off_diagonal_section(lines, i + 1, n)
                sections[self.SECTION_OFF_DIAGONAL] = off_df
                continue

            if "angle" in lower:
                n = self._first_int_in_line(line)
                if n is None:
                    i += 1
                    continue
                angle_df, i = self._parse_angle_section(lines, i + 1, n)
                sections[self.SECTION_ANGLE] = angle_df
                continue

            if "torsion" in lower:
                n = self._first_int_in_line(line)
                if n is None:
                    i += 1
                    continue
                torsion_df, i = self._parse_torsion_section(lines, i + 1, n)
                sections[self.SECTION_TORSION] = torsion_df
                continue

            if "hydrogen" in lower and "bond" in lower:
                n = self._first_int_in_line(line)
                if n is None:
                    i += 1
                    continue
                hbond_df, i = self._parse_hbond_section(lines, i + 1, n)
                sections[self.SECTION_HBOND] = hbond_df
                continue

            i += 1

        # Attach to self so properties can see them after parse()
        self._sections = sections

        # Summary DataFrame for ffield is intentionally empty
        df = pd.DataFrame()
        return df, meta

    # ---------------- section parsers ----------------------------
    def _parse_general_section(
        self, lines: List[str], start: int, n_params: int
    ) -> Tuple[pd.DataFrame, int]:
        """
        General section:

        Each line: "<value> ! title".
        - Title is taken from the comment text.
        - If title starts with 'Not used', we store it as
          "Not used: line X" where X is the *index within the general block*
          (1-based, i.e., parameter #).
        """
        records: List[Dict[str, Any]] = []

        for idx in range(n_params):
            if start + idx >= len(lines):
                break
            raw_line = lines[start + idx]

            if "!" in raw_line:
                left, comment = raw_line.split("!", 1)
                title_raw = comment.strip()
            else:
                left = raw_line
                title_raw = f"param_{idx + 1}"

            tokens = left.split()
            value = float(tokens[0]) if tokens else float("nan")

            if title_raw.lower().startswith("not used"):
                title = f"Not used: line {idx + 1}"
            else:
                title = title_raw

            records.append(
                {
                    "index": idx + 1,
                    "name": title,
                    "value": value,
                    "raw_comment": title_raw,
                }
            )

        df = pd.DataFrame.from_records(records).set_index("index")
        end = start + n_params
        return df, end

    def _parse_atom_section(
        self, lines: List[str], start: int, n_atoms: int
    ) -> Tuple[pd.DataFrame, int]:
        """
        Atom section:

        For each atom, we collect 4×8 = 32 numeric parameters in order:
            cov.r; valency; a.m; Rvdw; Evdw; gammaEEM; cov.r2; #el
            alfa; gammavdW; valency; Eunder; n.u.; chiEEM; etaEEM; n.u.
            cov r3; Elp; Heat inc.; 13BO1; 13BO2; 13BO3; n.u.; n.u.
            ov/un; val1; n.u.; val3; vval4; n.u.; n.u.; n.u.

        For titles "n.u." we use numbered names: n.u.1, n.u.2, ...

        We also try to read:
        - atom_index: leading integer on the first line
        - symbol: token after atom_index (if non-numeric)
        """
        names = self._number_unused_titles(self._ATOM_PARAM_NAMES_BASE)
        n_per_atom = len(names)

        records: List[Dict[str, Any]] = []
        n_lines = len(lines)

        # Skip the 4 description lines after the atom header:
        #  ! Nr of atoms; cov.r; ...
        #          alfa; gammavdW; ...
        #          cov r3; Elp; ...
        #          ov/un; val1; ...
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
                    # optional leading integer index
                    try:
                        atom_no = int(tokens[0])
                        j = 1
                    except ValueError:
                        atom_no = atom_idx
                        j = 0

                    # optional symbol
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
        """
        Bond section:

        For each bond we collect:
            Edis1; Edis2; Edis3; pbe1; pbo5; 13corr; pbo6; kov
            pbe2; pbo3; pbo4; n.u.; pbo1; pbo2; ovcorr; n.u.
        with "n.u." renamed to n.u.1, n.u.2, ...

        We also try to read atom-type indices `i` and `j` on the first line.
        """
        names = self._number_unused_titles(self._BOND_PARAM_NAMES_BASE)
        n_per_bond = len(names)

        records: List[Dict[str, Any]] = []
        n_lines = len(lines)

        # Skip the 2 description lines after the bond header:
        #  ! Nr of bonds; ...
        #              pbe2; pbo3; ...
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
        """
        Off-diagonal section:

        For each entry:
            Evdw; Rvdw; alfa; cov.r; cov.r2; cov.r3
        plus atom-type indices `i` and `j` (if present).
        """
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
        """
        Angle section:

        Parameters per angle:
            Theta0; ka; kb; pconj; pv2; kpenal; pv3
        We also try to read atom-type indices i, j, k on the first line.
        """
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
        """
        Torsion section:

        Parameters per torsion:
            V1; V2; V3; V2(BO); vconj; n.u.; n.u.
        with n.u. -> n.u.1, n.u.2, ...

        We also try to read atom-type indices i, j, k, l.
        """
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
        """
        H-bond section:

        Parameters per hydrogen bond:
            Rhb; Dehb; vhb1; vhb2
        plus i, j, k atom-type indices when present.
        """
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
        """Return the first integer found in a line, or None."""
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
        """
        Replace occurrences of `label` in `names` with numbered variants:
        n.u.1, n.u.2, ...
        """
        result: List[str] = []
        counter = 0
        for name in names:
            if name == label:
                counter += 1
                result.append(f"{label}{counter}")
            else:
                result.append(name)
        return result
