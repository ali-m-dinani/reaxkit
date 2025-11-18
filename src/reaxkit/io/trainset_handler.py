"""Handler for ReaxFF trainset / fort.99-style files."""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import pandas as pd

from reaxkit.io.template_handler import TemplateHandler


# Map raw section labels in the file to canonical section names
SECTION_MAP = {
    "CHARGE": "CHARGE",
    "HEATFO": "HEATFO",
    "GEOMETRY": "GEOMETRY",
    "CELL PARAMETERS": "CELL_PARAMETERS",
    "CELL": "CELL_PARAMETERS",   # in case it's written as CELL
    "ENERGY": "ENERGY",
}


def _split_inline_comment(line: str) -> tuple[str, str]:
    """Return (data_part, inline_comment) split on '!' (if present)."""
    if "!" in line:
        data, comment = line.split("!", 1)
        return data.strip(), comment.strip()
    return line.strip(), ""


def _parse_charge(lines: List[str], section_name: str) -> pd.DataFrame:
    """
    CHARGE block:

        CHARGE
        #Iden Weight Atom  Lit
        AlNH2q  0.10  1   0.83215 !charge for Al atom in AlNH2
        ...
        ENDCHARGE

    Columns: section, iden, weight, atom, lit, inline_comment, group_comment
    """
    rows = []
    group_comment = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # comment lines update group_comment
        if line.startswith("#"):
            group_comment = line.lstrip("#").strip()
            continue

        data, inline_comment = _split_inline_comment(line)
        tokens = data.split()
        if len(tokens) < 4:
            continue

        iden = tokens[0]
        weight = float(tokens[1])
        atom = int(tokens[2])
        lit = float(tokens[3])

        rows.append(
            {
                "section": section_name,
                "iden": iden,
                "weight": weight,
                "atom": atom,
                "lit": lit,
                "inline_comment": inline_comment,
                "group_comment": group_comment,
            }
        )

    return pd.DataFrame(rows)


def _parse_heatfo(lines: List[str], section_name: str) -> pd.DataFrame:
    """
    HEATFO block:

        HEATFO
        #Iden Weight Lit
        benzene  1.0  -19.82  !heat of formation
        ...
        ENDHEATFO

    Columns: section, iden, weight, lit, inline_comment, group_comment
    """
    rows = []
    group_comment = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("#"):
            group_comment = line.lstrip("#").strip()
            continue

        data, inline_comment = _split_inline_comment(line)
        tokens = data.split()
        if len(tokens) < 3:
            continue

        iden = tokens[0]
        weight = float(tokens[1])
        lit = float(tokens[2])

        rows.append(
            {
                "section": section_name,
                "iden": iden,
                "weight": weight,
                "lit": lit,
                "inline_comment": inline_comment,
                "group_comment": group_comment,
            }
        )

    return pd.DataFrame(rows)


def _parse_geometry(lines: List[str], section_name: str) -> pd.DataFrame:
    """
    GEOMETRY block:

        GEOMETRY
        #Iden Weight At1 At2 At3 At4 Lit
        chexane  0.01   1  2  3  4   1.54  !bond
        ...
        ENDGEOMETRY

    Columns: section, iden, weight, at1, at2, at3, at4, lit,
             inline_comment, group_comment
    """
    rows = []
    group_comment = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("#"):
            group_comment = line.lstrip("#").strip()
            continue

        data, inline_comment = _split_inline_comment(line)
        tokens = data.split()
        if len(tokens) < 7:
            continue

        iden = tokens[0]
        weight = float(tokens[1])
        at1 = int(tokens[2])
        at2 = int(tokens[3])
        at3 = int(tokens[4])
        at4 = int(tokens[5])
        lit = float(tokens[6])

        rows.append(
            {
                "section": section_name,
                "iden": iden,
                "weight": weight,
                "at1": at1,
                "at2": at2,
                "at3": at3,
                "at4": at4,
                "lit": lit,
                "inline_comment": inline_comment,
                "group_comment": group_comment,
            }
        )

    return pd.DataFrame(rows)


def _parse_cell_parameters(lines: List[str], section_name: str) -> pd.DataFrame:
    """
    CELL PARAMETERS block:

        CELL PARAMETERS
        #Iden Weight Type Lit
        mycell  1.0  1  0.0   !some description
        ...
        ENDCELLPARAMETERS (or similar)

    Columns: section, iden, weight, type, lit,
             inline_comment, group_comment
    """
    rows = []
    group_comment = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("#"):
            group_comment = line.lstrip("#").strip()
            continue

        data, inline_comment = _split_inline_comment(line)
        tokens = data.split()
        if len(tokens) < 4:
            continue

        iden = tokens[0]
        weight = float(tokens[1])
        type_ = tokens[2]   # keep as string
        lit = float(tokens[3])

        rows.append(
            {
                "section": section_name,
                "iden": iden,
                "weight": weight,
                "type": type_,
                "lit": lit,
                "inline_comment": inline_comment,
                "group_comment": group_comment,
            }
        )

    return pd.DataFrame(rows)


def _parse_energy(lines: List[str], section_name: str) -> pd.DataFrame:
    """
    ENERGY block:

        ENERGY
        #Weigh op1 Ide1  n1 op2 Ide2    n2    Lit
        #group comment...
         1.5   +  butbenz/1 -  butbenz_a/1   -90.00 !inline
         1   +   Zn_Pt111_top /1 - Pt111_slab /1 - Zn_atom/1   -40.11
         ...
        ENDENERGY

    Generalized parsing:

    - First token  -> weight
    - Last token   -> lit
    - Middle part  -> repeated groups of (op, iden, /n)
      We normalize tokens so that 'Zn_atom/1' becomes ['Zn_atom', '/1'].

    Columns: section, weight, lit,
             op1, id1, n1, op2, id2, n2, op3, id3, n3, ...
             inline_comment, group_comment
    """
    rows: List[Dict[str, Any]] = []
    group_comment = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # comment lines (header or group)
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            # skip header-like lines (Weigh / Weight ...)
            if text.lower().startswith(("weigh", "weight")):
                continue
            group_comment = text
            continue

        data, inline_comment = _split_inline_comment(line)
        tokens = data.split()
        if len(tokens) < 3:
            # need at least weight, something, lit
            continue

        # first token: weight
        try:
            weight = float(tokens[0])
        except ValueError:
            # not a valid energy line
            continue

        # last token: lit (target energy)
        try:
            lit = float(tokens[-1])
        except ValueError:
            continue

        middle_tokens = tokens[1:-1]
        if not middle_tokens:
            continue

        # --- normalize middle tokens ---
        # Ensure that each "iden/n" becomes two tokens: "iden", "/n"
        norm: List[str] = []
        for tok in middle_tokens:
            if "/" in tok and tok != "/":
                if tok.startswith("/"):
                    # already just "/n"
                    norm.append(tok)
                else:
                    # split "Zn_atom/1" -> "Zn_atom", "/1"
                    base, rest = tok.split("/", 1)
                    norm.append(base)
                    norm.append("/" + rest)
            else:
                norm.append(tok)

        # At this point we expect triples: [op, iden, "/n", op, iden, "/n", ...]
        row: Dict[str, Any] = {
            "section": section_name,
            "weight": weight,
            "lit": lit,
            "inline_comment": inline_comment,
            "group_comment": group_comment,
        }

        i = 0
        group_idx = 1
        while i + 2 < len(norm):
            op = norm[i]
            if op == "–":  # normalize en dash just in case
                op = "-"

            iden = norm[i + 1]
            n_tok = norm[i + 2]

            # parse n from "/n"
            n = 1
            if "/" in n_tok:
                # "/1" or possibly "iden/1" if something odd slipped through
                if n_tok.startswith("/"):
                    _, n_str = n_tok.split("/", 1)
                else:
                    _, n_str = n_tok.split("/", 1)
                try:
                    n = int(n_str.strip())
                except ValueError:
                    n = 1

            row[f"op{group_idx}"] = op
            row[f"id{group_idx}"] = iden
            row[f"n{group_idx}"] = n

            group_idx += 1
            i += 3

        rows.append(row)

    return pd.DataFrame(rows)


class TrainsetHandler(TemplateHandler):
    """
    Handler for ReaxFF TRAINSET / fort.99-style files.

    - No single "main" DataFrame: all real data is in metadata["tables"].
    - _parse returns:
        (empty_df,
         {
           "sections": [...],
           "tables": {
              "CHARGE": <df>,
              "HEATFO": <df>,
              ...
           }
         })
    """

    filetype = "trainset"

    def _parse(self) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        TemplateHandler expects _parse(self) with NO arguments.
        So we load file content here.
        """
        # read the file
        with open(self.path, "r") as f:
            lines = f.read().splitlines()

        tables: Dict[str, pd.DataFrame] = {}
        current_raw_label: Optional[str] = None
        current_canonical: Optional[str] = None
        buffer: List[str] = []

        def flush_section():
            nonlocal buffer, current_canonical, tables

            if not current_canonical or not buffer:
                buffer = []
                return

            name = current_canonical

            if name == "CHARGE":
                df = _parse_charge(buffer, name)
            elif name == "HEATFO":
                df = _parse_heatfo(buffer, name)
            elif name == "GEOMETRY":
                df = _parse_geometry(buffer, name)
            elif name == "CELL_PARAMETERS":
                df = _parse_cell_parameters(buffer, name)
            elif name == "ENERGY":
                df = _parse_energy(buffer, name)
            else:
                df = pd.DataFrame()

            tables[name] = df
            buffer = []

        for raw in lines:
            stripped = raw.strip()
            if not stripped:
                continue

            upper = stripped.upper()

            # SECTION START?
            if upper in SECTION_MAP:
                flush_section()
                current_raw_label = stripped
                current_canonical = SECTION_MAP[upper]
                buffer = []
                continue

            # INSIDE A SECTION
            if current_raw_label and current_canonical:
                end_token = "END" + current_raw_label.replace(" ", "").upper()

                if upper.startswith(end_token):
                    flush_section()
                    current_raw_label = None
                    current_canonical = None
                    buffer = []
                    continue

                buffer.append(raw)

        # Final flush
        flush_section()

        # RETURN EMPTY summary + metadata
        return pd.DataFrame(), {
            "sections": list(tables.keys()),
            "tables": tables,
        }

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def section(self, name: str) -> pd.DataFrame:
        """Return table for a given section (case-insensitive)."""
        meta = self.metadata()
        tables = meta.get("tables", {})
        key = name.upper()
        # normalize CELL vs CELL_PARAMETERS
        if key in ("CELL", "CELL PARAMETERS"):
            key = "CELL_PARAMETERS"
        if key not in tables:
            raise KeyError(f"Section '{name}' not found in trainset.")
        return tables[key]

    def charges(self) -> pd.DataFrame:
        return self.section("CHARGE")

    def heatfo(self) -> pd.DataFrame:
        return self.section("HEATFO")

    def geometry(self) -> pd.DataFrame:
        return self.section("GEOMETRY")

    def cell_parameters(self) -> pd.DataFrame:
        return self.section("CELL_PARAMETERS")

    def energy_terms(self) -> pd.DataFrame:
        return self.section("ENERGY")
