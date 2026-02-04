"""
ReaxFF training set definition (TRAINSET) handler.

This module provides a handler for parsing ReaxFF TRAINSET-style files,
which define reference data, weights, and targets used during
force-field parameter optimization.

TRAINSET files are sectioned and heterogeneous by design, containing
distinct blocks for charges, heats of formation, geometries, cell
parameters, and energies.
"""


from __future__ import annotations

from typing import Dict, Any, List, Optional
import pandas as pd

from reaxkit.io.base_handler import BaseHandler


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
        # group line 1
        # group line 2
        AlNH2q  0.10  1   0.83215 !charge for Al atom in AlNH2
        ...
        ENDCHARGE

    Columns: section, iden, weight, atom, lit,
             inline_comment, group_comment

    group_comment behavior:
    - Consecutive '#' lines are concatenated with " /// ".
    - All following data lines share that block.
    - When a new '#' block appears after data, it overwrites.
    """
    rows = []
    group_comment = ""
    last_was_comment = False  # track previous processed line

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # comment lines → update group_comment
        if line.startswith("#"):
            text = line.lstrip("#").strip()

            # skip header-like lines (Weigh / Weight ...)
            if "weigh" in text.lower():
                # header shouldn't join with group comments
                last_was_comment = False
                continue

            # If previous line was comment → append
            # If previous line was data/start → new block
            if last_was_comment and group_comment:
                group_comment += " /// " + text
            else:
                group_comment = text

            last_was_comment = True
            continue

        # data line → the next comment block should replace, not append
        last_was_comment = False

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
                "group_comment": group_comment,
                "iden": iden,
                "weight": weight,
                "atom": atom,
                "lit": lit,
                "inline_comment": inline_comment,
            }
        )

    return pd.DataFrame(rows)


def _parse_heatfo(lines: List[str], section_name: str) -> pd.DataFrame:
    """
    HEATFO block:

        HEATFO
        #Iden Weight Lit
        # group line 1
        # group line 2
        benzene  1.0  -19.82  !heat of formation
        ...
        ENDHEATFO

    Columns: section, iden, weight, lit, inline_comment, group_comment

    group_comment behavior:
    - Consecutive '#' lines are concatenated with " /// ".
    - All following data lines share that comment until a new '#' block
      appears, which overwrites the previous one.
    """
    rows = []
    group_comment = ""
    last_was_comment = False  # track whether previous processed line was a comment

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # comment lines → update group_comment (possibly multi-line)
        if line.startswith("#"):
            text = line.lstrip("#").strip()

            # skip header-like lines (Weigh / Weight ...)
            if "weigh" in text.lower():
                # header shouldn't join with group comments
                last_was_comment = False
                continue

            # Same block → append; new block → overwrite
            if last_was_comment and group_comment:
                group_comment += " /// " + text
            else:
                group_comment = text

            last_was_comment = True
            continue

        # data line → next comment block should overwrite, not append
        last_was_comment = False

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
                "group_comment": group_comment,
                "iden": iden,
                "weight": weight,
                "lit": lit,
                "inline_comment": inline_comment,
            }
        )

    return pd.DataFrame(rows)


def _parse_geometry(lines: List[str], section_name: str) -> pd.DataFrame:
    """
    GEOMETRY block:

        GEOMETRY
        #Iden   Weight At1 At2 At3 At4 Lit
        # group line 1
        # group line 2
        chexane  0.01   1  2           1.54     !bond
        chexane  1.00   1  2  3        111.0    !valence angle
        chexane  1.00   1  2  3  4     56.0     !torsion angle
        chexane  1.00                  0.01     !RMSG

    Required data per row:
        - iden, weight, lit
    Optional:
        - at1, at2, at3, at4 (if present)

    group_comment behavior:
    - Multiple '#' lines in a row are concatenated with " /// ".
    - All following data lines share that group_comment until a new
      '#' block appears, which overwrites the previous one.
    """
    rows = []
    group_comment = ""
    last_was_comment = False  # track whether previous processed line was a comment

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # comment lines → update group_comment (possibly multi-line)
        if line.startswith("#"):
            text = line.lstrip("#").strip()

            # skip header-like lines (Iden / Weight / Weigh ...)
            if "weigh" in text.lower() or "iden" in text.lower():
                last_was_comment = False
                continue

            # Same block → append; new block → overwrite
            if last_was_comment and group_comment:
                group_comment += " /// " + text
            else:
                group_comment = text

            last_was_comment = True
            continue

        # data line → next comment block should overwrite, not append
        last_was_comment = False

        data, inline_comment = _split_inline_comment(line)
        tokens = data.split()

        # Need at least: iden, weight, lit
        if len(tokens) < 3:
            continue

        iden = tokens[0]
        weight = float(tokens[1])
        lit = float(tokens[-1])

        # Middle tokens (between weight and lit) are optional atom indices
        atom_tokens = tokens[2:-1]

        row = {
            "section": section_name,
            "iden": iden,
            "weight": weight,
            "lit": lit,
            "inline_comment": inline_comment,
            "group_comment": group_comment,
        }

        # Fill at1–at4 only if present
        for i, tok in enumerate(atom_tokens[:4], start=1):
            try:
                row[f"at{i}"] = int(tok)
            except ValueError:
                # If something weird appears where an int is expected, skip it
                continue

        rows.append(row)

    # Build DataFrame and order columns nicely
    df = pd.DataFrame(rows)
    if not df.empty:
        base_cols = ["section", "iden", "weight"]
        atom_cols = [c for c in ["at1", "at2", "at3", "at4"] if c in df.columns]
        end_cols = [c for c in ["lit", "inline_comment", "group_comment"] if c in df.columns]
        other_cols = [c for c in df.columns if c not in (base_cols + atom_cols + end_cols)]
        df = df[base_cols + atom_cols + other_cols + end_cols]

    return df


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
    last_was_comment = False  # track whether previous processed line was a comment

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # comment lines update group_comment (possibly multi-line)
        if line.startswith("#"):
            text = line.lstrip("#").strip()

            # skip header-like lines (Weigh / Weight ...)
            if "weigh" in text.lower():
                # header shouldn't join with group comments
                last_was_comment = False
                continue

            # If previous line was also a comment, append (same block)
            # If previous line was data or start of section, start a new block
            if last_was_comment and group_comment:
                group_comment += " /// " + text
            else:
                group_comment = text

            last_was_comment = True
            continue

        # data line → next comments should be treated as a new block
        last_was_comment = False

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
                "group_comment": group_comment,
                "iden": iden,
                "weight": weight,
                "type": type_,
                "lit": lit,
                "inline_comment": inline_comment,
            }
        )

    return pd.DataFrame(rows)


def _parse_energy(lines: List[str], section_name: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_comment = ""
    last_was_comment = False  # track if previous processed line was a comment

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # comment lines (header or group)
        if line.startswith("#"):
            text = line.lstrip("#").strip()

            # skip header-like lines (Weigh / Weight ...)
            if "weigh" in text.lower():
                # header shouldn't join with group comments
                last_was_comment = False
                continue

            # If previous line was also a comment → same block, append
            # If previous line was data or start → new block, overwrite
            if last_was_comment and group_comment:
                group_comment += " /// " + text
            else:
                group_comment = text

            last_was_comment = True
            continue

        # ---- data line ----
        last_was_comment = False  # next comment block should overwrite

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

        # middle_part: everything between weight and lit
        middle_part = " ".join(tokens[1:-1]).strip()
        if not middle_part:
            continue

        middle_tokens = middle_part.split()

        # --- normalize middle tokens ---
        norm: List[str] = []
        for tok in middle_tokens:
            if "/" in tok and tok != "/":
                if tok.startswith("/"):
                    norm.append(tok)
                else:
                    base, rest = tok.split("/", 1)
                    norm.append(base)
                    norm.append("/" + rest)
            else:
                norm.append(tok)

        row: Dict[str, Any] = {
            "section": section_name,
            "group_comment": group_comment,
            "weight": weight,
        }

        i = 0
        group_idx = 1
        while i + 2 < len(norm):
            op = norm[i]
            if op == "–":  # normalize en dash just in case
                op = "-"

            iden = norm[i + 1]
            n_tok = norm[i + 2]

            n = 1.0
            if "/" in n_tok:
                _, n_str = n_tok.split("/", 1)
                try:
                    n = float(n_str.strip())
                except ValueError:
                    n = 1.0

            row[f"op{group_idx}"] = op
            row[f"id{group_idx}"] = iden
            row[f"n{group_idx}"] = n

            group_idx += 1
            i += 3

        row["lit"] = lit
        row["inline_comment"] = inline_comment

        rows.append(row)

    df = pd.DataFrame(rows)
    # --- Reorder columns: dynamic terms first, then lit, then inline_comment ---
    cols = list(df.columns)

    # Fixed columns to move to the end
    end_cols = ["lit", "inline_comment"]

    # Keep only those that exist (in case some are missing)
    end_cols = [c for c in end_cols if c in cols]

    # All other columns come first
    start_cols = [c for c in cols if c not in end_cols]

    # New column order
    df = df[start_cols + end_cols]

    return df


class TrainsetHandler(BaseHandler):
    """
    Parser for ReaxFF training set definition files (TRAINSET).

    This class parses TRAINSET files and exposes their contents as
    section-specific tables, one per training target category.

    Parsed Data
    -----------
    Summary table
        The main ``dataframe()`` is intentionally empty.
        TRAINSET files do not have a single global tabular representation.

    Section tables
        Returned via ``metadata()["tables"]`` or convenience accessors,
        with one table per section:

        - ``CHARGE``:
          Charge fitting targets, with columns:
          ["section", "iden", "weight", "atom", "lit",
           "inline_comment", "group_comment"]

        - ``HEATFO``:
          Heats of formation targets, with columns:
          ["section", "iden", "weight", "lit",
           "inline_comment", "group_comment"]

        - ``GEOMETRY``:
          Geometry-related targets (bond, angle, torsion, RMSG), with columns:
          ["section", "iden", "weight", "at1", "at2", "at3", "at4",
           "lit", "inline_comment", "group_comment"]
          (atom index columns are optional depending on the entry type)

        - ``CELL_PARAMETERS``:
          Cell and lattice targets, with columns:
          ["section", "iden", "weight", "type", "lit",
           "inline_comment", "group_comment"]

        - ``ENERGY``:
          Composite energy expressions, with dynamically generated columns:
          ["section", "weight",
           "op1", "id1", "n1",
           "op2", "id2", "n2", ...,
           "lit", "inline_comment"]

    Metadata
        Returned by ``metadata()``, containing:
        {
            "sections": list[str],        # present section names
            "tables": dict[str, DataFrame]  # section → parsed table
        }

    Notes
    -----
    - Consecutive ``#`` comment lines are grouped and stored as
      ``group_comment`` using ``" /// "`` as a separator.
    - Inline comments following ``!`` are preserved verbatim.
    - Sections appearing multiple times are concatenated automatically.
    - This handler is not frame-based; ``n_frames()`` always returns 0.
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

            # 🔧 KEY CHANGE: append rather than overwrite
            if name in tables and not tables[name].empty:
                tables[name] = pd.concat([tables[name], df], ignore_index=True)
            else:
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
