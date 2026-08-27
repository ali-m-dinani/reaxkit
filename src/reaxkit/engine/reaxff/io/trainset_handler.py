"""
ReaxFF training set definition (TRAINSET) handler.

This module provides a handler for parsing ReaxFF TRAINSET-style files,
which define reference data, weights, and targets used during
force-field parameter optimization.

TRAINSET files are sectioned and heterogeneous by design, containing
distinct blocks for charges, heats of formation, geometries, cell
parameters, and energies.

**Usage context**

- ReaxFF parsing: Read ReaxFF text outputs into normalized tabular structures.
- Workflow ingestion: Provide canonical handler interfaces used by adapters/workflows.
- Diagnostics/export: Preserve parsed metadata for reporting and downstream conversion.
"""


from __future__ import annotations

import math
from typing import Dict, Any, List, Optional
import pandas as pd

from reaxkit.core.platform.exceptions import ParseError
from reaxkit.engine.reaxff.io.base import BaseHandler


# Map raw section labels in the file to canonical section names
SECTION_MAP = {
    "CHARGE": "CHARGE",
    "CHARGES": "CHARGE",
    "HEATFO": "HEATFO",
    "GEOMETRY": "GEOMETRY",
    "CELL PARAMETERS": "CELL_PARAMETERS",
    "CELL": "CELL_PARAMETERS",   # in case it's written as CELL
    "ENERGY": "ENERGY",
}


def _split_inline_comment(line: str) -> tuple[str, str]:
    """
     split inline comment.

    Parameters
    ----------
    line : str
        Parameter description.

    Returns
    -------
    tuple[str, str]
        Return value description.

    """
    if "!" in line:
        data, comment = line.split("!", 1)
        return data.strip(), comment.strip()
    return line.strip(), ""


def _unpack_line_item(item: str | tuple[int, str]) -> tuple[int, str]:
    """Return (line_number, raw_line_text) for parser input items."""
    if isinstance(item, tuple) and len(item) == 2:
        try:
            return int(item[0]), str(item[1])
        except Exception:
            return -1, str(item[1])
    return -1, str(item)


def _advance_group_comment(
    *,
    current: str,
    text: str,
    last_was_comment: bool,
    current_line_number: object,
    occurrence: int,
    line_number: int,
) -> tuple[str, bool, object, int]:
    """Update one comment block while preserving its source occurrence."""
    if last_was_comment:
        if current and text:
            current = f"{current} /// {text}"
        elif text:
            current = text
        return current, True, current_line_number, occurrence
    return (
        text,
        True,
        line_number if line_number >= 0 else pd.NA,
        occurrence + 1,
    )


def _parse_charge(lines: List[str | tuple[int, str]], section_name: str) -> pd.DataFrame:
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
    group_comment_line_number: object = pd.NA
    group_comment_occurrence = 0
    last_was_comment = False  # track previous processed line

    for item in lines:
        line_number, raw = _unpack_line_item(item)
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
            (
                group_comment,
                last_was_comment,
                group_comment_line_number,
                group_comment_occurrence,
            ) = _advance_group_comment(
                current=group_comment,
                text=text,
                last_was_comment=last_was_comment,
                current_line_number=group_comment_line_number,
                occurrence=group_comment_occurrence,
                line_number=line_number,
            )
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
                "line_number": line_number if line_number >= 0 else pd.NA,
                "group_comment": group_comment,
                "group_comment_line_number": group_comment_line_number,
                "group_comment_occurrence": group_comment_occurrence,
                "iden": iden,
                "weight": weight,
                "atom": atom,
                "lit": lit,
                "inline_comment": inline_comment,
            }
        )

    return pd.DataFrame(rows)


def _parse_heatfo(lines: List[str | tuple[int, str]], section_name: str) -> pd.DataFrame:
    """
    HEATFO block:

        HEATFO
        #Iden Weight Lit
        # group line 1
        # group line 2
        benzene  1.0  -19.82  !heat of formation
        cyclohexane  2.0      !reference may be supplied by fort.99
        ...
        ENDHEATFO

    Columns: section, iden, weight, lit, inline_comment, group_comment

    ``lit`` is optional because some ReaxFF trainsets omit it while fort.99
    still reports the QM/literature target used by the optimization.

    group_comment behavior:
    - Consecutive '#' lines are concatenated with " /// ".
    - All following data lines share that comment until a new '#' block
      appears, which overwrites the previous one.
    """
    rows = []
    group_comment = ""
    group_comment_line_number: object = pd.NA
    group_comment_occurrence = 0
    last_was_comment = False  # track whether previous processed line was a comment

    for item in lines:
        line_number, raw = _unpack_line_item(item)
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
            (
                group_comment,
                last_was_comment,
                group_comment_line_number,
                group_comment_occurrence,
            ) = _advance_group_comment(
                current=group_comment,
                text=text,
                last_was_comment=last_was_comment,
                current_line_number=group_comment_line_number,
                occurrence=group_comment_occurrence,
                line_number=line_number,
            )
            continue

        # data line → next comment block should overwrite, not append
        last_was_comment = False

        data, inline_comment = _split_inline_comment(line)
        tokens = data.split()
        if len(tokens) < 2:
            continue

        iden = tokens[0]
        weight = float(tokens[1])
        lit = float(tokens[2]) if len(tokens) >= 3 else pd.NA

        rows.append(
            {
                "section": section_name,
                "line_number": line_number if line_number >= 0 else pd.NA,
                "group_comment": group_comment,
                "group_comment_line_number": group_comment_line_number,
                "group_comment_occurrence": group_comment_occurrence,
                "iden": iden,
                "weight": weight,
                "lit": lit,
                "inline_comment": inline_comment,
            }
        )

    return pd.DataFrame(rows)


def _parse_geometry(lines: List[str | tuple[int, str]], section_name: str) -> pd.DataFrame:
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
    group_comment_line_number: object = pd.NA
    group_comment_occurrence = 0
    last_was_comment = False  # track whether previous processed line was a comment

    for item in lines:
        line_number, raw = _unpack_line_item(item)
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
            (
                group_comment,
                last_was_comment,
                group_comment_line_number,
                group_comment_occurrence,
            ) = _advance_group_comment(
                current=group_comment,
                text=text,
                last_was_comment=last_was_comment,
                current_line_number=group_comment_line_number,
                occurrence=group_comment_occurrence,
                line_number=line_number,
            )
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
            "line_number": line_number if line_number >= 0 else pd.NA,
            "iden": iden,
            "weight": weight,
            "lit": lit,
            "inline_comment": inline_comment,
            "group_comment": group_comment,
            "group_comment_line_number": group_comment_line_number,
            "group_comment_occurrence": group_comment_occurrence,
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


def _parse_cell_parameters(lines: List[str | tuple[int, str]], section_name: str) -> pd.DataFrame:
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
    group_comment_line_number: object = pd.NA
    group_comment_occurrence = 0
    last_was_comment = False  # track whether previous processed line was a comment

    for item in lines:
        line_number, raw = _unpack_line_item(item)
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
            (
                group_comment,
                last_was_comment,
                group_comment_line_number,
                group_comment_occurrence,
            ) = _advance_group_comment(
                current=group_comment,
                text=text,
                last_was_comment=last_was_comment,
                current_line_number=group_comment_line_number,
                occurrence=group_comment_occurrence,
                line_number=line_number,
            )
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
                "line_number": line_number if line_number >= 0 else pd.NA,
                "group_comment": group_comment,
                "group_comment_line_number": group_comment_line_number,
                "group_comment_occurrence": group_comment_occurrence,
                "iden": iden,
                "weight": weight,
                "type": type_,
                "lit": lit,
                "inline_comment": inline_comment,
            }
        )

    return pd.DataFrame(rows)


def _parse_energy(
    lines: List[str | tuple[int, str]],
    section_name: str,
    *,
    strict: bool = False,
    source_path: str = "<unknown>",
) -> pd.DataFrame:
    """
     parse energy.

    Parameters
    ----------
    lines : List[str]
        Parameter description.
    section_name : str
        Parameter description.

    Returns
    -------
    pd.DataFrame
        Return value description.

    """
    rows: List[Dict[str, Any]] = []
    group_comment = ""
    group_comment_line_number: object = pd.NA
    group_comment_occurrence = 0
    last_was_comment = False  # track if previous processed line was a comment

    for item in lines:
        line_number, raw = _unpack_line_item(item)
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
            (
                group_comment,
                last_was_comment,
                group_comment_line_number,
                group_comment_occurrence,
            ) = _advance_group_comment(
                current=group_comment,
                text=text,
                last_was_comment=last_was_comment,
                current_line_number=group_comment_line_number,
                occurrence=group_comment_occurrence,
                line_number=line_number,
            )
            continue

        # ---- data line ----
        last_was_comment = False  # next comment block should overwrite

        data, inline_comment = _split_inline_comment(line)
        tokens = data.split()

        def reject(reason: str) -> None:
            if strict:
                raise ParseError(
                    f"Invalid ENERGY entry in '{source_path}' at line {line_number}: "
                    f"{reason}."
                )

        if len(tokens) < 4:
            reject("expected weight, at least one operand, and literature value")
            continue

        # first token: weight
        try:
            weight = float(tokens[0])
        except ValueError:
            reject(f"weight '{tokens[0]}' is not numeric")
            continue

        # last token: lit (target energy)
        try:
            lit = float(tokens[-1])
        except ValueError:
            reject(f"literature value '{tokens[-1]}' is not numeric")
            continue

        # middle_part: everything between weight and lit
        middle_part = " ".join(tokens[1:-1]).strip()
        if not middle_part:
            reject("missing ENERGY operand")
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

        normalized_operands: List[str] = []
        operand_index = 1
        i = 0
        valid = True
        while i < len(norm):
            if i + 1 >= len(norm):
                reject(f"operand {operand_index} is missing its identifier")
                valid = False
                break
            normalized_operands.extend((norm[i], norm[i + 1]))
            i += 2
            if i < len(norm) and norm[i].startswith("/"):
                normalized_operands.append(norm[i])
                i += 1
            else:
                normalized_operands.append("/1")
            operand_index += 1
        if not valid:
            continue
        norm = normalized_operands
        valid = True
        for operand_index in range(0, len(norm), 3):
            op, iden, divisor = norm[operand_index : operand_index + 3]
            if op == "\u2013":
                op = "-"
            if op not in {"+", "-"}:
                reject(f"operand {operand_index // 3 + 1} has invalid operator '{norm[operand_index]}'")
                valid = False
                break
            if not iden or any(char in iden for char in "+-/"):
                reject(f"operand {operand_index // 3 + 1} has invalid identifier '{iden}'")
                valid = False
                break
            divisor_value = divisor[1:] if divisor.startswith("/") else ""
            try:
                parsed_divisor = float(divisor_value)
            except ValueError:
                parsed_divisor = 0.0
            if not math.isfinite(parsed_divisor) or parsed_divisor <= 0:
                reject(
                    f"operand {operand_index // 3 + 1} has invalid divisor '{divisor}'; "
                    "expected / followed by a positive number"
                )
                valid = False
                break
        if not valid:
            continue

        row: Dict[str, Any] = {
            "section": section_name,
            "line_number": line_number if line_number >= 0 else pd.NA,
            "group_comment": group_comment,
            "group_comment_line_number": group_comment_line_number,
            "group_comment_occurrence": group_comment_occurrence,
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

    _CACHE_VERSION = "7"
    filetype = "trainset"

    def __init__(
        self,
        file_path: str = "trainset.in",
        reporter=None,
        *,
        strict: bool = False,
    ):
        """Init."""
        super().__init__(file_path)
        self._reporter = reporter
        self.strict = strict

    def _parse(self) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        TemplateHandler expects _parse(self) with NO arguments.
        So we load file content here.
        """
        # read the file
        with open(self.path, "r") as f:
            lines = f.read().splitlines()
        total_lines = len(lines)

        tables: Dict[str, pd.DataFrame] = {}
        current_raw_label: Optional[str] = None
        current_canonical: Optional[str] = None
        buffer: List[tuple[int, str]] = []
        section_occurrences: List[Dict[str, Any]] = []
        occurrence_counts: Dict[str, int] = {}
        current_section_start_line: Optional[int] = None

        def flush_section(end_line_number: Optional[int] = None):
            """Flush section.

            Parameters
            ----------
            None

            Returns
            -------
            Any
                Return value.

            Examples
            --------
            ```python
            # Example
            flush_section(...)
            ```
            """
            nonlocal buffer, current_canonical, tables

            if not current_canonical:
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
                df = _parse_energy(
                    buffer,
                    name,
                    strict=self.strict,
                    source_path=str(self.path),
                )
            else:
                df = pd.DataFrame()

            # 🔧 KEY CHANGE: append rather than overwrite
            if name in tables and not tables[name].empty:
                tables[name] = pd.concat([tables[name], df], ignore_index=True)
            else:
                tables[name] = df

            occurrence = occurrence_counts.get(name, 0) + 1
            occurrence_counts[name] = occurrence
            section_occurrences.append(
                {
                    "section": name,
                    "occurrence": occurrence,
                    "section_order": len(section_occurrences) + 1,
                    "start_line_number": current_section_start_line,
                    "end_line_number": end_line_number,
                    "entry_count": len(df),
                }
            )

            buffer = []

        for line_i, raw in enumerate(lines, start=1):
            if self._reporter and (line_i % 2000 == 0 or line_i == total_lines):
                self._reporter("load", line_i, total_lines, "Parsing trainset")
            stripped = raw.strip()
            if not stripped:
                continue

            upper = stripped.upper()

            # SECTION START?
            if upper in SECTION_MAP:
                flush_section(line_i - 1)
                current_raw_label = stripped
                current_canonical = SECTION_MAP[upper]
                buffer = []
                current_section_start_line = line_i
                continue

            # INSIDE A SECTION
            if current_raw_label and current_canonical:
                end_token = "END" + current_raw_label.replace(" ", "").upper()

                if upper.startswith(end_token):
                    flush_section(line_i)
                    current_raw_label = None
                    current_canonical = None
                    buffer = []
                    current_section_start_line = None
                    continue

                buffer.append((line_i, raw))

        # Final flush
        flush_section(total_lines if current_canonical else None)
        if self._reporter:
            self._reporter("load", total_lines, total_lines, "Finished parsing trainset")

        # RETURN EMPTY summary + metadata
        return pd.DataFrame(), {
            "sections": list(tables.keys()),
            "tables": tables,
            "section_occurrences": section_occurrences,
        }

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def section(self, name: str) -> pd.DataFrame:
        """
        Section.

        Parameters
        ----------
        name : str
            Parameter description.

        Returns
        -------
        pd.DataFrame
            Return value description.

        """
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
        """
        Charges.

        Returns
        -------
        pd.DataFrame
            Return value description.

        """
        return self.section("CHARGE")

    def heatfo(self) -> pd.DataFrame:
        """
        Heatfo.

        Returns
        -------
        pd.DataFrame
            Return value description.

        """
        return self.section("HEATFO")

    def geometry(self) -> pd.DataFrame:
        """
        Geometry.

        Returns
        -------
        pd.DataFrame
            Return value description.

        """
        return self.section("GEOMETRY")

    def cell_parameters(self) -> pd.DataFrame:
        """
        Cell parameters.

        Returns
        -------
        pd.DataFrame
            Return value description.

        """
        return self.section("CELL_PARAMETERS")

    def energy_terms(self) -> pd.DataFrame:
        """
        Energy terms.

        Returns
        -------
        pd.DataFrame
            Return value description.

        """
        return self.section("ENERGY")
