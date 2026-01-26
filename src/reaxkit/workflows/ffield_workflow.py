# reaxkit/workflows/ffield_workflow.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from reaxkit.io.ffield_handler import FFieldHandler
from reaxkit.analysis.ffield_analyzer import get_sections_data

# You added these in ffield_analyzer in the previous step.
# If the names differ in your repo, update these imports accordingly.
from reaxkit.analysis.ffield_analyzer import interpret_one_section, interpret_ffield_terms

from reaxkit.utils.path import resolve_output_path


# ------------------------ helpers ------------------------

def _normalize_section(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")


def _atom_maps(handler: FFieldHandler) -> Tuple[Dict[int, str], Dict[str, int]]:
    atom_df = handler.section_df(FFieldHandler.SECTION_ATOM)
    if "symbol" not in atom_df.columns:
        raise KeyError("Atom section missing 'symbol' column.")

    idx_to_sym: Dict[int, str] = {}
    sym_to_idx: Dict[str, int] = {}
    for idx, row in atom_df.iterrows():
        i = int(idx)
        sym = str(row["symbol"]).strip()
        idx_to_sym[i] = sym
        # assume unique symbol->index in typical ffield; if repeated, last wins
        sym_to_idx[sym] = i
    return idx_to_sym, sym_to_idx


def _split_term_string(term: str) -> List[str]:
    """
    Accepts: "C-H", "C H", "CCH", "1-2-3", "1 2 3"
    Returns list of tokens (symbols or indices as strings).
    """
    t = term.strip()
    if not t:
        return []
    # If it contains separators, split on them
    for sep in ["-", ",", " ", "_"]:
        if sep in t:
            toks = [x for x in t.replace(",", " ").replace("-", " ").replace("_", " ").split() if x]
            return toks

    # No separators: could be "CCH" or "1123" (indices without separators).
    # Heuristic: if all digits => treat as single token (ambiguous) -> user should use separators.
    if t.isdigit():
        return [t]

    # For symbols like CCH, split into element-like tokens (handles 1-2 letter symbols).
    # Example: "SiOH" -> ["Si","O","H"]
    toks: List[str] = []
    i = 0
    while i < len(t):
        ch = t[i]
        if ch.isupper():
            if i + 1 < len(t) and t[i + 1].islower():
                toks.append(t[i : i + 2])
                i += 2
            else:
                toks.append(ch)
                i += 1
        else:
            # unexpected; fall back
            toks.append(ch)
            i += 1
    return toks


def _term_cols_for_section(section: str) -> List[str]:
    if section in (FFieldHandler.SECTION_BOND, FFieldHandler.SECTION_OFF_DIAGONAL):
        return ["i", "j"]
    if section in (FFieldHandler.SECTION_ANGLE, FFieldHandler.SECTION_HBOND):
        return ["i", "j", "k"]
    if section == FFieldHandler.SECTION_TORSION:
        return ["i", "j", "k", "l"]
    raise KeyError(f"Unsupported term filtering for section: {section!r}")


def _make_term_series_indices(df: pd.DataFrame, cols: Sequence[str], *, unordered_2body: bool) -> pd.Series:
    """
    Build a comparable "term key" series from numeric columns.
    - For 2-body (bond/offdiag): optionally unordered, so (1,2)==(2,1).
    - For 3/4-body: keep order (CCH != CHC).
    """
    if len(cols) == 2 and unordered_2body:
        a = df[cols[0]].astype("Int64")
        b = df[cols[1]].astype("Int64")
        lo = a.where(a <= b, b)
        hi = b.where(a <= b, a)
        return lo.astype(str) + "-" + hi.astype(str)

    # ordered
    parts = [df[c].astype("Int64").astype(str) for c in cols]
    s = parts[0]
    for p in parts[1:]:
        s = s + "-" + p
    return s


def _filter_df_by_term(
    handler: FFieldHandler,
    section: str,
    df: pd.DataFrame,
    *,
    term: str,
    out_format: str,  # kept for API compatibility (not needed for filtering)
    unordered_2body: bool = True,
    any_order: bool = False,
) -> pd.DataFrame:
    """
    Filter df by a user term, supporting both interpreted and indices requests.

    any_order:
      - If True (recommended for angle/torsion/hbond): match all permutations of the atoms.
        Example: angle --term CCH --any-order matches CCH, CHC, HCC.
      - For 2-body sections, any_order behaves like unordered matching (same as unordered_2body=True).

    Examples:
      bond  --term C-H
      bond  --term 1-2
      angle --term CCH
      angle --term C-C-H
      angle --term 1-1-2
      angle --term CCH --any-order
    """
    cols = _term_cols_for_section(section)
    toks = _split_term_string(term)
    if not toks:
        return df

    # helper
    def _tokens_are_all_int(xs: Sequence[str]) -> bool:
        return all(x.isdigit() for x in xs)

    # --- parse "wanted" into integer indices ---
    if _tokens_are_all_int(toks):
        # indices provided directly
        idxs = [int(x) for x in toks]
    else:
        # symbols -> indices
        _, sym_to_idx = _atom_maps(handler)
        idxs: List[int] = []
        for s in toks:
            if s not in sym_to_idx:
                raise KeyError(f"Unknown atom symbol {s!r}. Available: {sorted(sym_to_idx.keys())}")
            idxs.append(int(sym_to_idx[s]))

    if len(idxs) != len(cols):
        raise ValueError(
            f"Term {term!r} implies {len(idxs)} atoms, but section '{section}' needs {len(cols)}."
        )

    # --- matching logic ---
    is_2body = len(cols) == 2

    # any_order for 2-body is equivalent to unordered matching
    if is_2body and (unordered_2body or any_order):
        a, b = sorted(idxs)
        wanted = f"{a}-{b}"
        key = _make_term_series_indices(df, cols, unordered_2body=True)
        return df.loc[key == wanted].copy()

    # any_order for 3/4-body: match permutations via sorted multiset equality
    if any_order and not is_2body:
        wanted_sorted = sorted(idxs)

        # Vectorized-ish approach: take the relevant columns, cast to int,
        # sort per-row, and compare to wanted_sorted.
        sub = df.loc[:, cols].astype("Int64")

        def _row_sorted_equals_wanted(r: pd.Series) -> bool:
            vals = [int(x) for x in r.tolist()]
            vals.sort()
            return vals == wanted_sorted

        mask = sub.apply(_row_sorted_equals_wanted, axis=1)
        return df.loc[mask].copy()

    # default ordered matching (CCH != CHC)
    wanted = "-".join(str(x) for x in idxs)
    key = _make_term_series_indices(df, cols, unordered_2body=False)
    return df.loc[key == wanted].copy()


    # indices
    df = get_sections_data(handler, section=section)
    cols = _term_cols_for_section(section)
    key = _make_term_series_indices(df, cols, unordered_2body=unordered_2body)
    return sorted(set(key.dropna().unique().tolist()))


def _export_df(df: pd.DataFrame, export_path: str | Path) -> None:
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(export_path, index=True)
    print(f"[Done] Saved the requested data in {export_path}")


# ------------------------ tasks ------------------------


def task_get(args: argparse.Namespace) -> int:
    handler = FFieldHandler(args.file)
    section = _normalize_section(args.section)

    interpreted = args.format == "interpreted"

    # get df (interpreted or base)
    if interpreted:
        df = interpret_one_section(handler, section=section)
    else:
        df = get_sections_data(handler, section=section)

    # optional filter
    if args.term:
        df = _filter_df_by_term(
            handler,
            section,
            df if not interpreted else get_sections_data(handler, section=section),
            term=args.term,
            out_format=args.format,
            unordered_2body=not args.ordered_2body,
            any_order=args.any_order,
        )
        # if interpreted requested, re-interpret the filtered subset for display/export
        if interpreted and not df.empty:
            df = interpret_one_section(handler, section=section).loc[df.index].copy()

    # export or preview
    if args.export:
        out_path = resolve_output_path(args.export, workflow="ffield")
        _export_df(df, out_path)
    else:
        with pd.option_context("display.max_rows", min(30, len(df)), "display.max_columns", 200):
            print(df.head(30))
            if len(df) > 30:
                print(f"... ({len(df)} rows total)")
    return 0


def task_export(args: argparse.Namespace) -> int:
    handler = FFieldHandler(args.file)
    interpreted = args.format == "interpreted"

    # where to put files
    out_dir = resolve_output_path(args.outdir, workflow="ffield")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = [
        FFieldHandler.SECTION_GENERAL,
        FFieldHandler.SECTION_ATOM,
        FFieldHandler.SECTION_BOND,
        FFieldHandler.SECTION_OFF_DIAGONAL,
        FFieldHandler.SECTION_ANGLE,
        FFieldHandler.SECTION_TORSION,
        FFieldHandler.SECTION_HBOND,
    ]

    if interpreted:
        # interpret only the term-bearing sections; keep atom/general as-is
        interpreted_map = interpret_ffield_terms(handler)
    else:
        interpreted_map = {}

    print('[Done] Exporting all sections data:')
    for sec in sections:
        if interpreted and sec in interpreted_map:
            df = interpreted_map[sec]
        else:
            df = handler.section_df(sec).copy()

        suffix = "interpreted" if interpreted else "indices"
        out_path = out_dir / f"{sec}_{suffix}.csv"
        df.to_csv(out_path, index=True)
        print(f"  {sec:12s} -> {out_path}")

    return 0


# ------------------------ CLI registration ------------------------
def _add_common_ffield_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--file", default="ffield", help="Path to ffield file.")
    p.add_argument(
        "--format",
        default="interpreted",
        choices=["interpreted", "indices"],
        help="Output format: interpreted uses atom symbols (C-H), indices uses numeric (1-2).",
    )
    p.add_argument(
        "--export",
        default=None,
        help="Path to export CSV. If omitted, prints a preview.",
    )

def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    CLI:
      reaxkit ffield get ...
      reaxkit ffield export ...
    """
    # ---- get ----
    p = subparsers.add_parser(
        "get",
        help="Get a single ffield section (optionally interpreted + filtered).",
        description=(
            "Examples:\n"
            "  # 1) Get all C-H bond rows (interpreted output)\n"
            "  reaxkit ffield get --section bond --term C-H --format interpreted --export CH_bond.csv\n"
            "\n"
            "  # 2) Same, but using indices\n"
            "  reaxkit ffield get --section bond --term 1-2 --format indices --export 1_2_bond.csv\n"
            "\n"
            "  # 3) Angles: get only C-C-H (ordered)\n"
            "  reaxkit ffield get --section angle --term CCH --format interpreted --export CCH_angles.csv\n"
            "\n"
            "  # 4) Angles: get all combinations of C-C-H in angle data\n"
            "  reaxkit ffield get --section angle --term CCH --format interpreted --any-order --export all_CCH_angles.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_ffield_args(p)
    p.add_argument("--section", required=True,
                   help="Section: general, atom, bond, off_diagonal, angle, torsion, hbond.")
    p.add_argument("--term", default=None,
                   help="Filter term, e.g. 'C-H', 'CCH', 'C-C-H', '1-2', '1-1-2'.")
    p.add_argument("--ordered-2body", action="store_true",
                   help="For 2-body sections (bond/off_diagonal), treat (C-H) and (H-C) as different. Default is unordered.")
    p.add_argument(
        "--any-order",
        action="store_true",
        help="Match all permutations of the given term (e.g. CCH matches CCH, CHC, HCC).",
    )
    p.set_defaults(_run=task_get)

    # ---- export ----
    p = subparsers.add_parser(
        "export",
        help="Export all ffield sections as separate CSV files.",
        description=(
            "Examples:\n"
            "  # Export everything interpreted (C-H, C-C-H, ...)\n"
            "  reaxkit ffield export --format interpreted --outdir reaxkit_outputs/ffield/all_ffield_csv\n"
            "\n"
            "  # Export everything in indices (1-2, 1-1-2, ...)\n"
            "  reaxkit ffield export --format indices --outdir reaxkit_outputs/ffield/all_ffield_csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--file", default="ffield", help="Path to ffield file.")
    p.add_argument(
        "--format",
        default="interpreted",
        choices=["interpreted", "indices"],
        help="Export format: interpreted uses atom symbols, indices uses numeric atom indices.",
    )
    p.add_argument("--outdir", default="ffield_export",
                   help="Directory to write CSVs (will be placed under reaxkit_output/...).")
    p.set_defaults(_run=task_export)
