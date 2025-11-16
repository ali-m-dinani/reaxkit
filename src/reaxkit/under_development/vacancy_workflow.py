# reaxkit/workflows/vacancy_workflow.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

# Import your analyzers (not handlers!)
# Adjust these import paths/names to match your actual analyzer modules.
from reaxkit.analysis.vacancy_analysis import VacancyAnalyzer
from reaxkit.analysis.xmolout_analysis import XmoloutAnalyzer  # ← expects .atom_frames() and .summary()
from reaxkit.analysis.fort7_analyzer import Fort7Analyzer       # ← expects .atom_frames()


def _build_analyzer_from_analyzers(xmol_path: str, fort7_path: str) -> VacancyAnalyzer:
    """
    Adapter layer: construct the VacancyAnalyzer from existing analyzers.
    If your analyzers use different method names, adjust here.
    """
    x_an = XmoloutAnalyzer(xmol_path)   # must expose atom_frames(), summary()
    f7_an = Fort7Analyzer(fort7_path)   # must expose atom_frames()

    return VacancyAnalyzer.from_analyzers(x_an, f7_an)


def vacancy_task(args: argparse.Namespace) -> int:
    """
    Run a vacancy analysis according to the selected mode and write xmolout subset if requested.
    Modes:
      - mode 'bo-deficit': uses valency_map & threshold on (valency - sum_BOs)
      - mode 'al-n-neigh': flags Al with fewer than N neighbors of type 'N'
      - mode 'compose-any': OR-composition of the above two
      - mode 'compose-all': AND-composition of the above two
    """
    analyzer = _build_analyzer_from_analyzers(args.xmolout, args.fort7)

    # Prepare criteria
    valency_map: Dict[str, float] = {"Al": 3.0, "N": 3.0}
    undercoord = analyzer.criterion_undercoordination(valency_map, threshold=args.threshold)
    al_n = analyzer.criterion_al_underconnected(min_n_neighbors_of_N=args.min_n)

    if args.mode == "bo-deficit":
        crit = undercoord
        need_neighbors = False
    elif args.mode == "al-n-neigh":
        crit = al_n
        need_neighbors = True
    elif args.mode == "compose-any":
        crit = analyzer.any_of(undercoord, al_n)
        need_neighbors = True
    elif args.mode == "compose-all":
        crit = analyzer.all_of(undercoord, al_n)
        need_neighbors = True
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Evaluate
    results, filtered_frames = analyzer.evaluate(crit, require_neighbor_types=need_neighbors)

    # Optional CSV dump of indices
    if args.dump_indices:
        out_csv = Path(args.dump_indices)
        rows = []
        for r in results:
            for idx in r.indices:
                rows.append({"frame_index": r.frame_index, "iteration": r.iteration, "row_index": idx})
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Optional xmolout write
    if args.out_xmol:
        out_path = analyzer.write_xmolout_subset(
            filtered_frames,
            args.out_xmol,
            label=args.label,
            skip_empty_frames=True,
        )
        print(f"[Done] Wrote filtered xmolout: {out_path}")

    # Summary print (minimal, not noisy)
    total_flagged = sum(len(r.indices) for r in results)
    print(f"Frames: {len(results)} | Flagged atoms total: {total_flagged}")
    return 0


def register_tasks(subparsers: argparse._SubP
