"""Build report payload dictionaries for active-site analysis commands.

This module converts active-site analyzer results into structured report
payloads consumed by presentation/report rendering layers. It is scoped to
payload assembly and interpretation text generation, not analysis execution.

**Usage context**

- Structural reporting: Build sectioned report payloads for structural outputs.
- Event reporting: Summarize reactive-event outputs into report-friendly schema.
- Registry wiring: Register active-site payload builders with report registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _as_int(value: Any, default: int = 0) -> int:
    """
    As int.
    """
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    """
    As float.
    """
    try:
        return float(value)
    except Exception:
        return float(default)


def _label_counts(table: pd.DataFrame) -> dict[str, int]:
    """
    Label counts.
    """
    if "label" not in table.columns or table.empty:
        return {}
    return {str(k): int(v) for k, v in table["label"].value_counts().to_dict().items()}


def _defect_counts(table: pd.DataFrame) -> dict[str, int]:
    """
    Defect counts.
    """
    if "defect_type" not in table.columns or table.empty:
        return {}
    counts = table["defect_type"].value_counts().to_dict()
    out: dict[str, int] = {}
    for key, value in counts.items():
        name = str(key)
        if name == "none":
            continue
        out[name] = int(value)
    return out


def _n_bonds_total(table: pd.DataFrame) -> int:
    """
    N bonds total.
    """
    if "n_bonds" not in table.columns or table.empty:
        return 0
    series = pd.to_numeric(table["n_bonds"], errors="coerce").fillna(0.0)
    return int(round(float(series.sum()) / 2.0))


def _n_grains(table: pd.DataFrame) -> int:
    """
    N grains.
    """
    if "grain_id" not in table.columns or table.empty:
        return 0
    series = pd.to_numeric(table["grain_id"], errors="coerce").dropna()
    valid = {int(v) for v in series.tolist() if int(v) >= 0}
    return len(valid)


def _dpyr_stats(table: pd.DataFrame, *, tau_opt: float) -> dict[str, float]:
    """
    Dpyr stats.
    """
    if table.empty or "d_pyr" not in table.columns:
        return {"mean_abs": 0.0, "median_abs": 0.0, "frac_above_tau": 0.0}
    if "is_undercoord" in table.columns:
        data = table.loc[~table["is_undercoord"], "d_pyr"]
    else:
        data = table["d_pyr"]
    arr = pd.to_numeric(data, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {"mean_abs": 0.0, "median_abs": 0.0, "frac_above_tau": 0.0}
    abs_arr = np.abs(arr)
    return {
        "mean_abs": float(abs_arr.mean()),
        "median_abs": float(np.median(abs_arr)),
        "frac_above_tau": float((abs_arr > float(tau_opt)).sum() / max(abs_arr.size, 1)),
    }


def _ring_summary_line(ring_histogram: dict[str, int]) -> str:
    """
    Ring summary line.
    """
    total = int(sum(int(v) for v in ring_histogram.values()))
    if total <= 0:
        return "Ring histogram is unavailable for this run."
    n6 = int(ring_histogram.get("6", 0))
    n5 = int(ring_histogram.get("5", 0))
    n7 = int(ring_histogram.get("7", 0))
    p6 = 100.0 * float(n6) / float(total)
    p5 = 100.0 * float(n5) / float(total)
    p7 = 100.0 * float(n7) / float(total)
    if p6 >= 90.0:
        context = "mostly hexagonal, close to ordered graphitic structure."
    elif p6 >= 75.0:
        context = "defected hexagonal network, consistent with reconstructed graphite."
    else:
        context = "strongly reconstructed network with substantial non-hexagonal character."
    return (
        f"{total} primitive rings: {n6} hexagons ({p6:.1f}%), "
        f"{n5} pentagons ({p5:.1f}%), {n7} heptagons ({p7:.1f}%) - {context}"
    )


def _structural_variable_description(var_name: str, *, tau_opt: float) -> str:
    """
    Structural variable description.
    """
    base = {
        "source": "Input structure path used for this frame analysis.",
        "is_periodic": "Whether periodic boundary conditions were active for neighbor/ring analysis.",
        "N_atoms": "Total number of atoms in the analyzed frame.",
        "N_undercoord": "Count of under-coordinated atoms (coordination < 3 in local network logic).",
        "N_bonds": "Total bonds in the constructed bond graph.",
        "N_grains": "Number of psi6 orientation grains detected by region-growing.",
        "dpyr_stats.mean_abs": "Mean absolute pyramidalization |d_pyr| over non-under-coordinated carbon atoms.",
        "dpyr_stats.median_abs": "Median absolute pyramidalization |d_pyr| over non-under-coordinated carbon atoms.",
        "dpyr_stats.frac_above_tau": f"Fraction of non-under-coordinated carbon atoms with |d_pyr| above tau_opt ({tau_opt:.3f} A).",
    }
    if var_name in base:
        return base[var_name]
    if var_name.startswith("ring_histogram."):
        size = var_name.split(".", 1)[1]
        return f"Count of primitive rings of size {size} (Franzblau shortest-path ring enumeration)."
    if var_name.startswith("defect_cluster_counts."):
        defect = var_name.split(".", 1)[1]
        return f"Number of atoms assigned defect_type '{defect}' in per-atom defect classification."
    if var_name.startswith("label_counts."):
        label = var_name.split(".", 1)[1]
        return f"Number of carbon atoms assigned label '{label}'."
    return "Computed summary variable from active-site structural analysis output."


def _summary_block_to_table_rows(summary_block: dict[str, Any], *, tau_opt: float) -> list[list[str]]:
    """
    Summary block to table rows.
    """
    rows: list[list[str]] = []

    ordered_scalars = ("source", "is_periodic", "N_atoms", "N_undercoord", "N_bonds", "N_grains")
    for key in ordered_scalars:
        if key in summary_block:
            rows.append([key, str(summary_block[key]), _structural_variable_description(key, tau_opt=tau_opt)])

    dp = summary_block.get("dpyr_stats")
    if isinstance(dp, dict):
        for metric in ("mean_abs", "median_abs", "frac_above_tau"):
            key = f"dpyr_stats.{metric}"
            if metric in dp:
                rows.append([key, str(dp.get(metric)), _structural_variable_description(key, tau_opt=tau_opt)])

    ring_hist = summary_block.get("ring_histogram")
    if isinstance(ring_hist, dict):
        for size in sorted(ring_hist.keys(), key=lambda s: int(str(s)) if str(s).isdigit() else str(s)):
            key = f"ring_histogram.{size}"
            rows.append([key, str(ring_hist[size]), _structural_variable_description(key, tau_opt=tau_opt)])

    defect_counts = summary_block.get("defect_cluster_counts")
    if isinstance(defect_counts, dict):
        for defect, val in sorted(defect_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            key = f"defect_cluster_counts.{defect}"
            rows.append([key, str(val), _structural_variable_description(key, tau_opt=tau_opt)])

    label_counts = summary_block.get("label_counts")
    if isinstance(label_counts, dict):
        for label, val in sorted(label_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            key = f"label_counts.{label}"
            rows.append([key, str(val), _structural_variable_description(key, tau_opt=tau_opt)])

    return rows


def _grain_summary_line(n_grains: int, n_carbons: int) -> str:
    """
    Grain summary line.
    """
    if n_grains <= 0:
        return "No crystalline grains were detected from psi6 clustering."
    high_cut = max(200, n_carbons // 5)
    medium_cut = max(50, n_carbons // 20)
    if n_grains >= high_cut:
        quality = "very fragmented microstructure"
    elif n_grains >= medium_cut:
        quality = "fragmented microstructure"
    else:
        quality = "moderately segmented microstructure"
    return f"{n_grains} grains detected, indicating {quality}."


def _figure_entries(analysis_dir: Path) -> list[dict[str, str]]:
    """
    Figure entries.
    """
    suffixes = (
        ("_dpyr_map.png", "Spatial map of |d_pyr| with under-coordinated atoms highlighted."),
        ("_label_map.png", "Spatial map of site labels."),
        ("_grain_map.png", "Spatial map of grain IDs from psi6 region-growing."),
        ("_dpyr_hist.png", "Distribution of |d_pyr| by label; tau_opt marker included."),
    )
    out: list[dict[str, str]] = []
    for suffix, caption in suffixes:
        matches = sorted(analysis_dir.glob(f"*{suffix}"))
        if not matches:
            continue
        out.append({"path": str(matches[0]), "caption": caption})
    return out


def build_structural_report_payload(
    result: Any,
    args: Any,
    analysis_dir: Path,
    *,
    tau_opt: float = 0.229,
) -> dict[str, Any] | None:
    """Build a report payload for `active_site_structural` command outputs.

    Parameters
    -----
    result : Any
        Analyzer result object, expected to expose `table` and optional summary.
    args : Any
        Command arguments used for contextual report metadata.
    analysis_dir : Path
        Analysis output directory used to discover generated figure assets.
    tau_opt : float, optional
        Threshold used for pyramidalization interpretation metrics.

    Returns
    -----
    dict[str, Any] | None
        Structured report payload dictionary, or `None` when no valid table
        data is available.

    Examples
    -----
    ```python
    payload = build_structural_report_payload(result, args, Path("analysis"))
    ```
    Sample output:
    `{"title": "...", "sections": [...], "figures": [...]}`
    Meaning:
    The payload is ready for report renderer consumption.
    """
    table = getattr(result, "table", None)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return None

    summary = getattr(result, "summary", None)
    summary = dict(summary) if isinstance(summary, dict) else {}
    request = getattr(result, "request", None)

    carbon_element = str(
        getattr(request, "carbon_element", None)
        or getattr(args, "carbon_element", None)
        or "C"
    )
    carbon_table = table[table["element"] == carbon_element].copy() if "element" in table.columns else table.copy()
    if carbon_table.empty:
        carbon_table = table.copy()

    n_atoms = int(len(table))
    n_carbon = int(len(carbon_table))
    if "is_undercoord" in carbon_table.columns:
        n_undercoord = int(pd.to_numeric(carbon_table["is_undercoord"], errors="coerce").fillna(False).astype(bool).sum())
    else:
        n_undercoord = 0

    n_bonds = _as_int(summary.get("n_bonds_total"), default=_n_bonds_total(table))
    ring_histogram = summary.get("ring_histogram")
    ring_histogram = {str(k): _as_int(v) for k, v in (ring_histogram or {}).items()} if isinstance(ring_histogram, dict) else {}
    defect_cluster_counts = summary.get("defect_cluster_counts")
    if isinstance(defect_cluster_counts, dict):
        defect_counts = {str(k): _as_int(v) for k, v in defect_cluster_counts.items()}
    else:
        defect_counts = _defect_counts(carbon_table)
    n_grains = _as_int(summary.get("n_grains"), default=_n_grains(carbon_table))
    dpyr_stats = summary.get("dpyr_stats")
    if isinstance(dpyr_stats, dict):
        dp_stats = {
            "mean_abs": _as_float(dpyr_stats.get("mean_abs")),
            "median_abs": _as_float(dpyr_stats.get("median_abs")),
            "frac_above_tau": _as_float(dpyr_stats.get("frac_above_tau")),
        }
    else:
        dp_stats = _dpyr_stats(carbon_table, tau_opt=tau_opt)
    label_counts = _label_counts(carbon_table)

    is_periodic = bool(summary.get("is_periodic", False))
    source = str(getattr(args, "xmolout", None) or getattr(args, "input", None) or "")

    summary_block = {
        "source": source,
        "is_periodic": is_periodic,
        "N_atoms": n_atoms,
        "N_undercoord": n_undercoord,
        "N_bonds": n_bonds,
        "ring_histogram": ring_histogram,
        "defect_cluster_counts": defect_counts,
        "N_grains": n_grains,
        "dpyr_stats": {
            "mean_abs": round(float(dp_stats["mean_abs"]), 3),
            "median_abs": round(float(dp_stats["median_abs"]), 3),
            "frac_above_tau": round(float(dp_stats["frac_above_tau"]), 3),
        },
        "label_counts": label_counts,
    }
    detailed_rows = _summary_block_to_table_rows(summary_block, tau_opt=tau_opt)

    zz = int(label_counts.get("edge_zigzag", 0))
    ac = int(label_counts.get("edge_armchair", 0))
    undercoord_frac = 100.0 * float(n_undercoord) / float(max(n_carbon, 1))
    above_tau_pct = 100.0 * float(dp_stats["frac_above_tau"])

    interpretations = [
        _ring_summary_line(ring_histogram),
        _grain_summary_line(n_grains, n_carbon),
        (
            f"{n_undercoord} under-coordinated atoms ({undercoord_frac:.1f}% of {carbon_element}): "
            f"{ac} armchair and {zz} zigzag."
            + (" In a periodic cell, these correspond to internal pore rims." if is_periodic else "")
        ),
        (
            "Defect motif counts from defect_type: "
            + (", ".join(f"{k}={v}" for k, v in sorted(defect_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]) if defect_counts else "none")
            + "."
        ),
        (
            f"{above_tau_pct:.1f}% above tau_opt = {tau_opt:.3f} A among non-under-coordinated {carbon_element} atoms "
            f"(mean |d_pyr| = {float(dp_stats['mean_abs']):.3f}, median |d_pyr| = {float(dp_stats['median_abs']):.3f})."
        ),
    ]

    frame_idx = getattr(args, "frame", summary.get("frame_idx", 0))
    run_id = getattr(args, "run_id", None)

    return {
        "title": "Active Site Structural Analysis Report",
        "subtitle": f"command=active_site_structural | frame={frame_idx} | run_id={run_id}",
        "sections": [
            {
                "title": "Summary Snapshot",
                "key_values": {
                    "Source": source or "n/a",
                    "Periodic": is_periodic,
                    "Total atoms": n_atoms,
                    f"Total {carbon_element} atoms": n_carbon,
                    "Under-coordinated atoms": n_undercoord,
                    "Total bonds": n_bonds,
                    "Detected grains": n_grains,
                },
            },
            {
                "title": "Detailed Summary",
                "table": {
                    "headers": ["Variable", "Value", "Description"],
                    "rows": detailed_rows,
                },
            },
            {
                "title": "Interpretation",
                "bullets": interpretations,
            },
            {
                "title": "Label Counts",
                "key_values": {k: v for k, v in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0]))},
            },
        ],
        "figures": _figure_entries(analysis_dir),
    }


def _events_figure_entries(analysis_dir: Path) -> list[dict[str, str]]:
    """
    Events figure entries.
    """
    entries: list[dict[str, str]] = []
    patterns = (
        ("*events*.png", "Event analysis figure."),
        ("*event*.png", "Event analysis figure."),
    )
    seen: set[str] = set()
    for pattern, caption in patterns:
        for path in sorted(analysis_dir.glob(pattern)):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            entries.append({"path": str(path), "caption": caption})
    return entries


def _top_event_atom_lines(table: pd.DataFrame, *, limit: int = 5) -> list[str]:
    """
    Top event atom lines.
    """
    required = {"atom_id", "n_events_O", "n_events_Si"}
    if not required.issubset(table.columns):
        return []
    ranked = table.copy()
    ranked["total_events"] = (
        pd.to_numeric(ranked["n_events_O"], errors="coerce").fillna(0).astype(int)
        + pd.to_numeric(ranked["n_events_Si"], errors="coerce").fillna(0).astype(int)
    )
    ranked = ranked[ranked["total_events"] > 0].sort_values(
        ["total_events", "n_events_O", "n_events_Si", "atom_id"],
        ascending=[False, False, False, True],
    )
    lines: list[str] = []
    for _, row in ranked.head(max(0, int(limit))).iterrows():
        lines.append(
            f"atom_id={int(row['atom_id'])}: total={int(row['total_events'])}, "
            f"O={int(row['n_events_O'])}, Si={int(row['n_events_Si'])}"
        )
    return lines


def _events_variable_description(var_name: str) -> str:
    """
    Events variable description.
    """
    mapping = {
        "source": "Input trajectory/source path used for event extraction.",
        "mode": "Event detection mode: bo (bond-order) or dist (distance cutoff).",
        "frames_analyzed": "Number of analyzed frames after applying frame selection and stride.",
        "frame_first": "First analyzed frame index.",
        "frame_last": "Last analyzed frame index.",
        "every": "Stride applied to selected frames (every Nth frame).",
        "persist": "Minimum consecutive analyzed frames required to confirm a binding event.",
        "n_carbon": "Number of carbon atoms considered for event extraction.",
        "n_reactive_O": "Number of carbon atoms with at least one confirmed C-O event.",
        "n_reactive_Si": "Number of carbon atoms with at least one confirmed C-Si event.",
        "n_reactive_any": "Number of carbon atoms with at least one confirmed event of any target species.",
        "total_events_O": "Total confirmed C-O events across all carbon atoms.",
        "total_events_Si": "Total confirmed C-Si events across all carbon atoms.",
        "bo_threshold": "Bond-order threshold used to define a bonded C-X contact in bo mode.",
        "r_CO": "Distance cutoff (A) for C-O contact in dist mode.",
        "r_CSi": "Distance cutoff (A) for C-Si contact in dist mode.",
        "mean_contact_O_when_bound": "Mean C-O contact metric during bound frames (BO in bo mode, distance in dist mode).",
        "mean_contact_Si_when_bound": "Mean C-Si contact metric during bound frames (BO in bo mode, distance in dist mode).",
    }
    return mapping.get(var_name, "Computed summary variable from active-site events analysis output.")


def _events_summary_to_table_rows(summary_block: dict[str, Any]) -> list[list[str]]:
    """
    Events summary to table rows.
    """
    rows: list[list[str]] = []
    order = (
        "source",
        "mode",
        "frames_analyzed",
        "frame_first",
        "frame_last",
        "every",
        "persist",
        "n_carbon",
        "n_reactive_O",
        "n_reactive_Si",
        "n_reactive_any",
        "total_events_O",
        "total_events_Si",
        "bo_threshold",
        "r_CO",
        "r_CSi",
        "mean_contact_O_when_bound",
        "mean_contact_Si_when_bound",
    )
    for key in order:
        if key not in summary_block:
            continue
        rows.append([key, str(summary_block[key]), _events_variable_description(key)])
    return rows


def build_events_report_payload(
    result: Any,
    args: Any,
    analysis_dir: Path,
) -> dict[str, Any] | None:
    """Build a report payload for `active_site_events` command outputs.

    Parameters
    -----
    result : Any
        Analyzer result object expected to expose an events `table`.
    args : Any
        Command arguments used for contextual metadata and fallback values.
    analysis_dir : Path
        Analysis output directory used to discover generated figure assets.

    Returns
    -----
    dict[str, Any] | None
        Structured report payload dictionary, or `None` when event table data
        is unavailable.

    Examples
    -----
    ```python
    payload = build_events_report_payload(result, args, Path("analysis"))
    ```
    Sample output:
    `{"title": "...", "sections": [...], "figures": [...]}`
    Meaning:
    The payload captures summary metrics and interpretations for events output.
    """
    table = getattr(result, "table", None)
    if not isinstance(table, pd.DataFrame):
        return None

    summary = getattr(result, "summary", None)
    summary = dict(summary) if isinstance(summary, dict) else {}

    n_carbon = _as_int(summary.get("n_carbon"), default=len(table))
    n_reactive_o = _as_int(summary.get("n_reactive_O"), default=int(table.get("is_reactive_O", pd.Series(dtype=bool)).sum()))
    n_reactive_si = _as_int(summary.get("n_reactive_Si"), default=int(table.get("is_reactive_Si", pd.Series(dtype=bool)).sum()))
    n_reactive_any = _as_int(summary.get("n_reactive_any"), default=int(table.get("is_reactive_any", pd.Series(dtype=bool)).sum()))
    total_events_o = int(pd.to_numeric(table.get("n_events_O", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    total_events_si = int(pd.to_numeric(table.get("n_events_Si", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())

    mean_o = float(pd.to_numeric(table.get("mean_contact_O_when_bound", pd.Series(dtype=float)), errors="coerce").dropna().mean()) \
        if "mean_contact_O_when_bound" in table.columns and table["mean_contact_O_when_bound"].notna().any() else float("nan")
    mean_si = float(pd.to_numeric(table.get("mean_contact_Si_when_bound", pd.Series(dtype=float)), errors="coerce").dropna().mean()) \
        if "mean_contact_Si_when_bound" in table.columns and table["mean_contact_Si_when_bound"].notna().any() else float("nan")

    reactive_frac = 100.0 * float(n_reactive_any) / float(max(n_carbon, 1))
    if reactive_frac < 2.0:
        reactive_context = "sparse reactive-site activation"
    elif reactive_frac < 10.0:
        reactive_context = "moderate reactive-site activation"
    else:
        reactive_context = "high reactive-site activation"

    mode = str(summary.get("mode") or "unknown")
    frames_analyzed = _as_int(summary.get("frames_analyzed"))
    frame_first = _as_int(summary.get("frame_first"))
    frame_last = _as_int(summary.get("frame_last"))
    every = _as_int(summary.get("every"), default=1)
    persist = _as_int(summary.get("persist"), default=1)

    summary_block: dict[str, Any] = {
        "source": str(getattr(args, "xmolout", None) or getattr(args, "input", None) or ""),
        "mode": mode,
        "frames_analyzed": frames_analyzed,
        "frame_first": frame_first,
        "frame_last": frame_last,
        "every": every,
        "persist": persist,
        "n_carbon": n_carbon,
        "n_reactive_O": n_reactive_o,
        "n_reactive_Si": n_reactive_si,
        "n_reactive_any": n_reactive_any,
        "total_events_O": total_events_o,
        "total_events_Si": total_events_si,
    }
    if "bo_threshold" in summary:
        summary_block["bo_threshold"] = _as_float(summary.get("bo_threshold"))
    if "r_CO" in summary:
        summary_block["r_CO"] = _as_float(summary.get("r_CO"))
    if "r_CSi" in summary:
        summary_block["r_CSi"] = _as_float(summary.get("r_CSi"))
    if np.isfinite(mean_o):
        summary_block["mean_contact_O_when_bound"] = round(mean_o, 4)
    if np.isfinite(mean_si):
        summary_block["mean_contact_Si_when_bound"] = round(mean_si, 4)
    detailed_rows = _events_summary_to_table_rows(summary_block)

    top_atoms = _top_event_atom_lines(table, limit=5)

    interpretations = [
        (
            f"{n_reactive_any} reactive carbon atoms out of {n_carbon} "
            f"({reactive_frac:.2f}%): {reactive_context}."
        ),
        (
            f"Event totals: O={total_events_o}, Si={total_events_si}. "
            f"Reactive counts: O={n_reactive_o}, Si={n_reactive_si}."
        ),
        (
            f"Extraction window: frames {frame_first}..{frame_last} "
            f"({frames_analyzed} analyzed frames, every={every}, persist={persist})."
        ),
    ]
    if mode == "bo" and "bo_threshold" in summary:
        interpretations.append(f"Bond-order mode used with bo_threshold={_as_float(summary['bo_threshold']):.3f}.")
    elif mode == "dist":
        r_co = _as_float(summary.get("r_CO"), default=_as_float(getattr(args, "r_co", None), default=0.0))
        r_csi = _as_float(summary.get("r_CSi"), default=_as_float(getattr(args, "r_csi", None), default=0.0))
        interpretations.append(f"Distance mode used with r_CO={r_co:.3f} A and r_CSi={r_csi:.3f} A.")

    if np.isfinite(mean_o) or np.isfinite(mean_si):
        o_txt = f"{mean_o:.4f}" if np.isfinite(mean_o) else "n/a"
        si_txt = f"{mean_si:.4f}" if np.isfinite(mean_si) else "n/a"
        interpretations.append(f"Mean contact metric when bound: O={o_txt}, Si={si_txt}.")
    if summary.get("note"):
        interpretations.append(str(summary.get("note")))

    frame_hint = f"{frame_first}:{frame_last}:{every}" if frames_analyzed > 0 else "n/a"
    run_id = getattr(args, "run_id", None)

    sections: list[dict[str, Any]] = [
        {
            "title": "Summary Snapshot",
            "key_values": {
                "Mode": mode,
                "Run ID": run_id or "n/a",
                "Frames analyzed": frames_analyzed,
                "Frame window": frame_hint,
                "Persist threshold": persist,
                "Reactive C atoms (any)": n_reactive_any,
                "Reactive C atoms (O)": n_reactive_o,
                "Reactive C atoms (Si)": n_reactive_si,
                "Total O events": total_events_o,
                "Total Si events": total_events_si,
            },
        },
        {
            "title": "Detailed Summary",
            "table": {
                "headers": ["Variable", "Value", "Description"],
                "rows": detailed_rows,
            },
        },
        {
            "title": "Interpretation",
            "bullets": interpretations,
        },
    ]
    if top_atoms:
        sections.append({"title": "Top Reactive Atoms", "bullets": top_atoms})

    return {
        "title": "Active Site Events Analysis Report",
        "subtitle": f"command=active_site_events | run_id={run_id}",
        "sections": sections,
        "figures": _events_figure_entries(analysis_dir),
    }


def build_active_site_report_payload(
    command: str,
    result: Any,
    args: Any,
    analysis_dir: Path,
) -> dict[str, Any] | None:
    """Dispatch active-site report payload construction by command name.

    Parameters
    -----
    command : str
        Command name, typically `active_site_structural` or `active_site_events`.
    result : Any
        Analyzer result object to summarize.
    args : Any
        Command arguments used for metadata context.
    analysis_dir : Path
        Directory containing generated analysis artifacts.

    Returns
    -----
    dict[str, Any] | None
        Command-specific report payload dictionary, or `None` when unsupported.

    Examples
    -----
    ```python
    payload = build_active_site_report_payload("active_site_events", result, args, Path("analysis"))
    ```
    Sample output:
    Report payload dictionary or `None`.
    Meaning:
    The function routes to the proper active-site payload builder.
    """
    cmd = str(command).strip()
    if cmd == "active_site_structural":
        return build_structural_report_payload(result, args, analysis_dir)
    if cmd == "active_site_events":
        return build_events_report_payload(result, args, analysis_dir)
    return None


def register_active_site_report_payloads() -> None:
    """Register active-site report payload builders in the report registry.

    Parameters
    -----
    None

    Returns
    -----
    None
        Registers command-keyed payload builder callbacks as a side effect.

    Examples
    -----
    ```python
    register_active_site_report_payloads()
    ```
    Sample output:
    `None`
    Meaning:
    Active-site commands become discoverable by report payload dispatch.
    """
    from reaxkit.presentation.report_registry import register_report_payload_builder

    register_report_payload_builder("active_site_structural", build_active_site_report_payload)
    register_report_payload_builder("active_site_events", build_active_site_report_payload)


__all__ = [
    "build_structural_report_payload",
    "build_events_report_payload",
    "build_active_site_report_payload",
    "register_active_site_report_payloads",
]
