"""Direct command workflow for force-field analyses."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd

from reaxkit.analysis import force_field as _force_field_tasks  # noqa: F401
from reaxkit.analysis.force_field.diagnostics import ParameterOptimizationDiagnosticRequest
from reaxkit.analysis.force_field.force_field import ForceFieldDataRequest, ForceFieldDataTask
from reaxkit.analysis.force_field.optimization import ForceFieldOptimizationRequest
from reaxkit.analysis.force_field.report import (
    ForceFieldOptimizationReportBulkModulusRequest,
    ForceFieldOptimizationReportEOSRequest,
    ForceFieldOptimizationReportRequest,
)
from reaxkit.analysis.force_field.structure_summary import StructureSummaryRequest
from reaxkit.analysis.force_field.trainset import TrainsetGroupCommentsRequest
from reaxkit.core.alias import normalize_choice, resolve_alias_from_columns
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.engine_registry import resolve_engine
from reaxkit.core.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.command_alias_resolver import resolve_command_name
from reaxkit.core.storage_layout import add_storage_cli_arguments, normalize_storage_args
from reaxkit.domain.data_models import ForceFieldParametersData, GeometrySummaryData
from reaxkit.presentation.dispatcher import export_result_csv, present_result

FORCE_FIELD_COMMANDS = (
    "force_field_data",
    "force_field_optimization",
    "structure_summary_data",
    "parameter_optimization_diagnostic",
    "parameter_optimization_most_sensitive",
    "parameter_optimization_tornado",
    "force_field_optimization_report",
    "force_field_optimization_report_eos",
    "force_field_optimization_report_bulk_modulus",
    "trainset_group_comments",
)

_FORCE_FIELD_TERM_SECTIONS = {
    "bond": ("i", "j"),
    "off_diagonal": ("i", "j"),
    "angle": ("i", "j", "k"),
    "torsion": ("i", "j", "k", "l"),
    "hbond": ("i", "j", "k"),
}


def _normalize_force_field_format(value: str) -> str:
    fmt = str(value).strip().lower()
    if fmt == "indices":
        return "raw"
    if fmt not in {"raw", "interpreted"}:
        raise ValueError("Force-field format must be 'raw', 'indices', or 'interpreted'.")
    return fmt


def _split_term_string(term: str) -> list[str]:
    text = str(term).strip()
    if not text:
        return []
    for sep in ("-", ",", " ", "_"):
        if sep in text:
            return [token for token in text.replace(",", " ").replace("-", " ").replace("_", " ").split() if token]
    if text.isdigit():
        return [text]

    tokens: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isupper():
            if i + 1 < len(text) and text[i + 1].islower():
                tokens.append(text[i : i + 2])
                i += 2
            else:
                tokens.append(ch)
                i += 1
        else:
            tokens.append(ch)
            i += 1
    return tokens


def _atom_maps_from_data(data: ForceFieldParametersData) -> tuple[dict[int, str], dict[str, int]]:
    atom_df = data.atom_parameters
    if atom_df.empty or "symbol" not in atom_df.columns:
        raise KeyError("ForceFieldParametersData.atom_parameters must include a non-empty 'symbol' column.")

    idx_to_sym: dict[int, str] = {}
    sym_to_idx: dict[str, int] = {}
    for idx, row in atom_df.iterrows():
        atom_idx = int(idx)
        symbol = str(row["symbol"]).strip()
        idx_to_sym[atom_idx] = symbol
        sym_to_idx[symbol] = atom_idx
    return idx_to_sym, sym_to_idx


def _make_term_series_indices(df: pd.DataFrame, cols: tuple[str, ...], *, unordered_2body: bool) -> pd.Series:
    if len(cols) == 2 and unordered_2body:
        a = df[cols[0]].astype("Int64")
        b = df[cols[1]].astype("Int64")
        lo = a.where(a <= b, b)
        hi = b.where(a <= b, a)
        return lo.astype(str) + "-" + hi.astype(str)

    parts = [df[col].astype("Int64").astype(str) for col in cols]
    out = parts[0]
    for part in parts[1:]:
        out = out + "-" + part
    return out


def _filter_force_field_table_by_term(
    data: ForceFieldParametersData,
    section: str,
    df: pd.DataFrame,
    *,
    term: str,
    unordered_2body: bool = True,
    any_order: bool = False,
) -> pd.DataFrame:
    if section not in _FORCE_FIELD_TERM_SECTIONS:
        raise ValueError(f"--term is only valid for sections: {', '.join(sorted(_FORCE_FIELD_TERM_SECTIONS))}.")

    cols = _FORCE_FIELD_TERM_SECTIONS[section]
    tokens = _split_term_string(term)
    if not tokens:
        return df

    if all(token.isdigit() for token in tokens):
        idxs = [int(token) for token in tokens]
    else:
        _, sym_to_idx = _atom_maps_from_data(data)
        idxs = []
        for token in tokens:
            if token not in sym_to_idx:
                raise KeyError(f"Unknown atom symbol {token!r}. Available: {sorted(sym_to_idx)}")
            idxs.append(int(sym_to_idx[token]))

    if len(idxs) != len(cols):
        raise ValueError(f"Term {term!r} implies {len(idxs)} atoms, but section '{section}' requires {len(cols)}.")

    is_2body = len(cols) == 2
    if is_2body and (unordered_2body or any_order):
        a, b = sorted(idxs)
        wanted = f"{a}-{b}"
        key = _make_term_series_indices(df, cols, unordered_2body=True)
        return df.loc[key == wanted].copy()

    if any_order and not is_2body:
        wanted_sorted = sorted(idxs)
        sub = df.loc[:, list(cols)].astype("Int64")

        def _row_matches(row: pd.Series) -> bool:
            values = [int(value) for value in row.tolist()]
            values.sort()
            return values == wanted_sorted

        mask = sub.apply(_row_matches, axis=1)
        return df.loc[mask].copy()

    wanted = "-".join(str(idx) for idx in idxs)
    key = _make_term_series_indices(df, cols, unordered_2body=False)
    return df.loc[key == wanted].copy()


def _load_force_field_data(args: argparse.Namespace) -> ForceFieldParametersData:
    normalized = normalize_storage_args(vars(args))
    adapter = resolve_engine(
        normalized.get("ffield") or normalized.get("input") or ".",
        engine=getattr(args, "engine", None),
    )
    return adapter.load(ForceFieldParametersData, normalized)


def _export_force_field_tables(tables: dict[str, pd.DataFrame], outdir: str | Path, *, fmt: str) -> None:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    suffix = "interpreted" if fmt == "interpreted" else "indices"
    for section, table in tables.items():
        table.to_csv(out_path / f"{section}_{suffix}.csv", index=True)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--input", default=".", help="Input file or directory for engine resolution")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection")
    parser.add_argument("--ffield", default="ffield", help="Path to ffield")
    parser.add_argument("--fort13", default="fort.13", help="Path to fort.13")
    parser.add_argument("--fort79", default="fort.79", help="Path to fort.79")
    parser.add_argument("--fort99", default="fort.99", help="Path to fort.99")
    parser.add_argument("--fort74", default="fort.74", help="Path to fort.74")
    parser.add_argument("--trainset", default="trainset.in", help="Path to trainset file")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level")
    add_storage_cli_arguments(parser)


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path")
    parser.add_argument("--export", default=None, help="Write the result table to CSV")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2")
    parser.add_argument("--xaxis", default=None, help="Optional x-axis column override")


def _load_geometry_summary(args: argparse.Namespace) -> GeometrySummaryData:
    normalized = normalize_storage_args(vars(args))
    adapter = resolve_engine(
        normalized.get("fort74") or normalized.get("input") or ".",
        engine=getattr(args, "engine", None),
    )
    return adapter.load(GeometrySummaryData, normalized)


def _build_force_field_data_request(args: argparse.Namespace) -> ForceFieldDataRequest:
    section = args.section if args.section else None
    fmt = _normalize_force_field_format(args.format)
    return ForceFieldDataRequest(
        section=section,
        interpret=(fmt == "interpreted"),
    )


def _build_force_field_optimization_request(args: argparse.Namespace) -> ForceFieldOptimizationRequest:
    return ForceFieldOptimizationRequest(epochs=args.epochs)


def _build_structure_summary_request(args: argparse.Namespace) -> StructureSummaryRequest:
    return StructureSummaryRequest(sort=args.sort, ascending=not args.descending)


def _build_parameter_optimization_diagnostic_request(args: argparse.Namespace) -> ParameterOptimizationDiagnosticRequest:
    return ParameterOptimizationDiagnosticRequest(
        interpret=bool(getattr(args, "interpret", False)),
    )


def _build_force_field_optimization_report_request(args: argparse.Namespace) -> ForceFieldOptimizationReportRequest:
    return ForceFieldOptimizationReportRequest()


def _build_force_field_optimization_report_eos_request(args: argparse.Namespace) -> ForceFieldOptimizationReportEOSRequest:
    return ForceFieldOptimizationReportEOSRequest(
        iden=args.iden,
    )


def _build_force_field_optimization_report_bulk_modulus_request(
    args: argparse.Namespace,
) -> ForceFieldOptimizationReportBulkModulusRequest:
    return ForceFieldOptimizationReportBulkModulusRequest(
        iden=args.iden,
        shift_min_to_zero=not args.no_shift_min_to_zero,
        flip_sign=args.flip_sign,
        min_points=int(args.min_points),
    )


def _build_trainset_group_comments_request(args: argparse.Namespace) -> TrainsetGroupCommentsRequest:
    return TrainsetGroupCommentsRequest(sort=args.sort)


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "force_field_data": _build_force_field_data_request,
    "force_field_optimization": _build_force_field_optimization_request,
    "structure_summary_data": _build_structure_summary_request,
    "parameter_optimization_diagnostic": _build_parameter_optimization_diagnostic_request,
    "parameter_optimization_most_sensitive": _build_parameter_optimization_diagnostic_request,
    "parameter_optimization_tornado": _build_parameter_optimization_diagnostic_request,
    "force_field_optimization_report": _build_force_field_optimization_report_request,
    "force_field_optimization_report_eos": _build_force_field_optimization_report_eos_request,
    "force_field_optimization_report_bulk_modulus": _build_force_field_optimization_report_bulk_modulus_request,
    "trainset_group_comments": _build_trainset_group_comments_request,
}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = resolve_command_name(command, task_names=FORCE_FIELD_COMMANDS)
    parser.set_defaults(command=canonical)
    parser.formatter_class = argparse.RawTextHelpFormatter

    _add_runtime_arguments(parser)
    _add_presentation_arguments(parser)

    if canonical == "force_field_data":
        parser.description = (
            "Load, filter, and export raw or interpreted force-field sections.\n\n"
            "Examples:\n"
            "  reaxkit force_field_data --section bond --term C-H --format interpreted --export CH_bond.csv\n"
            "  reaxkit force_field_data --section angle --term CCH --any-order --format interpreted --export all_CCH_angles.csv\n"
            "  reaxkit force_field_data --format interpreted --outdir ffield_export"
        )
        parser.add_argument("--section", default=None, help="Single section: general, atom, bond, off_diagonal, angle, torsion, hbond")
        parser.add_argument("--format", choices=["raw", "indices", "interpreted"], default="interpreted", help="Section view format")
        parser.add_argument("--term", default=None, help="Filter term, for example C-H, CCH, C-C-H, or 1-2")
        parser.add_argument("--ordered-2body", action="store_true", help="For bond/off_diagonal, keep i-j distinct from j-i")
        parser.add_argument("--any-order", action="store_true", help="For angle/torsion/hbond, match any atom-order permutation")
        parser.add_argument("--outdir", default=None, help="Export each selected section to a separate CSV file")
    elif canonical == "force_field_optimization":
        parser.description = (
            "Return total force-field error versus optimization epoch.\n\n"
            "Examples:\n"
            "  reaxkit force_field_optimization --fort13 fort.13 --plot single\n"
            "  reaxkit force_field_optimization --epochs 1 5 10 --export ff_opt.csv\n"
            "  reaxkit force_field_optimization --fort13 fort.13 --save ff_opt.png"
        )
        parser.add_argument("--epochs", type=int, nargs="*", default=None, help="Selected optimization epochs")
    elif canonical == "structure_summary_data":
        parser.description = (
            "Load structure-summary data from fort.74.\n\n"
            "Examples:\n"
            "  reaxkit structure_summary_data --sort V --export fort74.csv\n"
            "  reaxkit structure_summary_data --col Density --export fort74_density.csv\n"
            "  reaxkit structure_summary_data --sort Hf --descending\n"
            "  reaxkit structure_summary_data --plot single --xaxis identifier"
        )
        parser.add_argument("--sort", default=None, help="Optional sort column")
        parser.add_argument("--descending", action="store_true", help="Sort in descending order")
        parser.add_argument("--col", default="all", help="Single column to keep, preserving identifier")
    elif canonical == "parameter_optimization_diagnostic":
        parser.description = (
            "Compute sensitivity diagnostics from parameter-optimization data.\n\n"
            "Examples:\n"
            "  reaxkit parameter_optimization_diagnostic --export fort79_diag.csv\n"
            "  reaxkit parameter_optimization_diagnostic --plot single\n"
            "  reaxkit parameter_optimization_diagnostic --save fort79_diag.png"
        )
        parser.add_argument("--interpret", action="store_true", help="Interpret identifier triplets using force-field data")
    elif canonical == "parameter_optimization_most_sensitive":
        parser.description = (
            "Identify the parameter with the minimum sensitivity and inspect its epoch-wise ratios.\n\n"
            "Examples:\n"
            "  reaxkit parameter_optimization_most_sensitive --plot single\n"
            "  reaxkit parameter_optimization_most_sensitive --export most_sensitive.csv --export-all fort79_all.csv\n"
            "  reaxkit parameter_optimization_most_sensitive --save most_sensitive.png"
        )
        parser.add_argument("--interpret", action="store_true", help="Interpret identifier triplets using force-field data")
        parser.add_argument("--export-all", default=None, help="Optional CSV path for the full diagnostic table")
    elif canonical == "parameter_optimization_tornado":
        parser.description = (
            "Aggregate min/median/max parameter sensitivities into a tornado summary.\n\n"
            "Examples:\n"
            "  reaxkit parameter_optimization_tornado --top 6 --plot single\n"
            "  reaxkit parameter_optimization_tornado --top 10 --save tornado.png --export tornado.csv\n"
            "  reaxkit parameter_optimization_tornado --vline 1.0"
        )
        parser.add_argument("--interpret", action="store_true", help="Interpret identifier triplets using force-field data")
        parser.add_argument("--top", type=int, default=0, help="Only keep the top-N widest spans; 0 keeps all")
        parser.add_argument("--vline", type=float, default=1.0, help="Reference line value for the tornado plot")
    elif canonical == "force_field_optimization_report":
        parser.description = (
            "Load fort.99 report rows with QM-FF differences.\n\n"
            "Examples:\n"
            "  reaxkit force_field_optimization_report --export fort99.csv\n"
            "  reaxkit force_field_optimization_report --plot single --xaxis lineno"
        )
    elif canonical == "force_field_optimization_report_eos":
        parser.description = (
            "Build ENERGY-vs-volume data from fort.99 and fort.74.\n\n"
            "Examples:\n"
            "  reaxkit force_field_optimization_report_eos --iden MgO --plot single\n"
            "  reaxkit force_field_optimization_report_eos --iden all --export eos.csv\n"
            "  reaxkit force_field_optimization_report_eos --iden all --plot subplot --save eos.png"
        )
        parser.add_argument("--iden", default=None, help="Identifier to keep; use 'all' for all rows")
        parser.add_argument("--flip-sign", action="store_true", help="Flip the sign of the energy values before plotting/export")
    elif canonical == "force_field_optimization_report_bulk_modulus":
        parser.description = (
            "Fit a Vinet bulk modulus from fort.99 and fort.74.\n\n"
            "Examples:\n"
            "  reaxkit force_field_optimization_report_bulk_modulus --iden bulk_0\n"
            "  reaxkit force_field_optimization_report_bulk_modulus --iden all --export bulk_modulus.csv\n"
            "  reaxkit force_field_optimization_report_bulk_modulus --flip-sign --min-points 8"
        )
        parser.add_argument("--iden", default=None, help="Optional base identifier to fit; use 'all' for all eligible bases")
        parser.add_argument("--no-shift-min-to-zero", action="store_true", help="Do not shift minimum energy to zero before fitting")
        parser.add_argument("--flip-sign", action="store_true", help="Flip the sign of the energy values before fitting")
        parser.add_argument("--min-points", type=int, default=6, help="Minimum finite points required per base identifier")
    elif canonical == "trainset_group_comments":
        parser.description = (
            "Return unique trainset group comments by section.\n\n"
            "Examples:\n"
            "  reaxkit trainset_group_comments --export group_comments.csv\n"
            "  reaxkit trainset_group_comments --sort\n"
            "  reaxkit trainset_group_comments"
        )
        parser.add_argument("--sort", action="store_true", help="Sort the resulting comments")
    else:
        raise KeyError(f"Unsupported force-field command '{canonical}'.")

    return parser


def _prepare_result(command: str, result) -> object:
    if command == "force_field_data" and getattr(result, "table", None) is None and getattr(result, "tables", None):
        frames = []
        for section, table in result.tables.items():
            df = table.copy()
            df.insert(0, "section", section)
            frames.append(df)
        result.table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return result


def _filter_structure_summary_columns(result, requested_col: str) -> None:
    table = getattr(result, "table", None)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return

    raw = str(requested_col).strip()
    if not raw or raw.lower() == "all":
        return

    canonical = normalize_choice(raw)
    resolved = resolve_alias_from_columns(table.columns, canonical)
    if resolved is None:
        raise ValueError(
            f"Column {raw!r} not found. Available columns: {', '.join(str(col) for col in table.columns)}"
        )

    cols = []
    if "identifier" in table.columns:
        cols.append("identifier")
    if resolved != "identifier":
        cols.append(resolved)
    result.table = table.loc[:, cols].copy()
    if resolved != raw and resolved in result.table.columns:
        result.table = result.table.rename(columns={resolved: raw})


def _build_most_sensitive_result(base_result):
    table = getattr(base_result, "table", None)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return argparse.Namespace(table=pd.DataFrame(), values={}, metadata={})

    df = table.copy()
    idx_min = df["min_sensitivity"].idxmin()
    min_identifier = df.loc[idx_min, "identifier"]
    min_value = float(df.loc[idx_min, "min_sensitivity"])
    print(f"[Done] Minimum sensitivity value: {min_value:.6f}")
    print(f"Corresponding identifier: {min_identifier}")

    df["epoch_set"] = df.index + 1
    ratio_cols = [col for col in ("sensitivity1/3", "sensitivity2/3", "sensitivity4/3") if col in df.columns]
    subset = df.loc[df["identifier"] == min_identifier].copy()

    if ratio_cols:
        subset_long = (
            subset[["epoch_set", "identifier", "min_sensitivity", "max_sensitivity"] + ratio_cols]
            .melt(
                id_vars=["epoch_set", "identifier", "min_sensitivity", "max_sensitivity"],
                value_vars=ratio_cols,
                var_name="sensitivity_name",
                value_name="sensitivity",
            )
            .dropna(subset=["sensitivity"])
            .reset_index(drop=True)
        )
    else:
        subset_long = subset.loc[:, ["epoch_set", "identifier", "min_sensitivity", "max_sensitivity"]].copy()

    return argparse.Namespace(
        table=subset_long,
        values={"identifier": min_identifier, "min_sensitivity": min_value},
        metadata={"full_table": df},
    )


def _build_tornado_result(base_result, top: int):
    table = getattr(base_result, "table", None)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return argparse.Namespace(table=pd.DataFrame(), values={}, metadata={})

    sensitivities = table.copy()
    grouped = (
        sensitivities.groupby("identifier", dropna=True)
        .agg(min_eff=("min_sensitivity", "min"), max_eff=("max_sensitivity", "max"))
        .reset_index()
    )
    eff_union = (
        sensitivities.melt(
            id_vars=["identifier"],
            value_vars=["min_sensitivity", "max_sensitivity"],
            var_name="kind",
            value_name="eff",
        )
        .dropna(subset=["eff"])
    )
    grouped["median_eff"] = grouped["identifier"].map(eff_union.groupby("identifier", dropna=True)["eff"].median())
    grouped["span"] = grouped["max_eff"] - grouped["min_eff"]
    grouped = grouped.sort_values("span", ascending=False).reset_index(drop=True)
    if top and top > 0:
        grouped = grouped.head(int(top)).copy()
    return argparse.Namespace(table=grouped, values={}, metadata={})


def _prepare_eos_table(result, *, flip_sign: bool) -> None:
    table = getattr(result, "table", None)
    if not isinstance(table, pd.DataFrame) or table.empty or not flip_sign:
        return
    for col in ("E_other_iden", "ffield_value", "qm_value"):
        if col in table.columns:
            result.table[col] = -pd.to_numeric(table[col], errors="coerce")


def _run_force_field_data(args: argparse.Namespace) -> int:
    data = _load_force_field_data(args)
    request = REQUEST_BUILDERS["force_field_data"](args)
    task = ForceFieldDataTask()

    if args.term:
        if request.section is None:
            raise ValueError("--term requires exactly one selected section via --section.")
        selected_section = request.section

        raw_result = task.run(
            data,
            ForceFieldDataRequest(section=selected_section, interpret=False),
        )
        raw_table = raw_result.table
        filtered_raw = _filter_force_field_table_by_term(
            data,
            selected_section,
            raw_table,
            term=args.term,
            unordered_2body=not args.ordered_2body,
            any_order=args.any_order,
        )
        if request.interpret:
            interpreted_result = task.run(
                data,
                ForceFieldDataRequest(section=selected_section, interpret=True),
            )
            table = interpreted_result.table.loc[filtered_raw.index].copy()
        else:
            table = filtered_raw
        result = task.run(
            data,
            ForceFieldDataRequest(section=selected_section, interpret=request.interpret),
        )
        result.table = table
    else:
        result = task.run(data, request)

    result = _prepare_result("force_field_data", result)

    if args.outdir:
        export_tables: dict[str, pd.DataFrame] = {}
        if isinstance(getattr(result, "table", None), pd.DataFrame) and not result.table.empty:
            if "section" in result.table.columns:
                for section_name, group in result.table.groupby("section", dropna=False):
                    section_key = str(section_name)
                    export_tables[section_key] = group.drop(columns=["section"]).copy()
            else:
                selected = request.section or "force_field"
                export_tables[str(selected)] = result.table.copy()
        export_fmt = "interpreted" if request.interpret else "raw"
        _export_force_field_tables(export_tables, args.outdir, fmt=export_fmt)
        print(f"[Done] Exported section tables to {args.outdir}")

    if args.export:
        export_result_csv(result, args.export)
        print(f"[Done] Exported data to {args.export}")

    wants_plot = bool(getattr(args, "plot", None) or getattr(args, "save", None) or getattr(args, "show", False))
    if wants_plot or not (args.export or args.outdir):
        present_args = argparse.Namespace(**vars(args))
        present_args.export = None
        present_result("force_field_data", result, present_args, plot_payload_builder=_plot_payload)
    return 0


def _plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    table = getattr(result, "table", None)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return None

    if command == "force_field_optimization":
        return {
            "plot_type": "single_plot",
            "x": table["epoch"].tolist(),
            "y": pd.to_numeric(table["total_ff_error"], errors="coerce").tolist(),
            "xlabel": "Epoch",
            "ylabel": "Total FF Error",
            "title": "Force-Field Optimization",
        }

    if command == "parameter_optimization_diagnostic":
        return {
            "plot_type": "single_plot",
            "series": [
                {"x": table["identifier"].tolist(), "y": pd.to_numeric(table["min_sensitivity"], errors="coerce").tolist(), "label": "min"},
                {"x": table["identifier"].tolist(), "y": pd.to_numeric(table["max_sensitivity"], errors="coerce").tolist(), "label": "max"},
            ],
            "xlabel": "Identifier",
            "ylabel": "Sensitivity",
            "title": "Parameter Optimization Diagnostic",
            "legend": True,
        }

    if command == "parameter_optimization_most_sensitive":
        if "sensitivity" not in table.columns:
            return None
        series = []
        if "sensitivity_name" in table.columns:
            for name, group in table.groupby("sensitivity_name", dropna=False):
                series.append(
                    {
                        "x": pd.to_numeric(group["epoch_set"], errors="coerce").tolist(),
                        "y": pd.to_numeric(group["sensitivity"], errors="coerce").tolist(),
                        "label": str(name),
                        "marker": "o",
                        "linewidth": 0,
                        "markersize": 20,
                        "alpha": 1,
                    }
                )
        else:
            series.append(
                {
                    "x": pd.to_numeric(table["epoch_set"], errors="coerce").tolist(),
                    "y": pd.to_numeric(table["sensitivity"], errors="coerce").tolist(),
                    "label": str(table["identifier"].iloc[0]) if "identifier" in table.columns and not table.empty else "most_sensitive",
                    "marker": "o",
                    "linewidth": 0,
                    "markersize": 20,
                    "alpha": 1,
                }
            )
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": "Epoch Set",
            "ylabel": "Sensitivity (Error Response)",
            "title": "Most Sensitive Parameter Ratios",
            "legend": True,
        }

    if command == "parameter_optimization_tornado":
        return {
            "plot_type": "tornado_plot",
            "labels": table["identifier"].tolist(),
            "min_vals": pd.to_numeric(table["min_eff"], errors="coerce").tolist(),
            "max_vals": pd.to_numeric(table["max_eff"], errors="coerce").tolist(),
            "median_vals": pd.to_numeric(table["median_eff"], errors="coerce").tolist(),
            "xlabel": "Sensitivity (Error Response)",
            "ylabel": "Parameter",
            "title": "Parameter Sensitivity Tornado Plot",
            "vline": getattr(args, "vline", None),
        }

    if command == "force_field_optimization_report":
        x_col = getattr(args, "xaxis", None) or "lineno"
        if x_col not in table.columns:
            x_col = "lineno"
        y_col = "qm_ff_difference" if "qm_ff_difference" in table.columns else "error"
        return {
            "plot_type": "single_plot",
            "x": table[x_col].tolist(),
            "y": pd.to_numeric(table[y_col], errors="coerce").tolist(),
            "xlabel": x_col,
            "ylabel": y_col,
            "title": "Force-Field Optimization Report",
        }

    if command == "force_field_optimization_report_eos":
        work = table.copy()
        for col in ("V_other_iden", "E_other_iden"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        work = work.sort_values(["base_iden", "V_other_iden"]).reset_index(drop=True)
        groups = [(iden, grp.copy()) for iden, grp in work.groupby("base_iden", dropna=False)]
        if getattr(args, "plot", None) == "subplot" and len(groups) > 1:
            return {
                "plot_type": "multi_subplots",
                "subplots": [
                    [
                        {"x": grp["V_other_iden"].tolist(), "y": grp["E_other_iden"].tolist(), "label": f"{iden} energy"},
                    ]
                    for iden, grp in groups
                ],
                "xlabel": "Volume",
                "ylabel": "Energy",
                "title": "EOS Energy vs Volume",
                "legend": True,
                "grid": getattr(args, "grid", None),
            }
        series = []
        for iden, grp in groups:
            series.append({"x": grp["V_other_iden"].tolist(), "y": grp["E_other_iden"].tolist(), "label": f"{iden} energy"})
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": "Volume",
            "ylabel": "Energy",
            "title": "EOS Energy vs Volume",
            "legend": True,
        }

    if command == "structure_summary_data":
        x_col = getattr(args, "xaxis", None) or "identifier"
        if x_col not in table.columns:
            x_col = "identifier"
        numeric_cols = [col for col in table.columns if col != x_col and pd.api.types.is_numeric_dtype(table[col])]
        if not numeric_cols:
            return None
        if getattr(args, "plot", None) == "subplot":
            subplots = []
            for col in numeric_cols:
                subplots.append([{"x": table[x_col].tolist(), "y": pd.to_numeric(table[col], errors="coerce").tolist(), "label": col}])
            return {
                "plot_type": "multi_subplots",
                "subplots": subplots,
                "xlabel": x_col,
                "ylabel": "Value",
                "title": "Structure Summary",
                "legend": False,
                "grid": getattr(args, "grid", None),
            }
        series = [{"x": table[x_col].tolist(), "y": pd.to_numeric(table[col], errors="coerce").tolist(), "label": col} for col in numeric_cols]
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": x_col,
            "ylabel": "Value",
            "title": "Structure Summary",
            "legend": True,
        }

    if command == "force_field_data":
        x_col = getattr(args, "xaxis", None)
        if x_col is None or x_col not in table.columns:
            x_col = table.columns[0]
        numeric_cols = [col for col in table.columns if col != x_col and pd.api.types.is_numeric_dtype(table[col])]
        if not numeric_cols:
            return None
        first = numeric_cols[0]
        return {
            "plot_type": "single_plot",
            "x": table[x_col].tolist(),
            "y": pd.to_numeric(table[first], errors="coerce").tolist(),
            "xlabel": x_col,
            "ylabel": first,
            "title": "Force-Field Data",
        }

    return None


def run_main(command: str, args: argparse.Namespace) -> int:
    canonical = resolve_command_name(command, task_names=FORCE_FIELD_COMMANDS)
    if canonical == "force_field_data":
        return _run_force_field_data(args)

    if canonical == "parameter_optimization_most_sensitive":
        executor = AnalysisExecutor()
        base_result = executor.run(
            TASK_REGISTRY["parameter_optimization_diagnostic"](),
            REQUEST_BUILDERS["parameter_optimization_most_sensitive"](args),
            vars(args),
        )
        result = _build_most_sensitive_result(base_result)
        if getattr(args, "export_all", None):
            export_result_csv(argparse.Namespace(table=result.metadata["full_table"]), args.export_all)
            print(f"[Done] Exported full diagnostic table to {args.export_all}")
        present_result(canonical, result, args, plot_payload_builder=_plot_payload)
        return 0

    if canonical == "parameter_optimization_tornado":
        executor = AnalysisExecutor()
        base_result = executor.run(
            TASK_REGISTRY["parameter_optimization_diagnostic"](),
            REQUEST_BUILDERS["parameter_optimization_tornado"](args),
            vars(args),
        )
        result = _build_tornado_result(base_result, top=int(args.top))
        present_result(canonical, result, args, plot_payload_builder=_plot_payload)
        return 0

    task_cls = TASK_REGISTRY[canonical]
    request = REQUEST_BUILDERS[canonical](args)

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    result = _prepare_result(canonical, result)
    if canonical == "structure_summary_data":
        _filter_structure_summary_columns(result, getattr(args, "col", "all"))
    elif canonical == "force_field_optimization_report_eos":
        _prepare_eos_table(result, flip_sign=bool(getattr(args, "flip_sign", False)))
    present_result(canonical, result, args, plot_payload_builder=_plot_payload)
    return 0
