"""Generate classified EOS, scan-curve, restraint, and HeatFO plot collections."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from shutil import copyfile

import pandas as pd

import reaxkit.engine  # noqa: F401 - register engine adapters
from reaxkit.analysis.force_field.report import (
    FFieldOptimizationReportEOSRequest,
    FFieldOptimizationReportEOSTask,
    FFieldOptimizationReportRestraintRequest,
    FFieldOptimizationReportRestraintTask,
    _force_field_optimization_curve_tables,
)
from reaxkit.core.platform.engine_resolver import resolve_engine
from reaxkit.core.runtime.progress import resolve_reporter
from reaxkit.core.storage.storage_layout import (
    add_storage_cli_arguments,
    normalize_storage_args,
)
from reaxkit.domain.data_models import ForceFieldOptimizationPlotBundleData
from reaxkit.presentation.persist import persist_analysis_result
from reaxkit.presentation.plot import plot as render_plot
from reaxkit.workflows.force_field_opt.charge import (
    build_charge_table,
    charge_plot_payloads,
)
from reaxkit.workflows.force_field_opt.cell_parameters import (
    build_cell_parameter_table,
    cell_parameter_plot_payloads,
)
from reaxkit.workflows.force_field_opt.energy_categories import (
    build_energy_category_tables,
    energy_bar_plot_payloads,
    energy_curve_plot_groups,
)
from reaxkit.workflows.force_field_opt.geometry_targets import (
    build_geometry_target_table,
    geometry_target_plot_payloads,
)
from reaxkit.workflows.force_field_opt.heatfo import (
    build_heatfo_table,
    heatfo_plot_payloads,
)
from reaxkit.workflows.force_field_opt.report_linkage import (
    add_trainset_links,
    build_report_trainset_links,
)
from reaxkit.workflows.file_tools.ffield_workflow import (
    EOS_SINGLE_FIGSIZE,
    QM_PLOT_COLOR,
    REAXFF_PLOT_COLOR,
    _eos_material_name,
    _eos_plot_filename,
    _eos_plot_groups,
    _prepare_eos_table,
)

ALL_COMMANDS = ("get_ffield_opt_plots",)
ALL_LEGACY_COMMANDS: tuple[str, ...] = ()
FIGURE_GENERATOR_TEMPLATE_FILENAME = "template_ffield_opt_figure_generator.xlsx"
NOT_PLOTTED_COLUMNS = [
    "report_line_number",
    "section",
    "title",
    "ffield_value",
    "qm_value",
    "weight",
    "error",
    "total_ff_error",
    "reason",
]


def _figure_generator_template_source() -> Path:
    """Return the packaged custom-figure workbook template."""
    return (
        Path(__file__).resolve().parent
        / "data"
        / FIGURE_GENERATOR_TEMPLATE_FILENAME
    )


def _copy_figure_generator_template(output_dir: Path) -> Path:
    """Copy the packaged workbook template into one plot-result folder."""
    source = _figure_generator_template_source()
    if not source.is_file():
        raise FileNotFoundError(f"Figure-generator template is missing: {source}")
    destination = output_dir / FIGURE_GENERATOR_TEMPLATE_FILENAME
    copyfile(source, destination)
    return destination


def _safe_name(value: object) -> str:
    """Return a short, collision-resistant filesystem-safe identifier."""
    raw = str(value)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    safe = safe or "identifier"
    max_length = 40
    if len(safe) <= max_length:
        return safe
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    prefix = safe[: max_length - len(digest) - 1].rstrip("._-")
    return f"{prefix}_{digest}"


def _restraint_coordinate(identifier: object) -> float | None:
    """Extract the restraint scan value encoded at the end of an identifier."""
    text = str(identifier).strip()
    decimal = re.search(r"(?<!\d)([+-]?\d+)_([0-9]+)$", text)
    if decimal:
        integer, fraction = decimal.groups()
        return float(f"{integer}.{fraction}")
    number = re.search(r"(?<![A-Za-z0-9])([+-]?\d+(?:\.\d+)?)$", text)
    return float(number.group(1)) if number else None


def _restraint_plot_groups(table: pd.DataFrame) -> list[dict[str, object]]:
    """Build paired ReaxFF/QM restraint groups using identifier scan values."""
    return _scan_plot_groups(table, curve_type="restraint")


def _scan_plot_groups(
    table: pd.DataFrame,
    *,
    curve_type: str,
) -> list[dict[str, object]]:
    """Build paired ReaxFF/QM groups from geo or identifier scan coordinates."""
    required = {"base_iden", "other_iden", "ffield_value", "qm_value"}
    if table.empty or not required.issubset(table.columns):
        return []

    work = table.copy()
    if "scan_coordinate" in work.columns:
        parsed = pd.to_numeric(work["scan_coordinate"], errors="coerce")
        work["scan_coordinate"] = parsed.fillna(
            work["other_iden"].map(_restraint_coordinate)
        )
    else:
        work["scan_coordinate"] = work["other_iden"].map(_restraint_coordinate)
    work["ffield_value"] = pd.to_numeric(work["ffield_value"], errors="coerce")
    work["qm_value"] = pd.to_numeric(work["qm_value"], errors="coerce")
    groups: list[dict[str, object]] = []
    for identifier, raw_group in work.groupby("base_iden", dropna=False, sort=False):
        coordinate_label = {
            "bond": "Bond length (angstrom)",
            "angle": "Angle (degrees)",
            "other_curve": "Curve coordinate",
            "restraint": "Restraint scan coordinate",
        }.get(curve_type, "Scan coordinate")
        if raw_group["scan_coordinate"].isna().all():
            raw_group = raw_group.copy()
            raw_group["scan_coordinate"] = range(1, len(raw_group) + 1)
            coordinate_label = "Curve point"
        plotted = raw_group.dropna(subset=["scan_coordinate"]).sort_values(
            "scan_coordinate", kind="stable"
        )
        if plotted.empty:
            continue
        reaxff = plotted.dropna(subset=["ffield_value"])
        qm = plotted.dropna(subset=["qm_value"])
        if reaxff.empty and qm.empty:
            continue
        groups.append(
            {
                "identifier": str(identifier),
                "xlabel": coordinate_label,
                "reaxff_x": reaxff["scan_coordinate"].astype(float).tolist(),
                "reaxff_y": reaxff["ffield_value"].astype(float).tolist(),
                "qm_x": qm["scan_coordinate"].astype(float).tolist(),
                "qm_y": qm["qm_value"].astype(float).tolist(),
            }
        )
    return groups


def _series(group: dict[str, object]) -> list[dict[str, object]]:
    """Return ReaxFF and QM series for one plot group."""
    series: list[dict[str, object]] = []
    if group["reaxff_x"]:
        series.append(
            {
                "x": group["reaxff_x"],
                "y": group["reaxff_y"],
                "label": "ReaxFF",
                "marker": "o",
                "color": REAXFF_PLOT_COLOR,
            }
        )
    if group["qm_x"]:
        series.append(
            {
                "x": group["qm_x"],
                "y": group["qm_y"],
                "label": "QM/Literature",
                "marker": "o",
                "color": QM_PLOT_COLOR,
            }
        )
    return series


def _render_groups(
        groups: list[dict[str, object]],
        output_dir: Path,
        *,
        curve_type: str,
) -> list[Path]:
    """Render one image per curve group and return the written paths."""
    paths: list[Path] = []
    for group in groups:
        identifier = str(group["identifier"])
        filename_identifier = str(group.get("filename_identifier", identifier))
        if curve_type == "eos":
            filename = _eos_plot_filename(filename_identifier)
            curve_dir = output_dir / _eos_material_name(identifier)
            xlabel = str(group.get("xlabel", "Volume"))
            title = f"EOS {identifier}"
        else:
            prefix = {
                "bond": "bond",
                "angle": "angle",
                "other_curve": "other_curve",
                "restraint": "restraint",
                "energy_curve": "energy_curve",
            }.get(curve_type, "curve")
            title_prefix = {
                "bond": "Bond Scan",
                "angle": "Angle Scan",
                "other_curve": "Other Curve",
                "restraint": "Restraint",
                "energy_curve": "Energy Curve",
            }.get(curve_type, "Curve")
            filename = f"{prefix}_{_safe_name(filename_identifier)}.png"
            curve_dir = output_dir
            xlabel = str(group.get("xlabel", "Scan coordinate"))
            title = f"{title_prefix} {identifier}"
        path = curve_dir / filename
        render_plot(
            {
                "plot_type": "single_plot",
                "series": _series(group),
                "xlabel": xlabel,
                "ylabel": "Relative energy (kcal/mol)",
                "title": title,
                "legend": True,
                "save": path,
                **({"figsize": EOS_SINGLE_FIGSIZE} if curve_type == "eos" else {}),
            }
        )
        paths.append(path)
    return paths


def _render_heatfo(
        table: pd.DataFrame,
        output_dir: Path,
        *,
        expressions_per_figure: int,
) -> list[Path]:
    """Render chunked paired-bar HeatFO figures."""
    paths: list[Path] = []
    for payload in heatfo_plot_payloads(
            table, expressions_per_figure=expressions_per_figure
    ):
        path = output_dir / str(payload["filename"])
        render_plot({**payload, "save": path})
        paths.append(path)
    return paths


def _render_charge(
    table: pd.DataFrame,
    output_dir: Path,
    *,
    entries_per_figure: int,
) -> list[Path]:
    """Render chunked paired-bar CHARGE figures."""
    paths: list[Path] = []
    for payload in charge_plot_payloads(
        table, entries_per_figure=entries_per_figure
    ):
        path = output_dir / str(payload["filename"])
        render_plot({**payload, "save": path})
        paths.append(path)
    return paths


def _render_cell_parameters(
    table: pd.DataFrame,
    output_dir: Path,
    *,
    entries_per_figure: int,
) -> list[Path]:
    """Render chunked paired-bar cell-parameter figures."""
    paths: list[Path] = []
    for payload in cell_parameter_plot_payloads(
        table, entries_per_figure=entries_per_figure
    ):
        path = output_dir / str(payload["filename"])
        render_plot({**payload, "save": path})
        paths.append(path)
    return paths


def _render_geometry_targets(
    table: pd.DataFrame,
    output_dir: Path,
    *,
    entries_per_figure: int,
) -> list[Path]:
    """Render chunked paired-bar GEOMETRY figures."""
    paths: list[Path] = []
    for payload in geometry_target_plot_payloads(
        table, entries_per_figure=entries_per_figure
    ):
        path = output_dir / str(payload["filename"])
        render_plot({**payload, "save": path})
        paths.append(path)
    return paths


def _render_energy_bars(
    table: pd.DataFrame,
    output_dir: Path,
    *,
    entries_per_figure: int,
    filename_prefix: str,
    title: str,
    ylabel: str,
) -> list[Path]:
    """Render one classified ENERGY grouped-bar collection."""
    paths: list[Path] = []
    for payload in energy_bar_plot_payloads(
        table,
        entries_per_figure=entries_per_figure,
        filename_prefix=filename_prefix,
        title=title,
        ylabel=ylabel,
    ):
        path = output_dir / str(payload["filename"])
        render_plot({**payload, "save": path})
        paths.append(path)
    return paths


def _positive_int(value: str) -> int:
    """Argparse converter for strictly positive integer limits."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _not_plotted_entries(
    data: ForceFieldOptimizationPlotBundleData,
    assigned_tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """Return fort.99 rows absent from every classified output table."""
    report = pd.DataFrame(
        {
            "report_line_number": data.report.linenos,
            "section": data.report.sections,
            "title": data.report.titles,
            "ffield_value": data.report.ffield_values,
            "qm_value": data.report.qm_values,
            "weight": data.report.weights,
            "error": data.report.errors,
            "total_ff_error": data.report.total_ff_error,
        }
    )
    assigned_line_numbers: set[int] = set()
    for table in assigned_tables:
        if table.empty or "report_line_number" not in table.columns:
            continue
        line_numbers = pd.to_numeric(
            table["report_line_number"], errors="coerce"
        ).dropna()
        assigned_line_numbers.update(line_numbers.astype(int).tolist())

    report_line_numbers = pd.to_numeric(
        report["report_line_number"], errors="coerce"
    )
    unassigned = report.loc[
        ~report_line_numbers.isin(assigned_line_numbers)
    ].copy()
    unassigned["reason"] = "not assigned to a plotted data type"
    return unassigned.loc[:, NOT_PLOTTED_COLUMNS].reset_index(drop=True)


def _load_plot_bundle(
        args: argparse.Namespace,
        *,
        normalized: dict[str, object] | None = None,
) -> ForceFieldOptimizationPlotBundleData:
    """Resolve the engine and load report, summary, trainset, and geo inputs."""
    normalized = normalized or normalize_storage_args(vars(args))
    detection_path = (
            normalized.get("_snapshot_source_dir")
            or normalized.get("input")
            or normalized.get("run_dir")
            or "."
    )
    adapter = resolve_engine(str(detection_path), engine=getattr(args, "engine", None))
    return adapter.load(ForceFieldOptimizationPlotBundleData, normalized)


def build_parser(
        parser: argparse.ArgumentParser,
        *,
        command: str,
) -> argparse.ArgumentParser:
    """Configure the ``get_ffield_opt_plots`` command parser."""
    if command not in ALL_COMMANDS:
        raise ValueError(f"Unsupported command: {command}")
    parser.set_defaults(progress=True)
    parser.description = (
        "Classify optimization ENERGY expressions using fort.99, trainset comments, "
        "fort.74 volumes, and geo BOND/ANGLE restraints. Write separate EOS, bond, "
        "angle, other-curve, energy-curve, energy-difference, reaction-energy, "
        "single-identifier ENERGY, "
        "restraint, Charge, Geometry, Cell Parameters, and HeatFO "
        "collections. "
        "Every collection "
        "contains a CSV with ReaxFF, QM/literature, group-comment, and inline-comment data. "
        "The root not_plotted_entries.csv audits fort.99 entries not assigned to any "
        "plotted data type.\n\n"
        "Examples:\n"
        "  1. Analyze fort.99, fort.74, trainset.in, and geo in the current directory and save under\n"
        "     reaxkit_workspace/analysis/get_ffield_opt_plots/<run-id>/:\n"
        "       reaxkit get_ffield_opt_plots\n\n"
        "  2. Save the complete classified plot collection to an explicit folder:\n"
        "       reaxkit get_ffield_opt_plots --output ffield_opt_plots\n\n"
        "  3. Analyze a different optimization run with explicit input paths:\n"
        "       reaxkit get_ffield_opt_plots --run-dir run --fort99 run/fort.99 "
        "--fort74 run/fort.74 --trainset run/trainset.in --geo run/geo\n\n"
        "  4. Limit each grouped-bar figure to three entries (six paired bars):\n"
        "       reaxkit get_ffield_opt_plots --entry-per-figure 3\n\n"
        "  5. Flip the sign of EOS energies before exporting and plotting them:\n"
        "       reaxkit get_ffield_opt_plots --flip-sign-for-eos"
    )
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--input", default=".", help="Input path used for engine detection")
    parser.add_argument(
        "--run-dir", "--dir", dest="run_dir", default=".", help="Optimization run directory"
    )
    parser.add_argument("--fort99", default="fort.99", help="Path to fort.99")
    parser.add_argument("--fort74", default="fort.74", help="Path to fort.74")
    parser.add_argument("--trainset", default="trainset.in", help="Path to trainset file")
    parser.add_argument(
        "--geo",
        default="geo",
        help="Path to the multi-structure geo file containing scan restraints",
    )
    parser.add_argument(
        "--entry-per-figure",
        type=_positive_int,
        default=6,
        help=(
            "Maximum entries per grouped-bar figure; every entry contributes "
            "one ReaxFF and one QM/literature bar (default: 6)."
        ),
    )
    parser.add_argument(
        "--flip-sign-for-eos",
        action="store_true",
        help=(
            "Flip the sign of EOS energy values before exporting and plotting; "
            "equivalent to get_ffield_opt_eos --flip-sign."
        ),
    )
    parser.add_argument(
        "--output",
        "--outdir",
        "--save",
        dest="output",
        default=None,
        help=(
            "Optional output-folder override. By default, save under "
            "reaxkit_workspace/analysis/get_ffield_opt_plots/<run-id>/."
        ),
    )
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None)
    add_storage_cli_arguments(parser)
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    """Generate EOS, geo-classified curve, restraint, and HeatFO collections."""
    if command not in ALL_COMMANDS:
        raise ValueError(f"Unsupported command: {command}")

    normalized = normalize_storage_args(vars(args))
    for key, value in normalized.items():
        setattr(args, key, value)

    reporter = resolve_reporter(vars(args))
    progress_stage = "ffield optimization plots"
    progress_total = 19
    reporter(progress_stage, 0, progress_total, "Loading optimization files")
    data = _load_plot_bundle(args, normalized=normalized)
    reporter(progress_stage, 1, progress_total, "Classifying training data")
    eos_result = FFieldOptimizationReportEOSTask().run(
        data, FFieldOptimizationReportEOSRequest(iden="all")
    )
    _prepare_eos_table(
        eos_result,
        flip_sign=bool(getattr(args, "flip_sign_for_eos", False)),
    )
    restraint_result = FFieldOptimizationReportRestraintTask().run(
        data, FFieldOptimizationReportRestraintRequest(iden="all")
    )
    curve_tables = _force_field_optimization_curve_tables(data)
    bond_table = curve_tables["bond"]
    angle_table = curve_tables["angle"]
    other_curve_table = curve_tables["other_curve"]
    heatfo_table = build_heatfo_table(data)
    charge_table = build_charge_table(data)
    cell_parameter_table = build_cell_parameter_table(data)
    geometry_target_table = build_geometry_target_table(data)
    energy_category_tables = build_energy_category_tables(data)
    energy_curve_table = energy_category_tables["energy_curve"]
    energy_difference_table = energy_category_tables["energy_difference"]
    reaction_energy_table = energy_category_tables["reaction_energy"]
    single_energy_table = energy_category_tables["single_energy"]
    reporter(progress_stage, 2, progress_total, "Linking fort.99 to the training set")
    trainset_links = build_report_trainset_links(data)
    eos_result.table = add_trainset_links(eos_result.table, trainset_links)
    restraint_result.table = add_trainset_links(
        restraint_result.table, trainset_links
    )
    heatfo_table = add_trainset_links(heatfo_table, trainset_links)
    bond_table = add_trainset_links(bond_table, trainset_links)
    angle_table = add_trainset_links(angle_table, trainset_links)
    other_curve_table = add_trainset_links(other_curve_table, trainset_links)
    charge_table = add_trainset_links(charge_table, trainset_links)
    cell_parameter_table = add_trainset_links(cell_parameter_table, trainset_links)
    geometry_target_table = add_trainset_links(geometry_target_table, trainset_links)
    energy_curve_table = add_trainset_links(energy_curve_table, trainset_links)
    energy_difference_table = add_trainset_links(
        energy_difference_table, trainset_links
    )
    reaction_energy_table = add_trainset_links(reaction_energy_table, trainset_links)
    single_energy_table = add_trainset_links(single_energy_table, trainset_links)
    assigned_tables = [
        eos_result.table,
        restraint_result.table,
        heatfo_table,
        bond_table,
        angle_table,
        other_curve_table,
        charge_table,
        cell_parameter_table,
        geometry_target_table,
        energy_curve_table,
        energy_difference_table,
        reaction_energy_table,
        single_energy_table,
    ]
    not_plotted_table = _not_plotted_entries(data, assigned_tables)
    not_plotted_table = add_trainset_links(not_plotted_table, trainset_links)

    reporter(progress_stage, 3, progress_total, "Preparing output folders")
    uses_workspace = not bool(getattr(args, "output", None))
    if uses_workspace:
        root = persist_analysis_result(command, eos_result, args, write_csv=False)
    else:
        root = Path(args.output).expanduser()
    eos_dir = root / "eos_plots"
    restraint_dir = root / "restraint_plots"
    heatfo_dir = root / "heatfo_plots"
    bond_dir = root / "bond_plots"
    angle_dir = root / "angle_plots"
    other_curve_dir = root / "other_curve_plots"
    charge_dir = root / "charge_plots"
    cell_parameter_dir = root / "cell_parameters_plots"
    geometry_target_dir = root / "geometry_plots"
    energy_curve_dir = root / "energy_curve_plots"
    other_bar_dir = root / "other_bar_plots"
    reaction_energy_dir = root / "reaction_energy_plots"
    eos_dir.mkdir(parents=True, exist_ok=True)
    restraint_dir.mkdir(parents=True, exist_ok=True)
    heatfo_dir.mkdir(parents=True, exist_ok=True)
    bond_dir.mkdir(parents=True, exist_ok=True)
    angle_dir.mkdir(parents=True, exist_ok=True)
    other_curve_dir.mkdir(parents=True, exist_ok=True)
    charge_dir.mkdir(parents=True, exist_ok=True)
    cell_parameter_dir.mkdir(parents=True, exist_ok=True)
    geometry_target_dir.mkdir(parents=True, exist_ok=True)
    energy_curve_dir.mkdir(parents=True, exist_ok=True)
    other_bar_dir.mkdir(parents=True, exist_ok=True)
    reaction_energy_dir.mkdir(parents=True, exist_ok=True)
    figure_generator_template = _copy_figure_generator_template(root)

    reporter(progress_stage, 4, progress_total, "Writing CSV collections")
    eos_csv = eos_dir / "eos.csv"
    restraint_csv = restraint_dir / "restraints.csv"
    heatfo_csv = heatfo_dir / "heatfo.csv"
    bond_csv = bond_dir / "bonds.csv"
    angle_csv = angle_dir / "angles.csv"
    other_curve_csv = other_curve_dir / "other_curves.csv"
    charge_csv = charge_dir / "charges.csv"
    cell_parameter_csv = cell_parameter_dir / "cell_parameters.csv"
    geometry_target_csv = geometry_target_dir / "geometry.csv"
    energy_curve_csv = energy_curve_dir / "energy_curves.csv"
    energy_difference_csv = other_bar_dir / "energy_differences.csv"
    single_energy_csv = other_bar_dir / "single_identifier_energies.csv"
    reaction_energy_csv = reaction_energy_dir / "reaction_energies.csv"
    not_plotted_csv = root / "not_plotted_entries.csv"
    eos_result.table.to_csv(eos_csv, index=False)
    restraint_result.table.to_csv(restraint_csv, index=False)
    heatfo_table.to_csv(heatfo_csv, index=False)
    bond_table.to_csv(bond_csv, index=False)
    angle_table.to_csv(angle_csv, index=False)
    other_curve_table.to_csv(other_curve_csv, index=False)
    charge_table.to_csv(charge_csv, index=False)
    cell_parameter_table.to_csv(cell_parameter_csv, index=False)
    geometry_target_table.to_csv(geometry_target_csv, index=False)
    energy_curve_table.to_csv(energy_curve_csv, index=False)
    energy_difference_table.to_csv(energy_difference_csv, index=False)
    single_energy_table.to_csv(single_energy_csv, index=False)
    reaction_energy_table.to_csv(reaction_energy_csv, index=False)
    not_plotted_table.to_csv(not_plotted_csv, index=False)

    reporter(progress_stage, 5, progress_total, "Rendering EOS plots")
    eos_images = _render_groups(
        _eos_plot_groups(eos_result.table), eos_dir, curve_type="eos"
    )
    reporter(progress_stage, 6, progress_total, "Rendering restraint plots")
    restraint_images = _render_groups(
        _restraint_plot_groups(restraint_result.table),
        restraint_dir,
        curve_type="restraint",
    )
    reporter(progress_stage, 7, progress_total, "Rendering bond plots")
    bond_images = _render_groups(
        _scan_plot_groups(bond_table, curve_type="bond"),
        bond_dir,
        curve_type="bond",
    )
    reporter(progress_stage, 8, progress_total, "Rendering angle plots")
    angle_images = _render_groups(
        _scan_plot_groups(angle_table, curve_type="angle"),
        angle_dir,
        curve_type="angle",
    )
    reporter(progress_stage, 9, progress_total, "Rendering other curve plots")
    other_curve_images = _render_groups(
        _scan_plot_groups(other_curve_table, curve_type="other_curve"),
        other_curve_dir,
        curve_type="other_curve",
    )
    reporter(progress_stage, 10, progress_total, "Rendering charge plots")
    charge_images = _render_charge(
        charge_table,
        charge_dir,
        entries_per_figure=int(args.entry_per_figure),
    )
    reporter(progress_stage, 11, progress_total, "Rendering cell-parameter plots")
    cell_parameter_images = _render_cell_parameters(
        cell_parameter_table,
        cell_parameter_dir,
        entries_per_figure=int(args.entry_per_figure),
    )
    reporter(progress_stage, 12, progress_total, "Rendering geometry plots")
    geometry_target_images = _render_geometry_targets(
        geometry_target_table,
        geometry_target_dir,
        entries_per_figure=int(args.entry_per_figure),
    )
    reporter(progress_stage, 13, progress_total, "Rendering heat-of-formation plots")
    heatfo_images = _render_heatfo(
        heatfo_table,
        heatfo_dir,
        expressions_per_figure=int(args.entry_per_figure),
    )
    reporter(progress_stage, 14, progress_total, "Rendering energy-curve plots")
    energy_curve_images = _render_groups(
        energy_curve_plot_groups(energy_curve_table),
        energy_curve_dir,
        curve_type="energy_curve",
    )
    reporter(progress_stage, 15, progress_total, "Rendering energy-difference plots")
    energy_difference_images = _render_energy_bars(
        energy_difference_table,
        other_bar_dir,
        entries_per_figure=int(args.entry_per_figure),
        filename_prefix="energy_differences",
        title="Energy Differences",
        ylabel="Energy difference (kcal/mol)",
    )
    reporter(progress_stage, 16, progress_total, "Rendering single-identifier plots")
    single_energy_images = _render_energy_bars(
        single_energy_table,
        other_bar_dir,
        entries_per_figure=int(args.entry_per_figure),
        filename_prefix="single_identifier_energies",
        title="Single-Identifier Energies",
        ylabel="Energy (kcal/mol)",
    )
    reporter(progress_stage, 17, progress_total, "Rendering reaction-energy plots")
    reaction_energy_images = _render_energy_bars(
        reaction_energy_table,
        reaction_energy_dir,
        entries_per_figure=int(args.entry_per_figure),
        filename_prefix="reaction_energies",
        title="Reaction Energies",
        ylabel="Reaction energy (kcal/mol)",
    )
    reporter(progress_stage, 18, progress_total, "Finalizing result metadata")
    if uses_workspace:
        settings_path = root / "settings.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        settings["artifacts"]["csv"] = [
            eos_csv.relative_to(root).as_posix(),
            restraint_csv.relative_to(root).as_posix(),
            heatfo_csv.relative_to(root).as_posix(),
            bond_csv.relative_to(root).as_posix(),
            angle_csv.relative_to(root).as_posix(),
            other_curve_csv.relative_to(root).as_posix(),
            charge_csv.relative_to(root).as_posix(),
            cell_parameter_csv.relative_to(root).as_posix(),
            geometry_target_csv.relative_to(root).as_posix(),
            not_plotted_csv.relative_to(root).as_posix(),
            energy_curve_csv.relative_to(root).as_posix(),
            energy_difference_csv.relative_to(root).as_posix(),
            single_energy_csv.relative_to(root).as_posix(),
            reaction_energy_csv.relative_to(root).as_posix(),
        ]
        settings["artifacts"]["figures"] = [
            path.relative_to(root).as_posix()
            for path in (
                *eos_images,
                *restraint_images,
                *bond_images,
                *angle_images,
                *other_curve_images,
                *charge_images,
                *cell_parameter_images,
                *geometry_target_images,
                *heatfo_images,
                *energy_curve_images,
                *energy_difference_images,
                *single_energy_images,
                *reaction_energy_images,
            )
        ]
        settings["artifacts"]["workbooks"] = [
            figure_generator_template.relative_to(root).as_posix()
        ]
        settings_path.write_text(
            json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8"
        )

    reporter(progress_stage, progress_total, progress_total, "Finished plot generation")
    if eos_images:
        print(f"[Done] EOS: {len(eos_images)} images and {eos_csv}")
    else:
        print(f"[Skipped] EOS: no plottable expressions; wrote {eos_csv}")
    print(f"[Done] Restraints: {len(restraint_images)} images and {restraint_csv}")
    print(f"[Done] Bond scans: {len(bond_images)} images and {bond_csv}")
    print(f"[Done] Angle scans: {len(angle_images)} images and {angle_csv}")
    print(
        f"[Done] Other curves: {len(other_curve_images)} images and {other_curve_csv}"
    )
    print(f"[Done] Charges: {len(charge_images)} images and {charge_csv}")
    print(
        f"[Done] Cell parameters: {len(cell_parameter_images)} images and "
        f"{cell_parameter_csv}"
    )
    print(
        f"[Done] Geometry targets: {len(geometry_target_images)} images and "
        f"{geometry_target_csv}"
    )
    print(
        f"[Done] Heat of formation: {len(heatfo_images)} images and {heatfo_csv}"
    )
    print(
        f"[Done] Energy curves: {len(energy_curve_images)} images and "
        f"{energy_curve_csv}"
    )
    print(
        f"[Done] Energy differences: {len(energy_difference_images)} images and "
        f"{energy_difference_csv}"
    )
    print(
        f"[Done] Single-identifier energies: {len(single_energy_images)} images and "
        f"{single_energy_csv}"
    )
    print(
        f"[Done] Reaction energies: {len(reaction_energy_images)} images and "
        f"{reaction_energy_csv}"
    )
    print(
        f"[Warning] Not plotted: {len(not_plotted_table)} entries and "
        f"{not_plotted_csv}"
    )
    print(
        f"[Info] Custom plots: use {figure_generator_template} with the dedicated "
        "CSV files in each plot subfolder."
    )
    print(f"Results saved in:\n  {root}")
    return 0


__all__ = [
    "ALL_COMMANDS",
    "ALL_LEGACY_COMMANDS",
    "FIGURE_GENERATOR_TEMPLATE_FILENAME",
    "build_parser",
    "run_main",
]
