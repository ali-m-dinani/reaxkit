"""Focused tests for the aggregate force-field optimization plot workflow."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd
from PIL import Image
import pytest
import reaxkit.workflows.force_field_opt.get_ffield_opt_plots as plots_module

from reaxkit.workflows.force_field_opt.get_ffield_opt_plots import (
    FIGURE_GENERATOR_TEMPLATE_FILENAME,
    _figure_generator_template_source,
    _render_groups,
    _restraint_coordinate,
    _restraint_plot_groups,
    _safe_name,
    _scan_plot_groups,
    _series,
    _not_plotted_entries,
    build_parser,
)
from reaxkit.workflows.file_tools.ffield_workflow import (
    EOS_SINGLE_FIGSIZE,
    _eos_plot_groups,
    _plot_payload,
)
from reaxkit.workflows.force_field_opt.heatfo import (
    _plot_identifier,
    heatfo_plot_payloads,
)
from reaxkit.workflows.force_field_opt.charge import charge_plot_payloads


def test_build_parser_documents_examples_and_defaults_to_workspace() -> None:
    parser = build_parser(
        argparse.ArgumentParser(), command="get_ffield_opt_plots"
    )

    assert "Examples:" in str(parser.description)
    assert "reaxkit get_ffield_opt_plots --output ffield_opt_plots" in str(
        parser.description
    )
    assert "reaxkit get_ffield_opt_plots --flip-sign-for-eos" in str(
        parser.description
    )
    assert parser.parse_args([]).output is None
    assert parser.parse_args([]).entry_per_figure == 6
    assert parser.parse_args([]).geo == "geo"
    assert parser.parse_args([]).progress is True
    assert parser.parse_args([]).flip_sign_for_eos is False
    assert parser.parse_args([]).make_powerpoint is False
    assert parser.parse_args(["--make-powerpoint"]).make_powerpoint is True
    assert parser.parse_args(["--make-ppt"]).make_powerpoint is True
    assert parser.parse_args(["--entry-per-figure", "3"]).entry_per_figure == 3
    assert parser.parse_args(["--flip-sign-for-eos"]).flip_sign_for_eos is True


def test_custom_figure_generator_template_is_packaged() -> None:
    template = _figure_generator_template_source()

    assert template.name == FIGURE_GENERATOR_TEMPLATE_FILENAME
    assert template.is_file()
    assert template.read_bytes().startswith(b"PK")


def test_curve_series_use_reaxff_blue_and_qm_red() -> None:
    series = _series(
        {
            "reaxff_x": [1.0, 2.0],
            "reaxff_y": [3.0, 4.0],
            "qm_x": [1.0, 2.0],
            "qm_y": [3.5, 4.5],
        }
    )

    assert [item["color"] for item in series] == ["tab:blue", "#C0504D"]


def test_eos_group_renderer_uses_word_table_friendly_dimensions(
    monkeypatch, tmp_path
) -> None:
    rendered_payloads: list[dict[str, object]] = []
    monkeypatch.setattr(
        "reaxkit.workflows.force_field_opt.get_ffield_opt_plots.render_plot",
        rendered_payloads.append,
    )
    groups = [
        {
            "identifier": "bulk_0_mgo",
            "xlabel": "Volume (Å³)",
            "reaxff_x": [9.0, 10.0],
            "reaxff_y": [-4.0, -5.0],
            "qm_x": [9.0, 10.0],
            "qm_y": [-4.1, -5.1],
        }
    ]

    _render_groups(groups, tmp_path, curve_type="eos")

    assert rendered_payloads[0]["figsize"] == EOS_SINGLE_FIGSIZE == (6.0, 5.0)


def test_aggregate_eos_matches_dedicated_plot_specifications(monkeypatch, tmp_path):
    table = pd.DataFrame(
        {
            "base_iden": ["bulk_0_mgo", "bulk_0_mgo"],
            "other_iden": ["bulk_0.9_mgo", "bulk_1.0_mgo"],
            "V_other_iden": [9.0, 10.0],
            "E_other_iden": [-4.1, -5.1],
            "ffield_value": [-4.0, -5.0],
            "qm_value": [-4.1, -5.1],
        }
    )
    dedicated = _plot_payload(
        "get_ffield_opt_eos",
        SimpleNamespace(table=table),
        argparse.Namespace(plot="single"),
    )
    rendered = []
    monkeypatch.setattr(plots_module, "render_plot", rendered.append)

    paths = _render_groups(_eos_plot_groups(table), tmp_path, curve_type="eos")

    assert len(rendered) == len(dedicated) == 1
    expected = dict(dedicated[0])
    expected_path = tmp_path / expected.pop("subdirectory") / expected.pop("filename")
    assert paths == [expected_path]
    assert rendered == [{**expected, "save": expected_path}]


@pytest.fixture
def stub_plot_workflow(monkeypatch):
    empty = pd.DataFrame()
    empty_result = lambda: SimpleNamespace(table=empty.copy())
    fake_task = lambda: SimpleNamespace(run=lambda *_args, **_kwargs: empty_result())

    monkeypatch.setattr(plots_module, "normalize_storage_args", lambda values: values)
    monkeypatch.setattr(plots_module, "resolve_reporter", lambda _values: lambda *_args: None)
    monkeypatch.setattr(plots_module, "_load_plot_bundle", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(plots_module, "FFieldOptimizationReportEOSTask", fake_task)
    monkeypatch.setattr(plots_module, "FFieldOptimizationReportRestraintTask", fake_task)
    monkeypatch.setattr(
        plots_module,
        "_force_field_optimization_curve_tables",
        lambda _data: {name: empty.copy() for name in ("bond", "angle", "other_curve")},
    )
    for name in (
        "build_heatfo_table",
        "build_charge_table",
        "build_cell_parameter_table",
        "build_geometry_target_table",
    ):
        monkeypatch.setattr(plots_module, name, lambda _data: empty.copy())
    monkeypatch.setattr(
        plots_module,
        "build_energy_category_tables",
        lambda _data: {
            name: empty.copy()
            for name in (
                "energy_curve",
                "energy_difference",
                "reaction_energy",
                "single_energy",
            )
        },
    )
    monkeypatch.setattr(plots_module, "build_report_trainset_links", lambda _data: {})
    monkeypatch.setattr(
        plots_module,
        "add_trainset_links",
        lambda table, _links: table,
    )
    monkeypatch.setattr(
        plots_module,
        "_not_plotted_entries",
        lambda _data, _tables: empty.copy(),
    )
    monkeypatch.setattr(
        plots_module,
        "_copy_figure_generator_template",
        lambda root: root / plots_module.FIGURE_GENERATOR_TEMPLATE_FILENAME,
    )
    for name in (
        "_render_groups",
        "_render_charge",
        "_render_cell_parameters",
        "_render_geometry_targets",
        "_render_heatfo",
        "_render_energy_bars",
    ):
        monkeypatch.setattr(plots_module, name, lambda *_args, **_kwargs: [])

def test_aggregate_workflow_skips_empty_eos_and_finishes(stub_plot_workflow, tmp_path, capsys):
    result = plots_module.run_main(
        "get_ffield_opt_plots",
        argparse.Namespace(
            output=str(tmp_path / "plots"),
            entry_per_figure=6,
            flip_sign_for_eos=False,
        ),
    )

    output = capsys.readouterr().out
    summary_path = tmp_path / "plots" / "plot_summary.txt"
    assert result == 0
    assert "[Skipped] EOS: no plottable expressions" in output
    assert "[Done] Restraints: 0 images" in output
    assert summary_path.read_text(encoding="utf-8") == output


@pytest.mark.parametrize("workspace", [False, True])
@pytest.mark.parametrize("with_figures", [False, True])
def test_powerpoint_export_uses_current_figures_and_records_artifact(
    stub_plot_workflow, monkeypatch, tmp_path, capsys, workspace, with_figures,
):
    root = tmp_path / "plots"
    root.mkdir()

    def persist(*args, **kwargs):
        (root / "settings.json").write_text(json.dumps({"artifacts": {}}))
        return root

    monkeypatch.setattr(plots_module, "persist_analysis_result", persist)

    def image(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (60, 40), "white").save(path)
        return path

    image(root / "eos_plots" / "stale.png")
    if with_figures:
        def render_groups(groups, directory, *, curve_type):
            if curve_type == "eos":
                return [image(directory / "material" / "eos_current.png")]
            if curve_type == "angle":
                return [image(directory / "angle_current.png")]
            return []

        def render_bars(table, directory, **kwargs):
            prefix = kwargs["filename_prefix"]
            if prefix in {"energy_differences", "single_identifier_energies"}:
                return [image(directory / f"{prefix}.png")]
            return []

        monkeypatch.setattr(plots_module, "_render_groups", render_groups)
        monkeypatch.setattr(plots_module, "_render_energy_bars", render_bars)
    assert plots_module.run_main("get_ffield_opt_plots", argparse.Namespace(
        output=None if workspace else str(root), entry_per_figure=6,
        flip_sign_for_eos=False, make_powerpoint=True,
    )) == 0
    output = capsys.readouterr().out
    deck = root / "ffield_opt_plots.pptx"
    assert deck.exists() is with_figures
    if with_figures:
        with ZipFile(deck) as archive:
            assert len([name for name in archive.namelist() if name.startswith("ppt/media/")]) == 4
            first_titles = []
            for index in (2, 4, 6):
                slide = ET.fromstring(archive.read(f"ppt/slides/slide{index}.xml"))
                first_titles.append(slide.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}t").text)
            assert first_titles == ["Equation of state", "Angle scans", "Other energy bars"]
            summary = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
            namespaces = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
            rows = summary.findall(".//a:tbl/a:tr", namespaces)[1:]
            counts = {row.find(".//a:t", namespaces).text:
                      int(row.findall(".//a:t", namespaces)[1].text) for row in rows}
            assert len(counts) == 13
            assert sum(counts.values()) == 4
            assert counts["Energy differences"] == counts["Single-identifier energies"] == 1
            assert counts["Charges"] == 0
            warning = next(line for line in output.splitlines() if line.startswith("[Warning] Not plotted:"))
            assert warning in [element.text for element in summary.findall(".//a:t", namespaces)]
        assert "[Done] PowerPoint:" in output
    else:
        assert "[Skipped] PowerPoint: no figures were generated" in output
    if workspace:
        settings = json.loads((root / "settings.json").read_text())
        assert settings["artifacts"]["presentations"] == ([deck.name] if with_figures else [])


def test_heatfo_payloads_limit_expressions_and_keep_series_colors() -> None:
    table = pd.DataFrame(
        {
            "expression": [f"+structure_{index}/1" for index in range(13)],
            "plot_identifier": [f"structure_{index}" for index in range(13)],
            "ffield_value": [float(index) for index in range(13)],
            "qm_value": [float(index) + 0.5 for index in range(13)],
        }
    )

    payloads = heatfo_plot_payloads(table, expressions_per_figure=3)

    assert len(payloads) == 5
    assert [len(payload["labels"]) for payload in payloads] == [3, 3, 3, 3, 1]
    assert all(
        [series["color"] for series in payload["series"]]
        == ["tab:blue", "#C0504D"]
        for payload in payloads
    )
    assert all(
        sum(len(series["values"]) for series in payload["series"])
        == 2 * len(payload["labels"])
        for payload in payloads
    )
    assert payloads[0]["labels"] == ["structure_0", "structure_1", "structure_2"]
    assert payloads[0]["group_width"] == 0.48
    assert payloads[0]["grid"] is False
    assert payloads[0]["ylabel"] == "Heat of formation (kcal/mol)"


def test_charge_payloads_limit_entries_and_keep_series_colors() -> None:
    table = pd.DataFrame(
        {
            "label": [f"structure [atom {index}]" for index in range(31)],
            "ffield_value": [float(index) / 10 for index in range(31)],
            "qm_value": [float(index) / 10 + 0.1 for index in range(31)],
        }
    )

    payloads = charge_plot_payloads(table, entries_per_figure=5)

    assert len(payloads) == 7
    assert [len(payload["labels"]) for payload in payloads] == [5, 5, 5, 5, 5, 5, 1]
    assert all(
        [series["color"] for series in payload["series"]]
        == ["tab:blue", "#C0504D"]
        for payload in payloads
    )
    assert all(
        sum(len(series["values"]) for series in payload["series"])
        == 2 * len(payload["labels"])
        for payload in payloads
    )


def test_heatfo_plot_identifier_uses_the_uniquely_signed_operand() -> None:
    mostly_negative = pd.Series(
        {
            "op1": "+",
            "id1": "AlBN2_t_1008557",
            "op2": "-",
            "id2": "Al_cubic_134",
            "op3": "-",
            "id3": "B_trigonal_160",
            "op4": "-",
            "id4": "N2_cubic_154",
        }
    )
    mostly_positive = pd.Series(
        {
            "op1": "+",
            "id1": "reference_a",
            "op2": "+",
            "id2": "reference_b",
            "op3": "+",
            "id3": "reference_c",
            "op4": "-",
            "id4": "target_structure",
        }
    )

    assert _plot_identifier(mostly_negative) == "AlBN2_t_1008557"
    assert _plot_identifier(mostly_positive) == "target_structure"


def test_restraint_coordinate_supports_encoded_decimal_suffixes() -> None:
    assert _restraint_coordinate("H2BCH3_0_652") == 0.652
    assert _restraint_coordinate("scan_12") == 12.0
    assert _restraint_coordinate("dimer_3_5") == 3.5
    assert _restraint_coordinate("scan_diss") is None


def test_safe_name_shortens_long_windows_filenames_with_stable_hash() -> None:
    first = "Akarsh_Nadire_vanDuin_DFT_data_OHAlNBH_ANGLE_AlNB_1225"
    second = "Akarsh_Nadire_vanDuin_DFT_data_OHAlNBH_ANGLE_AlBN_1236"

    assert _safe_name("short_name") == "short_name"
    assert len(_safe_name(first)) <= 40
    assert _safe_name(first) == _safe_name(first)
    assert _safe_name(first) != _safe_name(second)


def test_restraint_groups_pair_reaxff_and_qm_values() -> None:
    table = pd.DataFrame(
        {
            "base_iden": ["H2BCH3", "H2BCH3"],
            "other_iden": ["H2BCH3_0_652", "H2BCH3_0_700"],
            "ffield_value": [-3.0, -2.0],
            "qm_value": [-4.0, -3.0],
            "group_comment": ["Restraint H2BCH3_bond"] * 2,
            "inline_comment": ["first", "second"],
        }
    )

    groups = _restraint_plot_groups(table)

    assert groups == [
        {
            "identifier": "H2BCH3",
            "xlabel": "Restraint scan coordinate",
            "reaxff_x": [0.652, 0.7],
            "reaxff_y": [-3.0, -2.0],
            "qm_x": [0.652, 0.7],
            "qm_y": [-4.0, -3.0],
        }
    ]


def test_scan_groups_use_geo_coordinates_and_curve_specific_axis_labels() -> None:
    table = pd.DataFrame(
        {
            "base_iden": ["AlNB_125", "AlNB_125"],
            "other_iden": ["AlNB_100", "AlNB_105"],
            "scan_coordinate": [100.0, 105.0],
            "ffield_value": [8.0, 5.0],
            "qm_value": [8.693, 5.425],
        }
    )

    groups = _scan_plot_groups(table, curve_type="angle")

    assert groups[0]["xlabel"] == "Angle (degrees)"
    assert groups[0]["reaxff_x"] == [100.0, 105.0]
    assert groups[0]["qm_y"] == [8.693, 5.425]


def test_not_plotted_entries_are_raw_report_rows_absent_from_all_tables() -> None:
    data = SimpleNamespace(
        report=SimpleNamespace(
            linenos=[11, 12, 13],
            sections=["ENERGY", "TORSION", "CHARGE"],
            titles=["Energy +a/1 -b/1", "unknown torsion", "mol charge atom: 1"],
            ffield_values=[1.0, 2.0, 3.0],
            qm_values=[1.1, 2.2, 3.3],
            weights=[0.1, 0.2, 0.3],
            errors=[0.01, 0.02, 0.03],
            total_ff_error=[4.0, 4.0, 4.0],
        )
    )
    eos_table = pd.DataFrame({"report_line_number": [11, pd.NA]})
    charge_table = pd.DataFrame({"report_line_number": [13]})

    result = _not_plotted_entries(data, [eos_table, charge_table])

    assert result["report_line_number"].tolist() == [12]
    assert result["section"].tolist() == ["TORSION"]
    assert result["title"].tolist() == ["unknown torsion"]
    assert result["ffield_value"].tolist() == [2.0]
    assert result["reason"].tolist() == [
        "not assigned to a plotted data type"
    ]
