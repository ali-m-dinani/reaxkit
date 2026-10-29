from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from reaxkit.analysis.trajectory.rdf import RDFRequest, RDFResult, RDFTask
from reaxkit.presentation.kymograph import kymograph_grid
from reaxkit.presentation.plot import plot
from reaxkit.presentation.specs import spec_to_plot_payload
from reaxkit.workflows.trajectory_workflow import _plot_payload, build_parser


def _result(mode="single"):
    return RDFResult(
        table=pd.DataFrame({
            "frame_index": [7, 2, 7, 2], "iter": [70, 20, 70, 20],
            "r": [1.5, 0.5, 0.5, 1.5], "g": [8.0, 1.0, 4.0, 2.0],
        }), request=RDFRequest(plot_mode=mode),
    )


def test_kymograph_preserves_amplitudes_and_axis_orientation(tmp_path):
    result = _result("kymograph")
    specification = RDFTask.recommended_presentations(result, {"table": result.table.to_dict("records")})[1]
    payload = spec_to_plot_payload(specification, result)
    assert payload["x"] == [2, 7]
    assert payload["y"] == [0.5, 1.5]
    np.testing.assert_array_equal(payload["z"], [[1, 4], [2, 8]])
    figure = plot({**payload, "save": str(tmp_path / "kymograph.png")})
    assert (tmp_path / "kymograph.png").stat().st_size > 1000
    assert figure.axes[0].collections[0].get_clim() == (0, 8)


def test_kymograph_rejects_incompatible_radial_grids():
    table = _result().table
    table.loc[0, "r"] = 1.7
    with pytest.raises(ValueError, match="R grids differ"):
        kymograph_grid(table)


def test_single_frame_kymograph(tmp_path):
    table = _result().table.query("frame_index == 2")
    coordinates, radii, values = kymograph_grid(table)
    figure = plot({"plot_type": "kymograph", "x": coordinates, "y": radii, "z": values, "save": str(tmp_path / "single.png")})
    assert figure.axes[0].collections[0].get_array().size == 2


def test_separate_specs_contain_only_their_frame():
    result = _result("separate")
    specifications = RDFTask.recommended_presentations(result, {"table": result.table.to_dict("records")})[1:]
    payloads = [spec_to_plot_payload(specification, result) for specification in specifications]
    assert len(payloads) == 2
    assert payloads[0]["y"] == [1, 2]
    assert payloads[1]["y"] == [4, 8]


def test_cli_kymograph_and_separate_options():
    parser = argparse.ArgumentParser()
    build_parser(parser, command="get_rdf")
    for mode in ("kymograph", "separate"):
        assert parser.parse_args(["--plot", mode]).plot == mode
    payload = _plot_payload("get_rdf", _result(), argparse.Namespace(plot="kymograph", xaxis="iter"))
    assert payload["x"] == [20, 70]
    payloads = _plot_payload("get_rdf", _result(), argparse.Namespace(plot="separate"))
    assert [item["filename"] for item in payloads] == ["rdf_frame_000002.png", "rdf_frame_000007.png"]
    assert [item["y"] for item in payloads] == [[1, 2], [4, 8]]


def test_plotly_kymograph_and_separate():
    pytest.importorskip("plotly")
    from dataclasses import asdict
    from reaxkit.webui.presentation.registry import render_figure

    for mode in ("kymograph", "separate"):
        result = _result(mode)
        rows = result.table.to_dict("records")
        specifications = RDFTask.recommended_presentations(result, {"table": rows})[1:]
        figures = [render_figure(rows, presentation=asdict(specification)) for specification in specifications]
        if mode == "kymograph":
            np.testing.assert_array_equal(figures[0].data[0].z, [[1, 4], [2, 8]])
        else:
            assert list(figures[0].data[0].y) == [1, 2]
            assert list(figures[1].data[0].y) == [4, 8]
