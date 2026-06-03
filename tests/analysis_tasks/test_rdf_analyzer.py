"""
Tests for RDF_analyzer.

Scope:
- pure utility helpers (_dominant_peak, _first_local_max)
- end-to-end RDF property extraction over frames (skips if backend deps missing)

Notes:
- FREUD and OVITO are optional dependencies. Tests that require them are skipped
  automatically if the dependency is not available in the environment.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from reaxkit.io.handlers.xmolout_handler import XmoloutHandler
from reaxkit.analysis.composed.RDF_analyzer import (
    _dominant_peak,
    _first_local_max,
    rdf_property_over_frames,
)


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _minimal_xmolout_two_frames() -> str:
    """
    Build a tiny xmolout-like text with 2 frames and 2 atoms each.
    Format expected by XmoloutHandler:
      - line 1: num atoms (int)
      - line 2: sim_name iter E a b c alpha beta gamma
      - N atom lines: atom_type x y z
    """
    txt = ""
    # Frame 0
    txt += "2\n"
    txt += "simA 0 -10.000000 10.0 10.0 10.0 90.0 90.0 90.0\n"
    txt += "Al 0.000 0.000 0.000\n"
    txt += "N  1.900 0.000 0.000\n"  # near 1.9 Å
    # Frame 1 (slightly different separation)
    txt += "2\n"
    txt += "simA 1 -9.500000 10.0 10.0 10.0 90.0 90.0 90.0\n"
    txt += "Al 0.000 0.000 0.000\n"
    txt += "N  2.100 0.000 0.000\n"  # near 2.1 Å
    return txt


# ----------------------
# Pure utilities
# ----------------------

def test_dominant_peak_empty_returns_nan():
    r = np.asarray([], float)
    g = np.asarray([], float)
    rp, gp = _dominant_peak(r, g)
    assert np.isnan(rp)
    assert np.isnan(gp)


def test_dominant_peak_returns_global_max():
    r = np.asarray([0.5, 1.0, 1.5, 2.0], float)
    g = np.asarray([0.9, 1.2, 3.4, 2.1], float)
    rp, gp = _dominant_peak(r, g)
    assert rp == 1.5
    assert gp == 3.4


def test_first_local_max_finds_first_or_falls_back():
    r = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5], float)
    g = np.asarray([0.8, 1.5, 1.2, 2.0, 1.7], float)  # first local max at r=1.0
    rp, gp = _first_local_max(r, g)
    assert rp == 1.0
    assert gp == 1.5

    # monotonic => falls back to dominant peak
    g2 = np.asarray([0.5, 0.7, 0.9, 1.1, 1.3], float)
    rp2, gp2 = _first_local_max(r, g2)
    assert rp2 == 2.5
    assert gp2 == 1.3


# ----------------------
# End-to-end properties (backend-dependent)
# ----------------------

@pytest.mark.parametrize("backend", ["freud", "ovito"])
def test_rdf_property_over_frames_peak_columns(tmp_path: Path, backend: str):
    """
    Verify rdf_property_over_frames returns a per-frame DataFrame with expected columns.
    Skips automatically if selected backend dependency is missing.
    """
    if backend == "freud":
        pytest.importorskip("freud")
    else:
        pytest.importorskip("ovito")

    p = _write_text(tmp_path / "xmolout", _minimal_xmolout_two_frames())
    xh = XmoloutHandler(p)

    df = rdf_property_over_frames(
        xh,
        backend=backend,
        property="first_peak",
        frames=[0, 1],
        r_max=4.0,
        bins=50,
    )

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["frame_index", "iter", "r_first_peak", "g_first_peak"]
    assert df["frame_index"].tolist() == [0, 1]
    assert df["iter"].tolist() == [0, 1]
    assert np.all(np.isfinite(df["r_first_peak"].to_numpy()))
    assert np.all(np.isfinite(df["g_first_peak"].to_numpy()))


@pytest.mark.parametrize("backend", ["freud", "ovito"])
def test_rdf_property_over_frames_area_columns(tmp_path: Path, backend: str):
    if backend == "freud":
        pytest.importorskip("freud")
    else:
        pytest.importorskip("ovito")

    p = _write_text(tmp_path / "xmolout", _minimal_xmolout_two_frames())
    xh = XmoloutHandler(p)

    df = rdf_property_over_frames(
        xh,
        backend=backend,
        property="area",
        frames=[0, 1],
        r_max=4.0,
        bins=50,
    )

    assert list(df.columns) == ["frame_index", "iter", "area"]
    assert len(df) == 2
    assert np.all(np.isfinite(df["area"].to_numpy()))


def test_rdf_property_over_frames_invalid_property_raises(tmp_path: Path):
    p = _write_text(tmp_path / "xmolout", _minimal_xmolout_two_frames())
    xh = XmoloutHandler(p)

    with pytest.raises(ValueError):
        rdf_property_over_frames(xh, property="not_a_real_property")


def test_rdf_property_over_frames_invalid_backend_raises(tmp_path: Path):
    p = _write_text(tmp_path / "xmolout", _minimal_xmolout_two_frames())
    xh = XmoloutHandler(p)

    with pytest.raises(ValueError):
        rdf_property_over_frames(xh, backend="nope", property="first_peak")
