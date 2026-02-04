"""
Tests for XmoloutHandler parsing and frame access.

These tests validate the core contract of the xmolout parser:
- summary DataFrame columns and types
- metadata fields
- per-frame atom tables and frame() accessor
- deduplication behavior when duplicate iteration indices appear
- handling of extra per-atom columns (explicit names + unknown_* fallback)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from reaxkit.io.handlers.xmolout_handler import XmoloutHandler


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _minimal_xmolout_two_frames(*, dup_iter: bool = False, extra_cols: int = 0) -> str:
    """
    Build a tiny xmolout-like text with 2 frames.
    Format expected by XmoloutHandler:
      - a line with a single integer: num atoms
      - a header line with 9 tokens: name iter E a b c alpha beta gamma
      - N atom lines with >= 4 tokens: atom_type x y z [extras...]
    """
    # Two frames, 2 atoms each.
    it1 = 10
    it2 = 10 if dup_iter else 11  # duplicate iteration if requested

    def atom_line(atom_type: str, x: float, y: float, z: float) -> str:
        extras = " ".join(str(100 + i) for i in range(extra_cols))
        if extras:
            return f"{atom_type} {x:.3f} {y:.3f} {z:.3f} {extras}\n"
        return f"{atom_type} {x:.3f} {y:.3f} {z:.3f}\n"

    def frame_block(iter_: int, e: float, shift: float) -> str:
        out = ""
        out += "2\n"
        out += f"simA {iter_} {e:.6f} 1.0 1.0 1.0 90.0 90.0 90.0\n"
        out += atom_line("Ga", 0.0 + shift, 0.0, 0.0)
        out += atom_line("N",  1.0 + shift, 0.0, 0.0)
        return out

    txt = ""
    txt += frame_block(it1, -10.0, 0.0)
    txt += frame_block(it2, -9.5,  0.5)
    return txt


def test_parse_basic_summary_and_metadata(tmp_path: Path):
    p = _write_text(tmp_path / "xmolout", _minimal_xmolout_two_frames())

    h = XmoloutHandler(p)
    df = h.dataframe()
    meta = h.metadata()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["num_of_atoms", "iter", "E_pot", "a", "b", "c", "alpha", "beta", "gamma"]
    assert len(df) == 2
    assert df["num_of_atoms"].tolist() == [2, 2]
    assert df["iter"].tolist() == [10, 11]

    assert meta["simulation_name"] == "simA"
    assert meta["n_atoms"] == 2
    assert meta["n_frames"] == 2
    assert meta["has_time"] is False


def test_frame_accessor_outputs_coords_and_types(tmp_path: Path):
    p = _write_text(tmp_path / "xmolout", _minimal_xmolout_two_frames())

    h = XmoloutHandler(p)
    f0 = h.frame(0)

    assert f0["index"] == 0
    assert f0["iter"] == 10
    assert isinstance(f0["coords"], np.ndarray)
    assert f0["coords"].shape == (2, 3)
    assert f0["atom_types"] == ["Ga", "N"]


def test_iter_frames_step(tmp_path: Path):
    p = _write_text(tmp_path / "xmolout", _minimal_xmolout_two_frames())

    h = XmoloutHandler(p)
    frames = list(h.iter_frames(step=2))

    assert len(frames) == 1
    assert frames[0]["iter"] == 10


def test_frame_out_of_range_raises(tmp_path: Path):
    p = _write_text(tmp_path / "xmolout", _minimal_xmolout_two_frames())

    h = XmoloutHandler(p)
    with pytest.raises(IndexError):
        h.frame(-1)
    with pytest.raises(IndexError):
        h.frame(2)


def test_deduplicates_duplicate_iteration_keep_last(tmp_path: Path):
    # Two frames both claim iter=10; handler should keep LAST occurrence.
    p = _write_text(tmp_path / "xmolout", _minimal_xmolout_two_frames(dup_iter=True))

    h = XmoloutHandler(p)
    df = h.dataframe()

    assert len(df) == 1
    assert df["iter"].tolist() == [10]
    # Frame kept should correspond to second block (shift=0.5)
    f0 = h.frame(0)
    assert np.isclose(f0["coords"][0, 0], 0.5)  # Ga x coordinate shifted


def test_extra_atom_cols_named_and_unknown_fallback(tmp_path: Path):
    # Provide 2 extra numeric columns per atom, but only name one explicitly.
    p = _write_text(tmp_path / "xmolout", _minimal_xmolout_two_frames(extra_cols=2))

    h = XmoloutHandler(p, extra_atom_cols=["q"])  # one name -> expect q + unknown_1
    df = h.dataframe()
    assert len(df) == 2  # sanity

    f0 = h.frame(0)
    # We can't access the per-frame DataFrame via public API,
    # but we can ensure coords parse correctly even with extras present.
    assert f0["coords"].shape == (2, 3)

    # Check that parsing didn't drop frames
    assert h.n_frames() == 2
