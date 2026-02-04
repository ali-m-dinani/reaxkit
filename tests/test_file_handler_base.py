import pandas as pd
import pytest
from pathlib import Path

from reaxkit.io.base_handler import BaseHandler


# ---- Dummy concrete handler for testing the abstract base ----
class DummyHandler(BaseHandler):
    def _parse(self):
        df = pd.DataFrame(
            {
                "iter": [0, 1, 2],
                "energy": [-10.0, -9.5, -9.2],
            }
        )
        meta = {"n_frames": 3}
        return df, meta


# ---- Tests ----

def test_filehandler_is_abstract():
    """FileHandler cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseHandler("some_file")


def test_parse_is_lazy(tmp_path: Path):
    """Parsing should not occur until dataframe() or metadata() is accessed."""
    h = DummyHandler(tmp_path / "dummy.out")
    assert h._parsed is False


def test_parse_sets_dataframe_and_metadata(tmp_path: Path):
    """Calling parse() populates dataframe and metadata."""
    h = DummyHandler(tmp_path / "dummy.out")
    h.parse()

    assert h._parsed is True
    assert isinstance(h.dataframe(), pd.DataFrame)
    assert h.metadata()["n_frames"] == 3


def test_dataframe_triggers_parse(tmp_path: Path):
    """Accessing dataframe() triggers parsing automatically."""
    h = DummyHandler(tmp_path / "dummy.out")

    df = h.dataframe()
    assert h._parsed is True
    assert list(df.columns) == ["iter", "energy"]


def test_metadata_triggers_parse(tmp_path: Path):
    """Accessing metadata() triggers parsing automatically."""
    h = DummyHandler(tmp_path / "dummy.out")

    meta = h.metadata()
    assert h._parsed is True
    assert meta["n_frames"] == 3


def test_dataframe_is_cached(tmp_path: Path):
    """Parsed DataFrame should be cached and reused."""
    h = DummyHandler(tmp_path / "dummy.out")

    df1 = h.dataframe()
    df2 = h.dataframe()

    assert df1 is df2


def test_metadata_is_copied(tmp_path: Path):
    """metadata() should return a copy, not the internal dict."""
    h = DummyHandler(tmp_path / "dummy.out")

    meta = h.metadata()
    meta["n_frames"] = 999

    assert h.metadata()["n_frames"] == 3
