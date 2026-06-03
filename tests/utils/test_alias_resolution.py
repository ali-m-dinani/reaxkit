"""
Tests for alias resolution utilities.

Covers:
- resolve_alias_from_columns (exact, alias, case-insensitive, heuristic)
- _resolve_alias helper behavior and error paths
- available_keys exposure
- normalize_choice behavior
"""

from __future__ import annotations

import pytest
import pandas as pd

from reaxkit.utils.alias import (
    resolve_alias_from_columns,
    _resolve_alias,
    available_keys,
    normalize_choice,
)


# ----------------------
# resolve_alias_from_columns
# ----------------------

def test_resolve_exact_match():
    cols = ["iter", "E_pot", "time"]
    assert resolve_alias_from_columns(cols, "iter") == "iter"


def test_resolve_alias_match():
    cols = ["Iteration", "Energy", "Time(fs)"]
    assert resolve_alias_from_columns(cols, "time") == "Time(fs)"


def test_resolve_case_insensitive():
    cols = ["ITER", "epot"]
    assert resolve_alias_from_columns(cols, "iter") == "ITER"


def test_resolve_heuristic_contains():
    cols = ["total_iteration_count", "energy"]
    assert resolve_alias_from_columns(cols, "iter") == "total_iteration_count"


def test_resolve_returns_none_when_missing():
    cols = ["energy", "temperature"]
    assert resolve_alias_from_columns(cols, "iter") is None


# ----------------------
# _resolve_alias helper
# ----------------------

def test__resolve_alias_from_dataframe():
    df = pd.DataFrame(columns=["Iter", "Epot(kcal/mol)"])
    assert _resolve_alias(df, "iter") == "Iter"


def test__resolve_alias_from_handler_like():
    class Dummy:
        def dataframe(self):
            return pd.DataFrame(columns=["Time", "Dens(kg/dm3)"])

    h = Dummy()
    assert _resolve_alias(h, "time") == "Time"


def test__resolve_alias_raises_keyerror():
    df = pd.DataFrame(columns=["energy"])
    with pytest.raises(KeyError):
        _resolve_alias(df, "iter")


# ----------------------
# available_keys
# ----------------------

def test_available_keys_includes_canonical_and_raw():
    cols = ["Iter", "Epot(kcal/mol)"]
    keys = available_keys(cols)

    # raw columns
    assert "Iter" in keys
    # canonical aliases
    assert "iter" in keys
    assert "E_pot" in keys


# ----------------------
# normalize_choice
# ----------------------

def test_normalize_choice_alias():
    assert normalize_choice("Time(fs)") == "time"


def test_normalize_choice_canonical_passthrough():
    assert normalize_choice("iter") == "iter"


def test_normalize_choice_unknown_passthrough():
    assert normalize_choice("foobar") == "foobar"


def test_normalize_choice_empty():
    assert normalize_choice("") == ""
