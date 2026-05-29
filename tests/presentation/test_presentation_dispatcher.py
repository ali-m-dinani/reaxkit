from __future__ import annotations

import argparse
from types import SimpleNamespace

import pandas as pd

from reaxkit.presentation.dispatcher import present_result


def test_present_result_prints_table_when_no_output_flags(capsys):
    args = argparse.Namespace(plot=None, show=False, save=None, export=None)
    result = SimpleNamespace(table=pd.DataFrame({"value": [1, 2]}))

    present_result("demo", result, args)

    out = capsys.readouterr().out
    assert "value" in out
    assert "1" in out
