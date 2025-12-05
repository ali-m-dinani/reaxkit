"""analyzer template which can be used for new analysis scripts.

This file provides a minimal, easy-to-extend structure for creating analysis
modules that operate on data exposed through a `TemplateHandler`. It illustrates
the typical workflow for analyzers in reaxkit: retrieving a DataFrame from a
handler, computing derived metrics, and returning results in tidy pandas
formats. Use this template as a starting point when developing new, specialized
analysis routines (e.g., energy processing, structural metrics, charge analysis,
or time-series extraction).

"""

from __future__ import annotations
from typing import List
import pandas as pd

from reaxkit.io.template_handler import TemplateHandler


def example_metric(handler: TemplateHandler) -> pd.DataFrame:
    """Compute an example metric from TemplateHandler data.
    """
    df = handler.dataframe()
    # Example: just return energy vs iteration
    return df[["iteration", "energy"]].copy()


def record_series(handler: TemplateHandler, field: str) -> List[float]:
    """Extract a list of values for a given field across frames.
    """
    return handler.dataframe()[field].tolist()
