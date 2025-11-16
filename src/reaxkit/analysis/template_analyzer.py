"""analyzer template which can be used for new analysis scripts."""
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
