"""
fort.79 (parameter sensitivity) analysis utilities.

This module provides helpers for analyzing force-field parameter
sensitivity information written by ReaxFF into ``fort.79`` files
via ``Fort79Handler``.

Typical use cases include:

- computing relative sensitivities of force-field error to parameter changes
- identifying parameters with the strongest impact on total error
- post-processing sensitivity tables for ranking or visualization
"""


import pandas as pd
from reaxkit.io.handlers.fort79_handler import Fort79Handler


def get_fort79_data_with_diff_sensitivities(handler: Fort79Handler) -> pd.DataFrame:
    """
    Compute relative force-field error sensitivities from a ``fort.79`` file.

    Sensitivities are computed by normalizing ``diff1``, ``diff2``, and
    (optionally) ``diff4`` by ``diff3``, which represents the error obtained
    using the current value of each parameter.

    Works on
    --------
    Fort79Handler — ``fort.79``

    Parameters
    ----------
    handler : Fort79Handler
        Parsed ``fort.79`` handler containing sensitivity data.

    Returns
    -------
    pandas.DataFrame
        Sensitivity table with columns:
        - ``identifier``: parameter identifier
        - ``sensitivity1/3``: ``diff1 / diff3``
        - ``sensitivity2/3``: ``diff2 / diff3``
        - ``sensitivity4/3``: ``diff4 / diff3`` (NaN if ``diff4`` is absent)
        - ``min_sensitivity``: minimum sensitivity across available diffs
        - ``max_sensitivity``: maximum sensitivity across available diffs

    Examples
    --------
    >>> from reaxkit.io.handlers.fort79_handler import Fort79Handler
    >>> from reaxkit.analysis.per_file.fort79_analyzer import get_fort79_data_with_diff_sensitivities
    >>> h = Fort79Handler("fort.79")
    >>> df = get_fort79_data_with_diff_sensitivities(h)
    """
    df = handler.dataframe() if hasattr(handler, "dataframe") else handler._parse()[0]

    # ensure numeric and safe division
    result = pd.DataFrame()

    result['identider'] = df['identifier']

    if "diff1" in df and "diff3" in df:
        result["sensitivity1/3"] = df["diff1"] / df["diff3"]
    else:
        result["sensitivity1/3"] = pd.Series(dtype=float)

    if "diff2" in df and "diff3" in df:
        result["sensitivity2/3"] = df["diff2"] / df["diff3"]
    else:
        result["sensitivity2/3"] = pd.Series(dtype=float)

    # optional diff4 handling
    if "diff4" in df and "diff3" in df:
        result["sensitivity4/3"] = df["diff4"] / df["diff3"]
    else:
        result["sensitivity4/3"] = pd.Series([float("nan")] * len(result))

    result["min_sensitivity"] = result[["sensitivity1/3", "sensitivity2/3", "sensitivity4/3"]].min(axis=1)
    result["max_sensitivity"] = result[["sensitivity1/3", "sensitivity2/3", "sensitivity4/3"]].max(axis=1)

    return result


