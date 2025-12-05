"""analyzer for fort.79 file"""

import pandas as pd
from reaxkit.io.fort79_handler import Fort79Handler


def diff_sensitivities(handler: Fort79Handler) -> pd.DataFrame:
    """Compute the sensitivitys of diff1, diff2, and diff4 relative to diff3 from a fort.79 file,
    which is used to find the most effective parameter for a reduction in total force field error.
    diff3 is the error value obtained by the current value of this parameter while other
    diffs are related to the error values obtained by changing this parameter. hence, 
    their diff sensitivity shows the sensitivity of force field error to a change in that parameter.

    Output columns:
        sensitivity1 = diff1 / diff3
        sensitivity2 = diff2 / diff3
        sensitivity4 = diff4 / diff3  (if diff4 exists; otherwise NaN)
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


