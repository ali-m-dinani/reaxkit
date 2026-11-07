"""SCM MCFF reports with explicit mapping fallback for unknown layouts."""

from ..base import Reader
from ..ams_reports import mcff_progress


def convert(root, options=None):
    reader = Reader(root, options)
    mcff_progress(reader)
    for path in sorted(reader.root.glob("ffield_last_accepted.*")):
        reader.artifact(path, "parameter_only")
    for name in ("ffield_best", "ffield_last"):
        reader.artifact(reader.root / name, "parameter_only")
    reader.artifact(reader.root / "fort.99")
    if (reader.root / "fort.99").exists():
        reader.result.warnings.append("fort.99 retained; prediction layout and candidate association require an AMS-version sample")
    else:
        reader.result.warnings.append("No fort.99 prediction report; parity data unavailable")
    return reader.result
