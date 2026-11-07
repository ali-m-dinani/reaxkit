"""SCM MCFF outputs with explicit version-specific column mapping."""

from ..base import Reader


def convert(root, options=None):
    reader = Reader(root, options)
    reader.configured_progress("MCFFOptimizer.log", "iteration")
    for name in ("ffield_best", "ffield_last"):
        reader.artifact(reader.root / name, "parameter_only")
    reader.artifact(reader.root / "fort.99")
    if (reader.root / "fort.99").exists():
        reader.result.warnings.append("fort.99 retained; prediction layout and candidate association require an AMS-version sample")
    return reader.result
