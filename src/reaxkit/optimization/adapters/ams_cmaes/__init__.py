"""SCM CMA-ES scalar reports with explicit mapping fallback."""

from ..base import Reader
from ..ams_reports import cmaes_progress


def convert(root, options=None):
    reader = Reader(root, options)
    cmaes_progress(reader)
    reader.artifact(reader.root / "ffield_best", "parameter_only")
    reader.artifact(reader.root / "cmaes_resume.txt", "native_state_at_last_improvement")
    reader.artifact(reader.root / "fort.99")
    reader.result.warnings.append("AMS prediction layout and candidate association remain unverified; no parity data extracted")
    reader.result.warnings.append("CMA-ES resume state is saved at improvements; it may predate the latest progress")
    return reader.result
