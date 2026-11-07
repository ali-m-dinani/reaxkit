"""SCM CMA-ES outputs with explicit version-specific column mapping."""

from ..base import Reader


def convert(root, options=None):
    reader = Reader(root, options)
    reader.configured_progress("cmaes_progress.txt", "generation")
    reader.artifact(reader.root / "ffield_best", "parameter_only")
    reader.artifact(reader.root / "cmaes_resume.txt", "native_state_at_last_improvement")
    reader.artifact(reader.root / "fort.99")
    reader.result.warnings.append("CMA-ES resume state is saved at improvements; it may predate the latest progress")
    return reader.result
