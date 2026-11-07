"""Read short-mode engine samples and full-mode best refinement snapshots."""

from ..base import Reader
from ...models import progress


def convert(root, options=None):
    reader = Reader(root, options)
    path = reader.root / "short_results" / "results.csv"
    if path.is_file():
        for line_number, row in enumerate(reader.csv(path), 2):
            if row.get("sample") is None or row.get("loss") is None:
                reader.result.warnings.append(f"{path.name}:{line_number}: missing sample/loss")
                continue
            reader.result.progress.append(progress(
                reader.relative(path), line_number, row["loss"],
                native_step=row["sample"], step_kind="evaluation",
                candidate_id=f"sample:{row['sample']}", evaluation_kind="engine",
            ))
    for path in sorted(reader.root.rglob("bruteforce_best.npz")):
        values = reader.npz(path)
        if values and "error" in values:
            reader.result.progress.append(progress(reader.relative(path), "best", values["error"], evaluation_kind="engine"))
            reader.artifact(path, "parameter_only")
    for path in sorted((reader.root / "short_results").glob("best_*")):
        reader.artifact(path, "parameter_only" if path.name == "best_ffield" else "neither")
    if not reader.result.progress:
        reader.result.warnings.append("Full-mode dl-data tables are dataset-specific predictions, not a universal fitness history")
    return reader.result
