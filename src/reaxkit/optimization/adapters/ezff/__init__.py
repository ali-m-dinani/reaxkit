"""Read epoch recommendations; preserve multi-objective vectors."""

from ..base import Reader
from ...models import finite, progress
from .cobalt import collect_cobalt


def convert(root, options=None):
    reader = Reader(root, options)
    parent = reader.root / "results" if (reader.root / "results").is_dir() else reader.root
    epochs = sorted((path for path in parent.iterdir() if path.is_dir() and path.name.isdigit()), key=lambda path: int(path.name))
    for epoch in epochs:
        path = epoch / "errors"
        if not path.is_file():
            continue
        for line_number, line in reader.lines(path):
            values = [finite(value) for value in line.split()]
            if not values:
                continue
            reader.result.progress.append(progress(
                reader.relative(path), line_number, values[0] if len(values) == 1 else None,
                native_step=int(epoch.name), step_kind="epoch",
                candidate_id=f"epoch:{epoch.name}:recommendation:{line_number}",
                objectives={f"objective_{index + 1}": value for index, value in enumerate(values)},
                status="ok" if all(value is not None for value in values) else "nonfinite",
                record_kind="recommendation",
            ))
        for path in sorted((epoch / "forcefields").glob("FF_*")):
            reader.artifact(path, "parameter_only")
        for path in sorted(epoch.glob("all_evaluated_*.npz")):
            reader.artifact(path)
    archives = [path for epoch in epochs for path in sorted(epoch.glob("all_evaluated_*.npz"))]
    archive = None
    if archives:
        path = archives[-1]
        archive = reader.npz(path)
        if isinstance(archive, dict) and isinstance(archive.get("errors"), list):
            for index, errors in enumerate(archive["errors"], 1):
                values = [finite(value) for value in errors] if isinstance(errors, list) else [finite(errors)]
                reader.result.progress.append(progress(
                    reader.relative(path), index, values[0] if len(values) == 1 else None,
                    native_step=index, step_kind="evaluation", candidate_evaluations=index,
                    counter_scope="native_archive_including_loaded_history",
                    candidate_id=f"evaluated:{index}", record_kind="evaluation",
                    objectives={f"objective_{component + 1}": value for component, value in enumerate(values)},
                    status="ok" if values and all(value is not None for value in values) else "nonfinite",
                ))
    collect_cobalt(reader, archive)
    reader.artifact(reader.root / "ffield_best", "parameter_only")
    reader.result.warnings.append("Epoch errors are recommended candidates, not all evaluations; component names/units and weights are defined by the external objective")
    return reader.result
