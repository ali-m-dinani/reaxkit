"""Read JAX-SOPPI's row history without importing JAX."""

import json

from ..base import Reader
from ...models import finite, prediction, progress


def residual_pairs(predicted, reference, index=""):
    if isinstance(predicted, list) and isinstance(reference, list) and len(predicted) == len(reference):
        for position, (value, target) in enumerate(zip(predicted, reference)):
            yield from residual_pairs(value, target, f"{index}:{position}".lstrip(":"))
    elif isinstance(predicted, (list, dict)) or isinstance(reference, (list, dict)):
        raise ValueError("residual shape mismatch")
    else:
        yield index, predicted, reference


def convert(root, options=None):
    reader = Reader(root, options)
    path = reader.root / "history.jsonl"
    records = []
    if path.is_file():
        for line_number, line in reader.lines(path):
            try:
                records.append((line_number, json.loads(line)))
            except ValueError:
                reader.result.warnings.append(f"history.jsonl:{line_number}: invalid record")
    elif (reader.root / "checkpoint.json").is_file():
        path = reader.root / "checkpoint.json"
        checkpoint = reader.json(path)
        if isinstance(checkpoint, dict):
            records = list(enumerate(checkpoint.get("history", []), 1))
    segment = 0
    previous_step = None
    for line_number, row in records:
        if not isinstance(row, dict) or "step_index" not in row:
            reader.result.warnings.append(f"{path.name}:{line_number}: missing step_index")
            continue
        source = reader.relative(path)
        current_step = finite(row["step_index"])
        if current_step is None:
            reader.result.warnings.append(f"{path.name}:{line_number}: invalid step_index")
            continue
        if previous_step is not None and current_step <= previous_step:
            segment += 1
        previous_step = current_step
        candidate = f"segment:{segment}:row:{row['step_index']}:accepted-state"
        reader.result.progress.append(progress(
            source, line_number, row.get("current_error"),
            native_step=row["step_index"], step_kind="parameter_row",
            candidate_id=candidate, best_score=finite(row.get("best_error")),
            accepted=row.get("accepted"), parameter_key=row.get("key"), segment=segment,
        ))
        for attempt_index, attempt in enumerate(row.get("attempts", [])):
            for property_name, residuals in attempt.get("candidate_residuals", {}).items():
                predicted = residuals.get("predicted", [])
                references = residuals.get("reference", [])
                try:
                    pairs = list(residual_pairs(predicted, references))
                except ValueError:
                    reader.result.warnings.append(f"{source}:{line_number}: residual shape mismatch")
                    continue
                for item_index, value, target in pairs:
                    reader.result.predictions.append(prediction(
                        source, f"segment:{segment}:row:{row['step_index']}:attempt:{attempt_index}:candidate",
                        f"{property_name}:{item_index}", target, value,
                        property=property_name, identity_scope="source_property_index",
                    ))
    for name in ("ffield_best", "ffield_final"):
        reader.artifact(reader.root / name, "parameter_only")
    reader.artifact(reader.root / "checkpoint.json", "native_state_matching_fingerprint")
    if reader.result.predictions:
        reader.result.warnings.append("Residual arrays have no structure labels/units; source-local identities cannot be joined across optimizers")
    return reader.result
