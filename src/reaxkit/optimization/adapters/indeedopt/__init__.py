"""Read short-mode engine samples and full-mode best refinement snapshots."""

import math
import re

from ..base import Reader
from ...models import finite, prediction, progress


def sample_predictions(reader, sample, score):
    if not re.fullmatch(r"[0-9]+", sample):
        return False
    path = reader.root / f"ffield-{sample}" / "jax_md_results" / "evaluation.json"
    if not path.is_file():
        reader.result.warnings.append(f"sample:{sample}: no JAX evaluation report; parity unavailable")
        return False
    report = reader.json(path)
    source = reader.relative(path)
    if not isinstance(report, dict) or not isinstance(report.get("items"), list):
        reader.result.warnings.append(f"{source}: invalid evaluation report")
        return False
    loss = finite(report.get("loss"))
    records = []
    errors = []
    for index, item in enumerate(report["items"]):
        if not isinstance(item, dict):
            break
        target, predicted, weight = (finite(item.get(key)) for key in ("target", "prediction", "weight"))
        if target is None or predicted is None or weight is None or weight <= 0:
            break
        try:
            error = ((predicted - target) / weight) ** 2
        except OverflowError:
            break
        if not math.isfinite(error):
            break
        errors.append(error)
        records.append(prediction(
            source, f"sample:{sample}", f"{source}:item:{index}", target, predicted,
            source_record=index, weight=weight, weight_semantics="accuracy_divisor",
            weighted_squared_error=error, engine_identity="JAX-ReaxFF (jax-md)",
        ))
    if (not records or len(records) != len(report["items"]) or loss is None
            or finite(score) is None or not math.isclose(loss, finite(score), rel_tol=1e-9, abs_tol=1e-8)
            or not math.isclose(sum(errors), loss, rel_tol=1e-9, abs_tol=1e-8)):
        reader.result.warnings.append(f"{source}: invalid values or loss mismatch; predictions withheld")
        return False
    reader.result.predictions.extend(records)
    reader.result.warnings.append(f"{source}: labels/units unavailable; prediction identities are source-local")
    return True


def convert(root, options=None):
    reader = Reader(root, options)
    path = reader.root / "short_results" / "results.csv"
    if path.is_file():
        for line_number, row in enumerate(reader.csv(path), 2):
            if row.get("sample") is None or row.get("loss") is None:
                reader.result.warnings.append(f"{path.name}:{line_number}: missing sample/loss")
                continue
            record = progress(
                reader.relative(path), line_number, row["loss"],
                native_step=row["sample"], step_kind="evaluation",
                candidate_id=f"sample:{row['sample']}", evaluation_kind="engine",
            )
            if sample_predictions(reader, row["sample"], row["loss"]):
                record["engine_identity"] = "JAX-ReaxFF (jax-md)"
            reader.result.progress.append(record)
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
