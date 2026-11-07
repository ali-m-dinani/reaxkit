"""Read cobalt evaluations and MoS2 best checkpoints without mixing losses."""

from ..base import Reader
from ...models import finite, progress


def convert(root, options=None):
    reader = Reader(root, options)
    path = reader.root / "history.csv"
    if path.is_file():
        for line_number, row in enumerate(reader.csv(path), 2):
            if not {"evaluation", "loss", "error"} <= row.keys():
                reader.result.warnings.append(f"{path.name}: unsupported history columns")
                break
            reader.result.progress.append(progress(
                reader.relative(path), line_number, row["loss"],
                native_step=finite(row["evaluation"]), step_kind="evaluation",
                candidate_evaluations=finite(row["evaluation"]),
                candidate_id=f"cobalt:{row['evaluation']}",
                status="penalty" if row["error"] else ("ok" if finite(row["loss"]) is not None else "nonfinite"),
                error=row["error"], objective="cobalt_subset_normalized_RMS",
            ))
    for path in sorted((reader.root / "checkpoints").glob("*")):
        values = reader.npz(path) if path.suffix == ".npz" else reader.json(path) if path.suffix == ".json" else None
        if isinstance(values, dict) and "best_loss" in values:
            reader.result.progress.append(progress(
                reader.relative(path), "best", values["best_loss"],
                best_score=finite(values["best_loss"]), candidate_id=path.stem,
                objective="mos2_native_energy_geometry_loss",
            ))
            reader.artifact(path, "parameter_only")
    reader.artifact(reader.root / "ffield_best", "parameter_only")
    summary_path = reader.root / "summary.json"
    if summary_path.is_file():
        summary = reader.json(summary_path)
        if isinstance(summary, dict) and "best_loss" in summary:
            reader.result.progress.append(progress(
                reader.relative(summary_path), "summary", summary["best_loss"],
                record_kind="summary", best_score=finite(summary["best_loss"]),
                candidate_evaluations=finite(summary.get("n_evals")),
                elapsed_seconds=finite(summary.get("wall_clock_s")),
                objective=summary.get("loss_definition"),
                training_groups=summary.get("training_groups"),
            ))
    reader.result.warnings.append("MoS2 loss includes per-atom energy offset/category weights and geometry; cobalt uses a different dataset/objective. Best-only snapshots are not full histories")
    if (reader.root / "benchmark_per_run.csv").exists():
        reader.artifact(reader.root / "benchmark_per_run.csv")
        reader.result.warnings.append("benchmark_per_run.csv reports post-hoc metrics; Loss(RMSE) is not assumed to be the native optimization score")
    return reader.result
