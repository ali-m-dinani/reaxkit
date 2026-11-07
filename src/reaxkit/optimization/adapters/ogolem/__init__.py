"""Read completed bridge archives; exclude pending candidate directories."""

from ..base import Reader, jax_reports
from ...models import finite, progress


def convert(root, options=None):
    reader = Reader(root, options)
    for directory in sorted((reader.root / "evaluations").glob("candidate_*")):
        path = directory / "fitness.json"
        if not path.is_file():
            continue
        values = reader.json(path)
        if not isinstance(values, dict):
            continue
        checkpoint = directory / "checkpoint.json"
        metadata = reader.json(checkpoint) if checkpoint.is_file() else {}
        score = values.get("fitness", (metadata or {}).get("fitness"))
        reader.result.progress.append(progress(
            reader.relative(path), "candidate", score, step_kind="evaluation",
            candidate_id=directory.name, completed_at=(metadata or {}).get("completed_at"),
            engine=values.get("engine"), evaluation_settings=values.get("settings"),
        ))
        reader.artifact(directory / "ffield", "parameter_only")
        jax_reports(reader, (directory / "jax-output").glob("report_*.txt"))
        if score is None:
            reader.result.warnings.append(f"{directory.name}: prediction-only response requires original target manifest to reconstruct fitness")
    for name in ("best.ffield", "latest.ffield"):
        reader.artifact(reader.root / name, "parameter_only")
    state_path = reader.root / "progress.json"
    if state_path.is_file():
        state = reader.json(state_path)
        if isinstance(state, dict) and isinstance(state.get("latest"), dict):
            reader.result.progress.append(progress(
                reader.relative(state_path), "latest", state["latest"].get("fitness"),
                record_kind="summary", candidate_id=state["latest"].get("evaluation"),
                candidate_evaluations=finite(state.get("completed_evaluations")),
                best_score=finite(state.get("best", {}).get("fitness")),
                completed_at=state["latest"].get("completed_at"),
            ))
    return reader.result
