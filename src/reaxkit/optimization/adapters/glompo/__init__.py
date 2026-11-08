"""Read file-only GloMPO exports without importing optimizer dependencies."""

import hashlib
import json

from ..base import Reader
from ...models import finite, progress


def convert(root, options=None):
    reader = Reader(root, options)
    directory = reader.root / "manager" if (reader.root / "manager").is_dir() else reader.root
    history = directory / "glompo_history.jsonl"
    metadata = directory / "glompo_history.meta.json"
    native = directory / "glompo_log.h5"
    profile_path = directory / "glompo-profile.json"
    profile = reader.json(profile_path) if profile_path.is_file() else {}
    profile = profile if isinstance(profile, dict) else {}
    context = dict(
        profile_source=reader.relative(profile_path),
        engine_identity=profile.get("engine", "unverified"),
        objective_definition=profile.get("objective", "unverified"),
        dataset_scope=profile.get("dataset_scope", "unverified"),
        optimizer_version=profile.get("optimizer_version"),
        parameter_space=profile.get("parameter_space", "unverified"),
    )
    if profile.get("profile") != "ams_params":
        reader.result.warnings.append("Benchmark or unknown profile; not validated ReaxFF training")
    if history.is_file() and metadata.is_file() and native.is_file():
        meta = reader.json(metadata)
        payload = reader.read(history)
        native_bytes = reader.read(native)
        valid = isinstance(meta, dict) and meta.get("schema_version") == 1 and meta.get("source") == native.name
        valid = valid and meta.get("history_sha256") == hashlib.sha256(payload).hexdigest()
        valid = valid and meta.get("source_sha256") == hashlib.sha256(native_bytes).hexdigest()
        rows = []
        try:
            if not valid or (payload and not payload.endswith(b"\n")):
                raise ValueError("stale hash or incomplete export")
            rows = [json.loads(line) for line in payload.splitlines()]
            seen = set()
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("evaluation must be an object")
                key = (row["optimizer_id"], row["call_id"])
                if (row.get("schema_version") != 1 or type(key[1]) is not int or key[1] < 1
                        or not isinstance(key[0], str) or key in seen
                        or not isinstance(row.get("parameters"), list)
                        or row.get("source") != native.name):
                    raise ValueError("invalid or duplicate evaluation")
                seen.add(key)
            if meta.get("count") != len(rows):
                raise ValueError("export count mismatch")
        except (ValueError, TypeError, KeyError) as error:
            reader.result.warnings.append(f"History rejected: {error}; re-export after the manager stops")
            rows = []
        best = None
        initial = profile.get("initial_parameters_scaled")
        initial_verified = False
        for row in sorted(rows, key=lambda item: (item["call_id"], item["optimizer_id"])):
            score = finite(row.get("score"))
            is_initial = (
                profile.get("initial_candidate_policy") == "exact_first_evaluation"
                and profile.get("parameter_space") == "scaled_0_1"
                and row["call_id"] == 1 and isinstance(initial, list) and bool(initial)
                and all(finite(value) is not None for value in initial)
                and row["parameters"] == initial
            )
            initial_verified = initial_verified or is_initial
            if score is not None:
                best = score if best is None else min(best, score)
            reader.result.progress.append(progress(
                reader.relative(history), [row["optimizer_id"], row["call_id"]], score,
                native_step=row["call_id"], step_kind="evaluation", best_score=best,
                candidate_id=f"{row['optimizer_id']}:call:{row['call_id']}",
                optimizer_id=row["optimizer_id"], parameters=row["parameters"],
                is_initial_candidate=is_initial,
                record_kind="initial_parameter_evaluation" if is_initial else "evaluation",
                native_source=reader.relative(native), source_dataset=row.get("source_dataset"),
                source_row=row.get("source_row"), raw_score=row.get("raw_score"),
                status="ok" if score is not None else "nonfinite", **context,
            ))
        if rows and profile.get("profile") == "ams_params" and not initial_verified:
            reader.result.warnings.append(
                "Initial parameter evaluation is unverified; the first score must not be labeled as the original-field baseline"
            )
    elif native.is_file():
        reader.result.warnings.append("Export native HDF5 with GloMPO reaxkit_runner.py --export-native after the job stops")
    for name in ("glompo_log.h5", "glompo_manager_log.yml", "trajectories.png", "best/candidate.json"):
        reader.artifact(directory / name)
    reader.artifact(directory / "best" / "ffield", "parameter_only")
    reader.artifact(directory / "ffield_best", "parameter_only")
    reader.result.warnings.append("No labeled reference/prediction export; parity and cross-optimizer comparability remain unverified")
    return reader.result
