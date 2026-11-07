"""Read-only evidence for a cluster smoke-test review, not scientific certification."""

from collections import Counter
from pathlib import Path

from .orchestrator import campaign_runs, status
from .storage import digest, read_json


def smoke_report(root, scheduler=None):
    root = Path(root).resolve()
    manifest, directories = campaign_runs(root)
    states = {row["run"]: row for row in status(root, scheduler)}
    counts = Counter()
    rows = []
    job_ids = []
    input_sets = []
    seeds = []
    for directory in directories:
        run = read_json(directory / "run.json")
        counts[run["optimizer"]] += 1
        input_sets.append(run["input_hashes"])
        if run["seed"] is not None:
            seeds.append(run["seed"])
        evidence = states[str(directory)]
        job_id = evidence["submission"].get("job_id")
        if job_id:
            job_ids.append(job_id)
        worker = evidence["worker"]
        checks = {
            "job_id_recorded": bool(job_id),
            "worker_job_matches_submission": bool(job_id) and worker.get("job_id") == job_id,
            "worker_completed": worker.get("state") == "completed" and worker.get("exit_code") == 0,
            "scheduler_completed": evidence["scheduler"]["state"] == "COMPLETED",
            "allocation_recorded": bool(worker.get("allocation")),
        }
        pointer = directory / "normalized/conversion.json"
        conversion = read_json(pointer) if pointer.is_file() else {}
        checks["progress_available"] = conversion.get("progress_count", 0) > 0
        artifacts = []
        if conversion.get("snapshot"):
            snapshot = (pointer.parent / conversion["snapshot"]).resolve()
            if pointer.parent not in snapshot.parents:
                raise ValueError("Snapshot path escapes normalized results")
            for item in read_json(snapshot / "artifacts.json"):
                native = (directory / run.get("native_results_relative", "native")).resolve()
                if directory not in native.parents:
                    raise ValueError("Native results path escapes the run directory")
                path = (native / item["path"]).resolve()
                matches = native in path.parents and path.is_file() and digest(path) == item["sha256"]
                artifacts.append({**item, "current_hash_matches_snapshot": matches})
        checks["artifacts_preserved"] = bool(artifacts) and all(item["current_hash_matches_snapshot"] for item in artifacts)
        rows.append({
            "run": str(directory), "optimizer": run["optimizer"], "replica": run["replica"],
            "checks": checks, "seed_requested": run["seed"], "seed_support": run["seed_support"],
            "seed_forwarded": worker.get("seed_forwarded"), "scheduler": evidence["scheduler"],
            "worker_state": worker.get("state"), "job_id": job_id,
            "allocation": worker.get("allocation"), "artifacts": artifacts,
            "conversion": conversion, "comparison_status": run["comparison_status"],
        })
    common_inputs = [{key: value for key, value in inputs.items() if key in ("ffield", "params", "trainset", "geo")}
                     for inputs in input_sets]
    checks = {
        "prepared_on_linux": manifest["prepared_platform"] == "Linux",
        "two_methods_two_replicas": len(counts) >= 2 and all(count >= 2 for count in counts.values()),
        "distinct_directories": len(set(directories)) == len(directories),
        "distinct_job_ids": len(job_ids) == len(directories) == len(set(job_ids)),
        "same_common_inputs": bool(common_inputs) and all(value == common_inputs[0] for value in common_inputs),
        "distinct_requested_seeds_when_present": len(seeds) == len(set(seeds)),
        "run_evidence_complete": bool(rows) and all(all(row["checks"].values()) for row in rows),
    }
    return {
        "schema_version": 1, "campaign": str(root), "checks": checks,
        "automated_checks_pass": all(checks.values()), "cluster_validated": False,
        "review_status": "manual_review_required", "runs": rows,
        "manual_acceptance_gates": [
            "Confirm these job IDs/allocations belong to the intended Roar installation; metadata alone cannot prove the site.",
            "Confirm each engine actually performed a tiny scientifically valid optimization, not only evaluation or a synthetic runner.",
            "Compare native and normalized scores, units, weights and candidate identities against original records.",
            "Inspect best-field artifacts and verify that they are complete, usable native force fields.",
            "Run a separately authorized interruption test; confirm final/offline collection and no duplicate resubmission.",
            "Record exact optimizer/engine/environment versions, validation evidence, and unavailable methods.",
        ],
    }
