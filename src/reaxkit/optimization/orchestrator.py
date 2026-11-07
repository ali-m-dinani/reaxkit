"""Prepare independent run directories and manage durable submission intent."""

from __future__ import annotations

import hashlib
import copy
import os
import platform
import shlex
import shutil
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from .conversion import collect_native
from .launching import prepare_launch, staged_hashes, verify_staged
from .models import OptimizationError, identity
from .registry import specification
from .schedulers.slurm import Slurm, SubmissionUnknown
from .storage import atomic_text, digest, lock, now, read_json, write_json


def prepare(config):
    root = Path(config["output"])
    for optimizer in config["optimizers"]:
        prepared = config["optimizer_settings"][optimizer].get("prepared_inputs")
        if prepared and (root == Path(prepared) or Path(prepared) in root.parents):
            raise OptimizationError("output must not be inside prepared_inputs")
    root.mkdir(parents=True, exist_ok=False)
    campaign_id = uuid.uuid4().hex
    manifest = {"schema_version": 1, "campaign_id": campaign_id, "created_at": now(),
                "prepared_platform": platform.system(), "state": "preparing", "runs": []}
    write_json(root / "campaign.json", manifest)
    atomic_text(root / "settings.original.yaml", config["_original"])
    resolved = {key: value for key, value in config.items() if not key.startswith("_")}
    atomic_text(root / "settings.resolved.yaml", yaml.safe_dump(resolved, sort_keys=False))
    input_hashes = {key: digest(Path(value)) for key, value in config["inputs"].items()
                    if key != "params_bounds_order" and value}
    dataset_fingerprint = identity({key: value for key, value in input_hashes.items() if key in ("trainset", "geo")})
    for optimizer in config["optimizers"]:
        spec = specification(optimizer)
        section = config["optimizer_settings"][optimizer]
        for replica in range(1, config["replicas"] + 1):
            relative = f"runs/{optimizer}/replica-{replica:04d}"
            directory = root / relative
            for child in ("input", "native", "job"):
                (directory / child).mkdir(parents=True)
            input_paths = {}
            for key, source in config["inputs"].items():
                if key == "params_bounds_order" or not source:
                    continue
                target = directory / "input" / ("trainset.in" if key == "trainset" else key)
                shutil.copyfile(source, target)
                input_paths[key] = str(target)
            if section.get("prepared_inputs"):
                if any(path.is_symlink() for path in Path(section["prepared_inputs"]).rglob("*")):
                    raise OptimizationError("prepared_inputs must contain regular files/directories, not symlinks")
                shutil.copytree(section["prepared_inputs"], directory / "native", dirs_exist_ok=True)
            staged_job = directory / "job" / "submit.sh"
            atomic_text(staged_job, Path(section["job"]).read_text(encoding="utf-8"))
            seed = None if config["seed"] is None else int.from_bytes(
                hashlib.sha256(f"{config['seed']}:{optimizer}:{replica}".encode()).digest()[:4], "big"
            ) % 2147483647
            run_id = f"rk-{campaign_id[:12]}-{optimizer}-{replica}"
            run = {
                "schema_version": 1, "run_id": run_id, "campaign_id": campaign_id,
                "optimizer": optimizer, "replica": replica, "seed": seed,
                "seed_support": spec["seed"], "seed_applied": None,
                "engine": spec["engine"], "objective": spec["objective"],
                "optimizer_version": section["runtime"].get("version"),
                "engine_version": section["runtime"].get("engine_version"),
                "adapter_revision": spec["revision"], "cluster_validated": False,
                "input_hashes": input_hashes, "dataset_fingerprint": dataset_fingerprint,
                "comparison_fingerprint": identity(input_hashes, optimizer, spec["engine"], spec["objective"], section["settings"], section["runtime"]),
                "comparison_status": "unverified",
                "created_at": now(), "native_directory": str(directory / "native"),
                "job_sha256": digest(staged_job),
                "collection_interval_seconds": config["collection_interval_seconds"],
            }
            request = {
                "schema_version": 1, "contract": "reaxkit-external-runner-v1",
                "run_id": run_id, "optimizer": optimizer, "replica": replica,
                "seed": seed, "seed_support": spec["seed"], "inputs": input_paths,
                "input_hashes": input_hashes, "params_bounds_order": config["inputs"]["params_bounds_order"],
                "settings": copy.deepcopy(section["settings"]), "explicit_settings": section["explicit_settings"],
                "runtime": section["runtime"], "conversion": section["conversion"],
                "native_directory": run["native_directory"],
                "external_file_hashes": {},
            }
            for key, rule in spec["settings"].items():
                value = request["settings"][key]
                if rule["type"] == "path" and value and Path(value).is_file():
                    if section["runtime"]["mode"] == "native" and key in (
                        "koppel2", "valid_file", "valid_geo_file", "restart", "random_file", "ams_job_file",
                    ):
                        target = directory / "input" / "settings" / key / Path(value).name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(value, target)
                        request["settings"][key] = str(target)
                    else:
                        request["external_file_hashes"][value] = digest(Path(value))
            request["launch"] = prepare_launch(request)
            if section["runtime"]["mode"] == "native":
                request["contract"] = "reaxkit-native-launch-v1"
            request["staged_hashes"] = staged_hashes(directory)
            run["launch_mode"] = section["runtime"]["mode"]
            run["native_results_directory"] = request["launch"]["results_directory"]
            run["native_results_relative"] = Path(run["native_results_directory"]).relative_to(directory).as_posix()
            run["launch_warnings"] = request["launch"]["warnings"]
            run["dataset_scope"] = request["launch"].get("dataset_scope", "native_input_semantics")
            for key in ("engine", "objective"):
                run[key] = request["launch"].get(key, run[key])
            run["comparison_fingerprint"] = identity(run["comparison_fingerprint"], request["staged_hashes"],
                                                     request["external_file_hashes"], run["dataset_scope"],
                                                     request["params_bounds_order"], run["job_sha256"],
                                                     run["engine"], run["objective"])
            write_json(directory / "run-config.json", request)
            run["config_sha256"] = digest(directory / "run-config.json")
            worker_command = [sys.executable, "-m", "reaxkit.optimization.worker", str(directory)]
            atomic_text(directory / "job" / "worker.sh", "#!/bin/bash\nset -e\nexec " + shlex.join(worker_command) + "\n")
            run["worker_sha256"] = digest(directory / "job" / "worker.sh")
            write_json(directory / "run.json", run)
            write_json(directory / "job" / "submission.json", {"state": "staged", "name": run_id})
            manifest["runs"].append(relative)
            write_json(root / "campaign.json", manifest)
    manifest["state"] = "prepared"
    write_json(root / "campaign.json", manifest)
    return root


def campaign_runs(root):
    root = Path(root).resolve()
    manifest = read_json(root / "campaign.json")
    if manifest.get("schema_version") != 1:
        raise OptimizationError("Unsupported campaign schema")
    runs = []
    for relative in manifest["runs"]:
        directory = (root / relative).resolve()
        if root not in directory.parents:
            raise OptimizationError("Run path escapes campaign directory")
        runs.append(directory)
    return manifest, runs


def submit(root, scheduler=None):
    if os.name != "posix" or platform.system() != "Linux":
        raise OptimizationError("Submit on Roar Linux; local validation/conversion does not require Slurm")
    root = Path(root).resolve()
    scheduler = scheduler or Slurm()
    with lock(root, ".submission.lock"):
        manifest, runs = campaign_runs(root)
        if manifest["prepared_platform"] != "Linux":
            raise OptimizationError("Prepare a fresh campaign on Linux so paths/interpreter are valid there")
        if manifest["state"] == "preparing":
            raise OptimizationError("Campaign preparation is incomplete")
        outcomes = []
        for directory in runs:
            ledger_path = directory / "job" / "submission.json"
            ledger = read_json(ledger_path)
            if ledger["state"] != "staged":
                outcomes.append({"run": str(directory), **ledger})
                continue
            run = read_json(directory / "run.json")
            if digest(directory / "run-config.json") != run["config_sha256"] or digest(directory / "job" / "submit.sh") != run["job_sha256"]:
                raise OptimizationError(f"{directory}: staged settings/job changed; prepare a new campaign")
            if digest(directory / "job" / "worker.sh") != run["worker_sha256"]:
                raise OptimizationError(f"{directory}: staged worker changed; prepare a new campaign")
            request = read_json(directory / "run-config.json")
            verify_staged(directory, request.get("staged_hashes", {}), request.get("external_file_hashes", {}))
            ledger.update(state="submitting", attempted_at=now())
            write_json(ledger_path, ledger)
            try:
                result = scheduler.submit(directory / "job" / "submit.sh", directory / "job" / "worker.sh",
                                          directory / "native", ledger["name"], directory / "job")
                ledger.update(result, state="submitted")
            except SubmissionUnknown as error:
                ledger.update(state="unknown", error=str(error))
            except (OSError, OptimizationError) as error:
                ledger.update(state="submission_failed", error=str(error))
            write_json(ledger_path, ledger)
            outcomes.append({"run": str(directory), **ledger})
            if ledger["state"] != "submitted":
                break
        manifest["state"] = "submission_attempted"
        write_json(root / "campaign.json", manifest)
        return outcomes


def status(root, scheduler=None):
    scheduler = scheduler or Slurm()
    _, runs = campaign_runs(root)
    ledgers = [read_json(directory / "job" / "submission.json") for directory in runs]
    states = scheduler.statuses([ledger["job_id"] for ledger in ledgers if ledger.get("job_id")])
    results = []
    for directory, ledger in zip(runs, ledgers):
        row = {"run": str(directory), "submission": ledger,
               "scheduler": states.get(ledger.get("job_id"), {"state": "UNKNOWN", "source": "no_job_id"})}
        if ledger["state"] in ("unknown", "submitting"):
            date = (datetime.fromisoformat(ledger["attempted_at"]) - timedelta(days=1)).date().isoformat()
            row["reconciliation_candidates"] = scheduler.find_submission(ledger["name"], date)
            row["reconciliation_note"] = "Read-only evidence; never resubmit an ambiguous job automatically"
        worker_file = directory / "worker.json"
        row["worker"] = read_json(worker_file) if worker_file.exists() else {"state": "not_reported"}
        results.append(row)
    return results


def collect_run(directory):
    directory = Path(directory)
    run = read_json(directory / "run.json")
    request = read_json(directory / "run-config.json")
    worker_file = directory / "worker.json"
    if worker_file.exists():
        run["seed_forwarded"] = read_json(worker_file).get("seed_forwarded")
    native = (directory / run.get("native_results_relative", "native")).resolve()
    if directory.resolve() not in native.parents:
        raise OptimizationError("Native results path escapes the run directory")
    return collect_native(run["optimizer"], native, directory / "normalized",
                          request["conversion"], metadata=run)


def collect(root):
    _, runs = campaign_runs(root)
    results = []
    for directory in runs:
        try:
            results.append({"run": str(directory), **collect_run(directory)})
        except (OptimizationError, OSError, ValueError) as error:
            results.append({"run": str(directory), "error": str(error)})
    return results
