import copy

import pytest

from reaxkit.optimization.config import load_config
from reaxkit.optimization.models import OptimizationError
from reaxkit.optimization.registry import JOB_ABBREVIATIONS, OPTIMIZERS
from reaxkit.optimization.orchestrator import prepare, submit
from reaxkit.optimization.schedulers.render import render_job, resolve_resources
from reaxkit.optimization.schedulers.slurm import JOB_TEMPLATE, validate_job
from reaxkit.optimization.storage import read_json
from .test_configuration import settings_file, edit_config
from .test_cli_execution import invoke
from .test_lifecycle import FakeSlurm


def test_multiple_methods_replicas_get_distinct_resource_jobs(settings_file, monkeypatch):
    def update(data):
        data["optimizers"] = ["jax_soppi", "flocky"]
        data["resources"] = dict(nodes=1, ntasks=1, cpus_per_task=8, time="01:15:00")
        data["optimizer_settings"]["flocky"] = copy.deepcopy(data["optimizer_settings"]["jax_soppi"])
        data["optimizer_settings"]["flocky"]["settings"] = {}
    edit_config(settings_file, update)
    monkeypatch.setattr("reaxkit.optimization.orchestrator.platform.system", lambda: "Linux")
    root = prepare(load_config(settings_file))
    runs = read_json(root / "campaign.json")["runs"]
    assert len(runs) == 4
    for relative in runs:
        job = root / relative / "job/submit.sh"
        text = validate_job(job)
        serial = "flocky" in relative
        assert f"#SBATCH --cpus-per-task={1 if serial else 8}" in text
        assert "#SBATCH --time=01:15:00" in text
        assert text.index("#SBATCH --time") < text.index("export REAXKIT_OPTIMIZER=")
        run = read_json(root / relative / "run.json")
        name = read_json(root / relative / "job/submission.json")["name"]
        assert name.startswith(f"{JOB_ABBREVIATIONS[run['optimizer']]}-r{run['replica']:02d}-")
        assert f"#SBATCH --job-name={name}" in text
        assert text.count("#SBATCH --job-name=") == 1
    from types import SimpleNamespace
    monkeypatch.setattr("reaxkit.optimization.orchestrator.os", SimpleNamespace(name="posix"))
    scheduler = FakeSlurm()
    assert len(submit(root, scheduler)) == 4
    assert len(scheduler.calls) == 4
    assert len({call[3] for call in scheduler.calls}) == 4
    for call in scheduler.calls:
        assert f"#SBATCH --job-name={call[3]}" in call[0].read_text()
    submit(root, scheduler)
    assert len(scheduler.calls) == 4


def test_job_cli_override(settings_file):
    job = settings_file.parent / "alternate.sh"
    job.write_text(JOB_TEMPLATE)
    edit_config(settings_file, lambda data: data.update(job="missing.sh"))
    outcome = invoke("--settings", settings_file, "--job", job, "--dry-run")
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr


def test_generic_template_and_job_name_aliases():
    assert "#SBATCH --account=ACCOUNT" in JOB_TEMPLATE
    assert "#SBATCH --partition=PARTITION" in JOB_TEMPLATE
    for directive in ("--job-name=old", "-J old", "-Jold"):
        template = JOB_TEMPLATE.replace("--job-name=reaxkit-fit", directive)
        rendered = render_job(template, "ogolem", {}, "ogo-r02-example")
        assert "#SBATCH --job-name=ogo-r02-example" in rendered
        assert directive not in rendered


def test_unique_three_letter_job_abbreviations():
    assert set(JOB_ABBREVIATIONS) == set(OPTIMIZERS)
    assert len(set(JOB_ABBREVIATIONS.values())) == len(OPTIMIZERS)
    for optimizer, abbreviation in JOB_ABBREVIATIONS.items():
        assert len(abbreviation) == 3 and abbreviation.isalpha()
        assert f"#SBATCH --job-name={abbreviation}\n" in render_job(JOB_TEMPLATE, optimizer, {})


@pytest.mark.parametrize("override", [None, "alternate.control"])
def test_per_method_control_validation_staging_and_hashes(settings_file, monkeypatch, override):
    shared = settings_file.parent / "control"
    shared.write_text("1 igeopt\n")
    alternate = settings_file.parent / "alternate.control"
    alternate.write_text("0 igeopt\n")
    def update(data):
        data["inputs"]["control"] = "control"
        data["optimizers"] = ["jax_soppi", "jax_reaxff"]
        data["optimizer_settings"]["jax_reaxff"] = {
            "runtime": {"command": ["runner", "{run_config}"]},
            "inputs": {"control": override},
        }
    edit_config(settings_file, update)
    observed = {}
    def validate(config, optimizer, section):
        observed[optimizer] = config["inputs"]["control"]
    monkeypatch.setattr("reaxkit.optimization.launching.validate_native", validate)
    config = load_config(settings_file)
    assert observed["jax_soppi"] == str(shared)
    assert observed["jax_reaxff"] == (str(alternate) if override else None)
    root = prepare(config)
    from reaxkit.optimization.storage import digest
    for relative in read_json(root / "campaign.json")["runs"]:
        directory = root / relative
        request = read_json(directory / "run-config.json")
        metadata = read_json(directory / "run.json")
        expected = shared if request["optimizer"] == "jax_soppi" else alternate if override else None
        assert (directory / "input/control").exists() == (expected is not None)
        assert ("control" in request["inputs"]) == (expected is not None)
        assert metadata["input_hashes"] == request["input_hashes"]
        assert metadata["input_hashes"].get("control") == (digest(expected) if expected else None)
    reloaded = load_config(root / "settings.resolved.yaml")
    assert reloaded["optimizer_settings"]["jax_reaxff"]["inputs"] == config["optimizer_settings"]["jax_reaxff"]["inputs"]


def test_control_override_must_exist_and_dataset_override_rejected(settings_file):
    edit_config(settings_file, lambda data: data["optimizer_settings"]["jax_soppi"].update(inputs={"control": "missing"}))
    with pytest.raises(OptimizationError, match="not found"):
        load_config(settings_file)
    edit_config(settings_file, lambda data: data["optimizer_settings"]["jax_soppi"].update(inputs={"geo": "geo"}))
    with pytest.raises(OptimizationError, match="unknown keys"):
        load_config(settings_file)


def test_resource_aliases_and_serial_constraints():
    assert resolve_resources("ams_mcff", {"nodes": 2}, {"ntasks": 16, "ntasks_per_node": 8})["ntasks"] == 16
    template = JOB_TEMPLATE.replace("#SBATCH --cpus-per-task=1", "#SBATCH -c 16")
    rendered = render_job(template, "jax_soppi", {"cpus_per_task": 4})
    assert "#SBATCH -c" not in rendered
    assert rendered.count("--cpus-per-task") == 1
    assert render_job(template, "flocky", resolve_resources("flocky", {}, {})).count("--ntasks=") == 1
    with pytest.raises(OptimizationError, match="serial"):
        resolve_resources("flocky", {}, {"ntasks": 2})
    with pytest.raises(OptimizationError, match="one Slurm option"):
        render_job(template.replace("-c 16", "-c 16 -n 4"), "jax_soppi", {"cpus_per_task": 4})
    with pytest.raises(OptimizationError, match="precede"):
        render_job(template.replace("set -e", "set -e\n#SBATCH --mem=1G"), "jax_soppi", {})


@pytest.mark.parametrize("resources", [{"time": "01:75:00"}, {"time": "00:00:00"},
                                     {"mem": "1G\nmalicious"}, {"nodes": True}])
def test_invalid_resources(resources):
    with pytest.raises(OptimizationError):
        resolve_resources("jax_soppi", {}, resources)
