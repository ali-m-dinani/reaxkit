import sys
from pathlib import Path

import pytest

from reaxkit.optimization.config import load_config
from reaxkit.optimization.launching import control_text, expand_command, parameters, validate_jax_data
from reaxkit.optimization.models import OptimizationError
from reaxkit.optimization.orchestrator import collect, prepare
from reaxkit.optimization.storage import read_json
from reaxkit.optimization.worker import run_worker
from .test_configuration import edit_config, settings_file


def native_config(path, optimizer, settings=None, runtime=None, control=False):
    root = path.parent
    (root / "params").write_text("2 1 1 0.1 2 1 ! interval and reversed bounds\n")
    (root / "geo").write_text("DESCRP molecule\nEND\n")
    (root / "trainset.in").write_text("ENERGY\n1 + molecule/1 0\nENDENERGY\n")
    (root / "control").write_text("      1 imetho\n     17 mcffit\n")
    (root / "jax-root").mkdir(exist_ok=True)
    (root / "md-root").mkdir(exist_ok=True)
    options = dict(settings or {})
    if optimizer == "jax_soppi":
        options.update(jax_reaxff_root=str(root / "jax-root"), jax_md_root=str(root / "md-root"))

    def update(data):
        data["optimizers"] = [optimizer]
        data["optimizer_settings"] = {optimizer: {
            "runtime": {"mode": "native", "command": [sys.executable], **(runtime or {})},
            "settings": options,
        }}
        if control:
            data["inputs"]["control"] = "control"
    edit_config(path, update)
    return path


def first_run(config):
    root = prepare(config)
    directory = root / read_json(root / "campaign.json")["runs"][0]
    return directory, read_json(directory / "run-config.json")


@pytest.mark.parametrize("order,text,expected", [
    ("max_min", "1 2 0 1D-2 4 -3 ! retained scientific meaning\n", (-3, 4)),
    ("min_max", "1 2 0 0.1 -5 2\n", (-5, 2)),
])
def test_bound_order_and_fourth_column(tmp_path, order, text, expected):
    path = tmp_path / "params"
    path.write_text(text)
    rows = parameters(path, order)
    assert rows[0][4:] == expected
    assert rows[0][3] in (0.01, 0.1)


@pytest.mark.parametrize("text", [
    "1 1 0 .1 4 2", "1 1 0 .1 nan 2", "1 1 0 .1 2 2",
    "1 1 0 .1 1 2\n1 1 0 .1 1 2", "1 1 0 .1 1 2 extra", "",
])
def test_bad_parameter_rows_fail_before_staging(tmp_path, text):
    path = tmp_path / "params"
    path.write_text(text)
    with pytest.raises(OptimizationError):
        parameters(path, "min_max")


def test_control_precedence_and_fixed_width(tmp_path):
    path = tmp_path / "control"
    path.write_text("17 mcffit\n2 mcbeta\n1 imetho\n")
    settings = {"mcffit": 5, "mcbeta": 0, "mcstep": 1, "replic": None}
    rendered = control_text(path, settings, ["mcffit"], fixed_width=True)
    values = {line[8:14].strip(): float(line[:8]) for line in rendered.splitlines()}
    assert values == {"mcffit": 5, "mcbeta": 2, "mcstep": 1, "imetho": 1}
    path.write_text("1 mcffit\n2 mcffit\n")
    with pytest.raises(OptimizationError, match="duplicate"):
        control_text(path, settings, [])


@pytest.mark.parametrize("optimizer,iopt", [("ams_mcff", "4"), ("ams_cmaes", "7")])
def test_ams_native_staging_and_control_override(settings_file, optimizer, iopt):
    native_config(settings_file, optimizer, {"mcffit": 2}, control=True)
    directory, request = first_run(load_config(settings_file))
    assert (directory / "native/iopt").read_text().strip() == iopt
    assert (directory / "native/params").read_text().split()[-2:] == ["1", "2"]
    assert (directory / "native/control").read_text().count("mcffit") == 1
    assert request["launch"]["seed_forwarded"] is None
    assert request["launch"]["commands"][0]["argv"] == [sys.executable]


def test_cma_restart_is_copied_and_checksummed(settings_file):
    restart = settings_file.parent / "resume.txt"
    restart.write_text("native last-improvement checkpoint")
    native_config(settings_file, "ams_cmaes", {"restart": str(restart)}, control=True)
    directory, request = first_run(load_config(settings_file))
    assert (directory / "native/cmaes_restart.txt").read_text() == restart.read_text()
    assert "native/cmaes_restart.txt" in request["staged_hashes"]


def test_soppi_native_flags_and_seed_limit(settings_file):
    native_config(settings_file, "jax_soppi", {"max_rows": 1, "parsca": 0.5}, control=True)
    directory, request = first_run(load_config(settings_file))
    command = request["launch"]["commands"][0]["argv"]
    assert command[command.index("--max-rows") + 1] == "1"
    assert "--control" in command and "--bounds-order" in command
    assert "--seed" not in command
    assert request["contract"] == "reaxkit-native-launch-v1"
    assert (directory / "native/inputs/params").read_text().split()[-2:] == ["1", "2"]


def test_jax_evaluation_report_and_independent_seed_arguments(settings_file):
    native_config(settings_file, "jax_reaxff", {"num_trials": 0, "machine_report": True})
    root = prepare(load_config(settings_file))
    requests = [read_json(root / relative / "run-config.json") for relative in read_json(root / "campaign.json")["runs"]]
    seeds = []
    for request in requests:
        command = request["launch"]["commands"][0]["argv"]
        seeds.append(command[command.index("--seed") + 1])
        assert command[command.index("--machine_report") + 1].endswith("machine_report.json")
        assert command[command.index("--use_valid") + 1] == "False"
    assert len(set(seeds)) == 2


def test_jax_machine_report_wrong_mode_rejected(settings_file):
    native_config(settings_file, "jax_reaxff", {"num_trials": 1, "machine_report": True})
    with pytest.raises(OptimizationError, match="evaluation-only"):
        load_config(settings_file)


@pytest.mark.parametrize("training", [
    "STRESS\n1 2\nENDSTRESS\n",
    "ENERGY\n1 + absent/1 0\nENDENERGY\n",
    "GEOMETRY\nmolecule 1 0\nENDGEOMETRY\n",
    "ENERGY\n0 + molecule/1 0\nENDENERGY\n",
    "ENERGY\n1 + molecule/0 0\nENDENERGY\n",
    "FORCES\nmolecule 1 1 1 2\nENDFORCES\n",
    "ENERGY\n1 + molecule/1 0\n",
])
def test_jax_rejects_silent_data_loss(settings_file, training):
    native_config(settings_file, "jax_reaxff")
    (settings_file.parent / "trainset.in").write_text(training)
    with pytest.raises(OptimizationError):
        load_config(settings_file)


def test_jax_supported_vector_force_records(settings_file):
    native_config(settings_file, "jax_reaxff")
    trainset = settings_file.parent / "trainset.in"
    trainset.write_text("FORCES\nmolecule 1 1 .1 .2 .3\nENDFORCES\n")
    validate_jax_data(trainset, settings_file.parent / "geo")


@pytest.mark.parametrize("optimizer", ["indeedopt", "ezff", "pibo"])
def test_native_unsupported_workflows_never_silently_substitute(settings_file, optimizer):
    native_config(settings_file, optimizer)
    with pytest.raises(OptimizationError, match="external runner"):
        load_config(settings_file)


def test_pibo_native_cobalt_subset_is_explicit(settings_file):
    native_config(settings_file, "pibo", {"driver": "cobalt", "budget": 2})
    with pytest.raises(OptimizationError, match="acknowledge"):
        load_config(settings_file)
    edit_config(settings_file, lambda data: data["optimizer_settings"]["pibo"]["settings"].update(acknowledge_cobalt_subset=True))
    (settings_file.parent / "trainset.in").write_text("# General Co_clusters\n1 + a/1 - b/1 0\n")
    directory, request = first_run(load_config(settings_file))
    assert request["launch"]["dataset_scope"].startswith("Co_clusters")
    assert (directory / "native/inputs/ffield_lit").exists()
    command = request["launch"]["commands"][0]["argv"]
    assert command[command.index("--budget") + 1] == "2"
    assert "--seed" in command


def test_legacy_units_and_required_random_input(settings_file):
    native_config(settings_file, "standalone_soppi", control=True)
    with pytest.raises(OptimizationError, match="random_file"):
        load_config(settings_file)
    random_file = settings_file.parent / "random"
    random_file.write_text("12345\n")
    edit_config(settings_file, lambda data: data["optimizer_settings"]["standalone_soppi"]["settings"].update(random_file=str(random_file)))
    directory, request = first_run(load_config(settings_file))
    assert (directory / "native/fort.20").read_text() == "  1\n"
    assert (directory / "native/fort.21").read_text().split()[-2:] == ["2", "1"]
    assert (directory / "native/fort.35").read_text() == "12345\n"
    assert (directory / "native/fort.4").exists()
    assert request["launch"]["seed_forwarded"] is None


def test_ogolem_external_bridge_plan_and_custom_evaluator(settings_file):
    source = settings_file.parent / "ogolem"
    (source / "tools").mkdir(parents=True)
    (source / "tools/reaxff_bridge.py").write_text("fixture")
    native_config(settings_file, "ogolem", {"backend": "custom", "evaluator_command": ["engine-runner"], "include_initial": True},
                  {"source": str(source), "bridge_python": sys.executable})
    directory, request = first_run(load_config(settings_file))
    steps = request["launch"]["commands"]
    assert len(steps) == 2 and steps[0]["phase"] == "prepare"
    assert "--seed" in steps[0]["argv"]
    assert steps[1]["argv"][-3:] == ["--adaptive", "fit.ogo", "1"]
    assert request["settings"]["engine_python"] is None
    assert request["runtime"]["bridge_python"] == sys.executable
    assert request["launch"]["seed_forwarded"] is None
    from reaxkit.optimization.adapters.ogolem.launch import after_prepare
    bundle = Path(request["launch"]["results_directory"])
    bundle.mkdir()
    (bundle / "evaluator.json").write_text('{"command":[]}')
    after_prepare(request)
    assert read_json(bundle / "evaluator.json")["command"] == ["engine-runner"]
    (bundle / "evaluations/candidate_test").mkdir(parents=True)
    (bundle / "evaluations/candidate_test/fitness.json").write_text('{"fitness":1.5}')
    assert collect(directory.parents[2])[0]["progress_count"] == 1


def test_native_worker_forwards_argv_without_shell(settings_file, monkeypatch):
    recorder = settings_file.parent / "recorder.py"
    recorder.write_text("import json,pathlib,sys\npathlib.Path('args.json').write_text(json.dumps(sys.argv[1:]))\nprint('Trial-1 is starting...\\nIteration: 0\\nTrue total loss: 8.25', flush=True)\n")
    native_config(settings_file, "jax_reaxff", {"num_steps": 1}, {"command": [sys.executable, str(recorder)]})
    directory, request = first_run(load_config(settings_file))
    monkeypatch.setattr("reaxkit.optimization.worker.require_allocation", lambda: None)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-only")
    assert run_worker(directory) == 0
    arguments = read_json(directory / "native/args.json")
    assert arguments == request["launch"]["commands"][0]["argv"][2:]
    state = read_json(directory / "worker.json")
    assert state["seed_forwarded"] == request["seed"]
    assert state["last_conversion"]["progress_count"] == 1


def test_executable_environment_expansion_is_not_a_shell():
    assert expand_command(["${ENGINE_HOME}/reaxff", "$literal"], {"ENGINE_HOME": "/path with spaces"}) == ["/path with spaces/reaxff", "$literal"]
    with pytest.raises(OptimizationError, match="unset"):
        expand_command(["${MISSING}/runner"], {})


def test_native_results_can_be_collected_after_copying_campaign(settings_file):
    import shutil
    native_config(settings_file, "jax_reaxff")
    directory, request = first_run(load_config(settings_file))
    (directory / "native/stdout.log").write_text("Trial-1 is starting...\nIteration: 0\nTrue total loss: 5\n")
    root = directory.parents[2]
    copied = settings_file.parent / "downloaded campaign"
    shutil.copytree(root, copied)
    assert collect(copied)[0]["progress_count"] == 1
