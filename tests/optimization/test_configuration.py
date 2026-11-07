import pytest
import yaml

from reaxkit.optimization.config import generate_template, load_config
from reaxkit.optimization.models import OptimizationError
from reaxkit.optimization.orchestrator import prepare
from reaxkit.optimization.registry import OPTIMIZERS, specification
from reaxkit.optimization.schedulers.slurm import JOB_TEMPLATE


@pytest.fixture
def settings_file(tmp_path):
    for name in ("ffield", "params", "trainset.in", "geo"):
        (tmp_path / name).write_text("synthetic input\n", encoding="utf-8")
    (tmp_path / "job.sh").write_text(JOB_TEMPLATE, encoding="utf-8")
    config = {
        "schema_version": 1, "replicas": 2, "optimizers": ["jax_soppi"],
        "seed": 42, "output": "campaign with spaces", "job": "job.sh",
        "inputs": {"ffield": "ffield", "params": "params", "trainset": "trainset.in",
                   "geo": "geo", "params_bounds_order": "max_min"},
        "optimizer_settings": {"jax_soppi": {
            "runtime": {"command": ["/path/to/runner", "{run_config}"]},
            "settings": {"max_rows": 1},
        }},
    }
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def edit_config(path, update):
    data = yaml.safe_load(path.read_text())
    update(data)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_template_contains_all_audited_settings_and_no_site_defaults(tmp_path):
    path = generate_template(tmp_path / "template.yaml")
    text = path.read_text()
    data = yaml.safe_load(text)
    assert list(data["optimizer_settings"]) == list(OPTIMIZERS)
    for optimizer in OPTIMIZERS:
        assert set(data["optimizer_settings"][optimizer]["settings"]) == set(specification(optimizer)["settings"])
    for site_value in ("acv13", "genoa", "ams/2023.103", "48"):
        assert site_value not in text
    with pytest.raises(FileExistsError):
        generate_template(path)


@pytest.mark.parametrize("key,value", [("replicas", 0), ("replicas", True), ("seed", float("nan")), ("unknown", 5)])
def test_invalid_shared_values_fail(settings_file, key, value):
    edit_config(settings_file, lambda data: data.update({key: value}))
    with pytest.raises(OptimizationError):
        load_config(settings_file)


def test_unknown_native_setting_and_duplicate_yaml_fail(settings_file):
    edit_config(settings_file, lambda data: data["optimizer_settings"]["jax_soppi"]["settings"].update(typo=5))
    with pytest.raises(OptimizationError, match="typo"):
        load_config(settings_file)
    settings_file.write_text("schema_version: 1\nschema_version: 1\n")
    with pytest.raises(OptimizationError, match="Duplicate"):
        load_config(settings_file)


def test_paths_and_independent_replica_staging(settings_file, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "replacement").write_text("override field")
    monkeypatch.chdir(elsewhere)
    config = load_config(settings_file, {"ffield": "replacement"})
    assert config["inputs"]["ffield"] == str(elsewhere / "replacement")
    assert config["inputs"]["geo"] == str(tmp_path / "geo")
    root = prepare(config)
    from reaxkit.optimization.storage import read_json
    manifest = read_json(root / "campaign.json")
    runs = [root / name for name in manifest["runs"]]
    assert len(runs) == 2
    seeds = {read_json(run / "run.json")["seed"] for run in runs}
    assert len(seeds) == 2
    assert read_json(runs[0] / "run.json")["seed_support"] == "not_applicable"
    assert read_json(runs[0] / "run-config.json")["params_bounds_order"] == "max_min"
    assert (runs[0] / "input" / "ffield").read_text() == "override field"
    assert "exec " in (runs[0] / "job" / "worker.sh").read_text()
    with pytest.raises(FileExistsError):
        prepare(config)


def test_missing_runner_and_native_command_rejected(settings_file):
    edit_config(settings_file, lambda data: data["optimizer_settings"]["jax_soppi"]["runtime"].update(command=["python", "-m", "soppi.cli"]))
    with pytest.raises(OptimizationError, match="run_config"):
        load_config(settings_file)


def test_resolved_yaml_can_be_validated_again(settings_file):
    config = load_config(settings_file)
    root = prepare(config)
    resolved = load_config(root / "settings.resolved.yaml")
    assert resolved["optimizer_settings"]["jax_soppi"]["explicit_settings"] == ["max_rows"]


def test_optimization_inventory_matches_committed_snapshot():
    import json
    from pathlib import Path
    from reaxkit.cli.inventory import parser_rows
    from reaxkit.cli.main import build_parser
    parser = build_parser("optimize-ffield")
    child = next(action.choices["optimize-ffield"] for action in parser._actions
                 if isinstance(getattr(action, "choices", None), dict) and "optimize-ffield" in action.choices)
    expected = json.loads((Path(__file__).resolve().parents[2] / "cli-help-inventory.json").read_text(encoding="utf-8"))
    expected = [row for row in expected if row["parser"] == "reaxkit optimize-ffield"]
    actual = sorted(parser_rows("reaxkit optimize-ffield", child), key=lambda row: row["flags"][0])
    assert actual == expected
    from reaxkit.cli.help_metadata import metadata_for_flag
    for row in actual:
        flag = next((value for value in row["flags"] if value.startswith("--")), row["flags"][0])
        assert (row["category"], row["visibility"]) == metadata_for_flag(flag, row["parser"])


def test_pibo_nested_replicas_and_force_term_rejected(settings_file):
    def configure(data):
        data["optimizers"] = ["pibo"]
        data["optimizer_settings"] = {"pibo": {
            "runtime": {"command": ["runner", "{run_config}"]}, "settings": {"replicates": 3},
        }}
    edit_config(settings_file, configure)
    with pytest.raises(OptimizationError):
        load_config(settings_file)
    edit_config(settings_file, lambda data: data["optimizer_settings"]["pibo"].update(settings={"w_forces": 1}))
    with pytest.raises(OptimizationError):
        load_config(settings_file)


def test_cli_registration_help_and_invalid_action_combinations(settings_file, capsys):
    from reaxkit.cli.main import build_parser
    from reaxkit.optimization.cli import run_main
    parser = build_parser("optimize-ffield")
    with pytest.raises(SystemExit) as result:
        parser.parse_args(["optimize-ffield", "--help"])
    assert result.value.code == 0
    assert "--generate-template" in capsys.readouterr().out
    args = parser.parse_args(["optimize-ffield", "--settings", str(settings_file), "--validate"])
    assert run_main(args) == 0
    assert not (settings_file.parent / "campaign with spaces").exists()
    args = parser.parse_args(["optimize-ffield", "--status", ".", "--dry-run"])
    with pytest.raises(OptimizationError, match="require --settings"):
        run_main(args)
