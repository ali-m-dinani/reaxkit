import json
from argparse import Namespace

from reaxkit.core.runtime.provenance import (
    effective_settings_from_args,
    json_safe,
    user_settings_from_args,
)


def test_provenance_redacts_nested_credentials_without_mutating_arguments():
    args = Namespace(
        api_key="test-api-credential",
        elements=["Mg", "Te", "O"],
        nested=[{"MP_API_KEY": "test-nested-credential", "password": "test-password"}],
        access_token=None,
        _run="internal",
    )

    for settings in (effective_settings_from_args(args), user_settings_from_args(args), json_safe(args)):
        assert settings["api_key"] == "[REDACTED]"
        assert settings["nested"] == [{"MP_API_KEY": "[REDACTED]", "password": "[REDACTED]"}]
        assert settings["elements"] == ["Mg", "Te", "O"]
        assert "test-api-credential" not in json.dumps(settings)

    assert args.api_key == "test-api-credential"
    assert args.nested[0]["MP_API_KEY"] == "test-nested-credential"
    assert effective_settings_from_args(args)["access_token"] is None
    assert "access_token" not in user_settings_from_args(args)
    assert "_run" not in effective_settings_from_args(args)


def test_generator_settings_and_index_do_not_persist_credentials(tmp_path):
    from reaxkit.core.runtime.generator_runtime import persist_generator_metadata
    from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout

    layout = ReaxkitStorageLayout(project_root=tmp_path)
    layout.ensure_input_run_layout("test-run")
    output = layout.input_run_dir("test-run") / "trainset.in"
    output.write_text("test output", encoding="utf-8")
    args = Namespace(run_id="test-run", project_root=str(tmp_path), api_key="fake-credential")

    settings_path = persist_generator_metadata(
        args,
        command="trainset heatfo",
        output_path=output,
        layout=layout,
        extra={"authentication": {"api-key": "fake-credential"}},
    )

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    for section in ("args", "effective_settings", "user_settings"):
        assert settings[section]["api_key"] == "[REDACTED]"
    assert settings["extra"]["authentication"]["api-key"] == "[REDACTED]"
    for artifact in tmp_path.rglob("*.json"):
        assert "fake-credential" not in artifact.read_text(encoding="utf-8")
    assert args.api_key == "fake-credential"
