import subprocess
import sys

import yaml

from .test_configuration import settings_file
from .test_launching import native_config


def invoke(*arguments):
    command = [sys.executable, "-c", "from reaxkit.cli_startup import main; raise SystemExit(main())",
               "optimize-ffield", *map(str, arguments)]
    return subprocess.run(command, capture_output=True, text=True, timeout=60)


def test_full_cli_generates_template_and_native_dry_run(settings_file):
    template = settings_file.parent / "generated.yaml"
    result = invoke("--generate-template", template)
    assert result.returncode == 0, result.stdout + result.stderr
    data = yaml.safe_load(template.read_text())
    assert data["optimizer_settings"]["jax_soppi"]["runtime"]["mode"] == "native"
    native_config(settings_file, "jax_reaxff", {"num_steps": 1})
    result = invoke("--settings", settings_file, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    root = settings_file.parent / "campaign with spaces"
    assert (root / "campaign.json").is_file()
    assert not list(root.rglob("worker.json"))
