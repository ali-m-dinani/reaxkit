"""Explicit unit mapping for the audited legacy ffopt executable."""

from pathlib import Path

from ...launching import control_text, native_plan, parameters, stage_common
from ...models import OptimizationError
from ...storage import atomic_text


def validate(config, section):
    if not config["inputs"].get("control"):
        raise OptimizationError("standalone_soppi requires a compatible legacy control file")
    if not section["settings"].get("random_file"):
        raise OptimizationError("standalone_soppi requires random_file (native fort.35 input), even for deterministic search")
    control_text(config["inputs"]["control"], {key: value for key, value in section["settings"].items() if key != "random_file"},
                 section["explicit_settings"], fixed_width=True)
    for row in parameters(config["inputs"]["params"], config["inputs"]["params_bounds_order"]):
        if abs(row[4]) < 0.001 or abs(row[5]) < 0.001:
            raise OptimizationError("standalone_soppi interprets bounds near zero as unbounded; use an external runner for an explicitly reviewed legacy setup")


def prepare(request):
    import shutil
    native = Path(request["native_directory"])
    stage_common(request, native, bounds_order="max_min", names={"ffield": "fort.4", "geo": "fort.3", "params": "fort.21"})
    settings = {key: value for key, value in request["settings"].items() if key != "random_file"}
    atomic_text(native / "control", control_text(request["inputs"]["control"], settings, request["explicit_settings"], fixed_width=True))
    atomic_text(native / "fort.20", "  1\n")
    atomic_text(native / "fort.22", "  0\n")
    atomic_text(native / "fort.23", "")
    shutil.copyfile(request["settings"]["random_file"], native / "fort.35")
    return native_plan(request, list(request["runtime"]["command"]), warnings=[
        "No linked parameters are staged (empty units 22/23). Use external_runner for linked-parameter workflows.",
        "Legacy source headers restrict redistribution; provide your own compatible executable.",
        "The native executable clamps steps smaller than 1e-5; the original fourth-column values are preserved.",
    ])
