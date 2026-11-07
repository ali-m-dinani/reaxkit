"""SCM standalone MCFF input/command mapping."""

from pathlib import Path

from ...launching import control_text, native_plan, stage_common
from ...models import OptimizationError
from ...storage import atomic_text


def validate(config, section):
    if not config["inputs"].get("control"):
        raise OptimizationError("AMS native launches require the user's compatible control file")
    settings = {key: value for key, value in section["settings"].items() if key != "restart"}
    control_text(config["inputs"]["control"], settings, section["explicit_settings"], fixed_width=True)


def prepare(request):
    native = Path(request["native_directory"])
    stage_common(request, native)
    settings = {key: value for key, value in request["settings"].items() if key != "restart"}
    atomic_text(native / "control", control_text(request["inputs"]["control"], settings, request["explicit_settings"], fixed_width=True))
    atomic_text(native / "iopt", "4\n" if request["optimizer"] == "ams_mcff" else "7\n")
    return native_plan(request, list(request["runtime"]["command"]), warnings=[
        "The params fourth column is retained unchanged; its native meaning is method-specific.",
        "AMS version/license and progress column mapping require installation-specific validation.",
    ])
