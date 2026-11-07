"""Launch the existing cobalt/LAMMPS driver without changing its objective."""

from pathlib import Path

from ...launching import native_plan, parameters, stage_common
from ...models import OptimizationError


def validate(config, section):
    settings = section["settings"]
    if settings["driver"] != "cobalt":
        raise OptimizationError("pibo mos2 uses its own VASP dataset, fixed parameter catalog and repository-relative working files; common-input isolated native runs are unsupported. Use an external runner in the PIBO project.")
    if not settings["acknowledge_cobalt_subset"]:
        raise OptimizationError("pibo cobalt selects only Co_clusters/Co2_cluster energy targets; set acknowledge_cobalt_subset: true to explicitly accept that native objective")
    unsupported = [key for key in section["explicit_settings"]
                   if key not in ("driver", "budget", "acknowledge_cobalt_subset") and settings[key] is not None]
    if unsupported:
        raise OptimizationError(f"pibo cobalt: settings belong to the MoS2 driver, not this CLI: {unsupported}; remove them")
    if config["inputs"].get("control"):
        raise OptimizationError("pibo cobalt does not read ReaxFF control; omit it")
    for section_id, entry, column, *_ in parameters(config["inputs"]["params"], config["inputs"]["params_bounds_order"]):
        if section_id not in (2, 3) or entry != 1 or column < 1 or column > (32 if section_id == 2 else 16):
            raise OptimizationError("pibo cobalt supports only first-row atom/bond parameters of its cobalt force field")
    training = Path(config["inputs"]["trainset"]).read_text(encoding="utf-8")
    if not any(f"# General {group}" in training for group in ("Co_clusters", "Co2_cluster")):
        raise OptimizationError("pibo cobalt: no native cobalt training-group marker found")


def prepare(request):
    inputs = Path(request["native_directory"]) / "inputs"
    stage_common(request, inputs, names={"ffield": "ffield_lit"})
    command = list(request["runtime"]["command"])
    command.extend(["--data-root", str(inputs), "--output", request["native_directory"]])
    if request["settings"]["budget"] is not None:
        command.extend(["--budget", str(request["settings"]["budget"])])
    if request["seed"] is not None:
        command.extend(["--seed", str(request["seed"])])
    return native_plan(request, command, seed_forwarded=request["seed"], warnings=[
        "Uses the native cobalt subset normalized RMS objective and LAMMPS, not the complete common trainset or MoS2 objective.",
        "Native cobalt history is written at completion; best-field checkpoints can appear earlier.",
    ], dataset_scope="Co_clusters/Co2_cluster energy targets only",
        engine="LAMMPS ReaxFF", objective="cobalt_subset_accuracy_divided_normalized_RMS")
