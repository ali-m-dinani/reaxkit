"""Native command contracts without importing optimizer or engine libraries."""

from __future__ import annotations

import importlib
import math
import re
import shutil
from pathlib import Path

from .models import OptimizationError
from .storage import atomic_text, digest


EXTERNAL_ONLY = {
    "indeedopt": "The audited script has fixed dataset constants and overwrites bounds using gpf=5. Use a separately maintained external runner; these settings cannot be applied through its native CLI.",
    "ezff": "EZFF exposes a Python API and a user-defined objective, not a universal CLI. Use an external runner in the EZFF project/environment.",
}


def adapter(optimizer):
    if optimizer in EXTERNAL_ONLY:
        raise OptimizationError(f"{optimizer}: {EXTERNAL_ONLY[optimizer]}")
    return importlib.import_module(f"reaxkit.optimization.adapters.{optimizer}.launch")


def validate_native(config, optimizer, section):
    if section["runtime"]["mode"] != "native":
        return
    native = adapter(optimizer)
    parameters(config["inputs"]["params"], config["inputs"]["params_bounds_order"])
    if section.get("prepared_inputs"):
        raise OptimizationError(f"{optimizer}: prepared_inputs belongs to external_runner mode; native mode stages its own inputs")
    native.validate(config, section)


def parameters(filename, order):
    rows = []
    seen = set()
    for line_number, line in enumerate(Path(filename).read_text(encoding="utf-8").splitlines(), 1):
        body = re.split(r"[!#]", line, maxsplit=1)[0].strip()
        if not body:
            continue
        fields = body.split()
        try:
            if len(fields) != 6:
                raise ValueError("expected six columns")
            indices = tuple(int(value) for value in fields[:3])
            numbers = [float(value.replace("D", "E").replace("d", "e")) for value in fields[3:]]
            if not all(math.isfinite(value) for value in numbers):
                raise ValueError("nonfinite value")
            step, first, second = numbers
            lower, upper = (first, second) if order == "min_max" else (second, first)
            if indices[0] < 1 or indices[1] < 1 or indices[2] < 0 or lower >= upper:
                raise ValueError("invalid index or bounds inconsistent with params_bounds_order")
            if indices in seen:
                raise ValueError("duplicate parameter")
        except ValueError as error:
            raise OptimizationError(f"{filename}:{line_number}: {error}") from error
        seen.add(indices)
        rows.append((*indices, step, lower, upper))
    if not rows:
        raise OptimizationError(f"{filename}: no six-column parameter records")
    return rows


def parameter_text(rows, order="min_max"):
    lines = []
    for section, entry, parameter, step, lower, upper in rows:
        bounds = (lower, upper) if order == "min_max" else (upper, lower)
        lines.append(f"{section} {entry} {parameter} {step:.17g} {bounds[0]:.17g} {bounds[1]:.17g}\n")
    return "".join(lines)


def validate_jax_data(trainset, geo):
    names = []
    for line in Path(geo).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == "DESCRP":
            if len(fields) != 2:
                raise OptimizationError("JAX geometry DESCRP must have one structure identifier")
            names.append(fields[1])
    if not names or len(set(names)) != len(names):
        raise OptimizationError("JAX geometries require unique DESCRP identifiers")
    active = None
    count = 0
    references = set()
    for line_number, line in enumerate(Path(trainset).read_text(encoding="utf-8").splitlines(), 1):
        body = re.split(r"[!#]", line, maxsplit=1)[0].strip()
        if not body:
            continue
        if body in ("ENERGY", "CHARGE", "GEOMETRY", "FORCES"):
            if active:
                raise OptimizationError(f"{trainset}:{line_number}: nested training sections")
            active = body
            continue
        if active and body == "END" + active:
            active = None
            continue
        try:
            if not active:
                raise ValueError("unsupported section or record outside a section")
            fields = body.replace("/", " / ").split()
            if active == "ENERGY":
                if len(fields) < 6 or (len(fields) - 2) % 4:
                    raise ValueError("expected accuracy, signed structure/divisor terms, target")
                weight = float(fields[0])
                for offset in range(1, len(fields) - 1, 4):
                    sign, name, separator, divisor = fields[offset:offset + 4]
                    divisor = float(divisor)
                    if sign not in ("+", "-") or separator != "/" or not math.isfinite(divisor) or divisor <= 0:
                        raise ValueError("invalid energy term")
                    references.add(name)
                numeric = [weight, float(fields[-1])]
            else:
                allowed = {"CHARGE": (4,), "GEOMETRY": (5, 6, 7), "FORCES": (6,)}[active]
                if len(fields) not in allowed:
                    raise ValueError("unsupported record shape (RMSG-only geometry loss is not implemented)")
                references.add(fields[0])
                weight = float(fields[1])
                indices = fields[2:-1] if active == "GEOMETRY" else fields[2:3]
                if any(int(value) < 1 for value in indices):
                    raise ValueError("atom indices must be positive")
                numeric = [weight] + [float(value) for value in (fields[3:] if active == "FORCES" else fields[-1:])]
            if weight <= 0 or not all(math.isfinite(value) for value in numeric):
                raise ValueError("targets must be finite and accuracy divisors positive")
        except (ValueError, IndexError) as error:
            raise OptimizationError(f"{trainset}:{line_number}: {error}") from error
        count += 1
    if active or not count:
        raise OptimizationError(f"{trainset}: incomplete or empty supported training set")
    if references - set(names):
        raise OptimizationError(f"{trainset}: missing geometries would be silently filtered by JAX: {sorted(references - set(names))}")


def control_text(filename, settings, explicit, fixed_width=False):
    original = Path(filename).read_text(encoding="utf-8") if filename else ""
    lines = []
    seen = set()
    for line in original.splitlines():
        body = re.split(r"[!#]", line, maxsplit=1)[0].split()
        if not body:
            if line.strip():
                lines.append("# " + line.lstrip().lstrip("#!").strip() if fixed_width else line)
            continue
        if len(body) < 2:
            raise OptimizationError("control: expected value followed by keyword")
        keyword = body[1].lower()
        if keyword in seen:
            raise OptimizationError(f"control: duplicate keyword {keyword}")
        seen.add(keyword)
        if keyword in explicit and settings.get(keyword) is not None:
            continue
        if fixed_width:
            numeric = body[0]
            if len(numeric) > 7:
                raise OptimizationError(f"control: {keyword} value does not fit the legacy eight-column format")
            line = f"{numeric:>7} {keyword:<6}"
        lines.append(line)
    for keyword, value in settings.items():
        if value is not None and (keyword in explicit or keyword not in seen):
            numeric = format(value, ".15g")
            if len(numeric) > 7 and fixed_width:
                raise OptimizationError(f"control: {keyword} value does not fit the legacy eight-column format")
            lines.append(f"{numeric:>7} {keyword:<6}" if fixed_width else f"{numeric} {keyword}")
    return "\n".join(lines) + "\n"


def stage_common(request, destination, bounds_order="min_max", names=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    names = names or {}
    result = {}
    for key, source in request["inputs"].items():
        target = destination / names.get(key, "trainset.in" if key == "trainset" else key)
        if target.exists():
            raise OptimizationError(f"Refusing to overwrite native input {target}")
        if key == "params":
            rows = parameters(source, request["params_bounds_order"])
            atomic_text(target, parameter_text(rows, bounds_order))
        else:
            shutil.copyfile(source, target)
        result[key] = str(target)
    return result


def native_plan(request, command, cwd=None, **values):
    native = request["native_directory"]
    return {"mode": "native", "commands": [{"argv": command, "cwd": str(cwd or native), "phase": "optimizer"}],
            "results_directory": native, "environment": {}, "seed_forwarded": None,
            "warnings": [], **values}


def add_settings(command, settings, rules, excluded=()):
    for key, value in settings.items():
        if value is not None and key not in excluded:
            flag = rules[key]["native"]
            if not re.fullmatch(r"--[\w-]+", flag):
                raise OptimizationError(f"No direct native argument mapping for {key}")
            command.extend([flag, str(value)])


def prepare_launch(request):
    if request["runtime"]["mode"] == "native":
        return adapter(request["optimizer"]).prepare(request)
    directory = Path(request["native_directory"]).parent
    command = [str(directory / "run-config.json") if argument == "{run_config}" else argument
               for argument in request["runtime"]["command"]]
    return {"mode": "external_runner", "commands": [{"argv": command, "cwd": request["native_directory"], "phase": "optimizer"}],
            "results_directory": request["native_directory"], "environment": {}, "seed_forwarded": None,
            "warnings": ["External runner is responsible for settings, input compatibility, engine/objective fidelity and seeding."]}


def staged_hashes(directory):
    directory = Path(directory)
    return {path.relative_to(directory).as_posix(): digest(path)
            for folder in ("input", "native") for path in sorted((directory / folder).rglob("*")) if path.is_file()}


def verify_staged(directory, checksums, external=None):
    directory = Path(directory).resolve()
    for relative, expected in checksums.items():
        path = (directory / relative).resolve()
        if directory not in path.parents or not path.is_file() or digest(path) != expected:
            raise OptimizationError(f"Staged input changed or unavailable: {relative}")
    for filename, expected in (external or {}).items():
        path = Path(filename)
        if not path.is_file() or digest(path) != expected:
            raise OptimizationError(f"External input/interpreter changed or unavailable: {filename}")


def expand_command(command, environment):
    result = list(command)
    executable = result[0]
    if executable.startswith("$"):
        match = re.fullmatch(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))(/.*)?", executable)
        key = (match[1] or match[2]) if match else None
        if not key or not environment.get(key):
            raise OptimizationError(f"Executable environment variable is unset: {executable}")
        result[0] = environment[key] + (match[3] or "")
    return result
