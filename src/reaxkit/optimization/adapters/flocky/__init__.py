"""Read native Flocky checkpoints without linking optimizer or engine code."""

import re

from ..base import Reader
from ...models import finite, progress


def convert(root, options=None):
    reader = Reader(root, options)
    profile = reader.root / "flocky-profile.json"
    if profile.is_file():
        reader.json(profile)
    for path in sorted(reader.root.glob("opti_log.out.*")):
        match = re.fullmatch(r"opti_log\.out\.(\d+)", path.name)
        if not match:
            continue
        cycle = int(match[1])
        previous = -1
        for row, line in reader.lines(path):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2 or not fields[0].isdigit():
                reader.result.warnings.append(f"{path.name}:{row}: malformed checkpoint")
                continue
            iteration = int(fields[0])
            if iteration <= previous:
                reader.result.warnings.append(f"{path.name}:{row}: duplicate/restarted iteration deferred")
                continue
            previous = iteration
            score = finite(fields[1])
            if score is None:
                reader.result.warnings.append(f"{path.name}:{row}: nonfinite fitness; not a usable scientific result")
            reader.result.progress.append(progress(
                reader.relative(path), row, fields[1], native_step=iteration,
                step_kind="iteration", cycle=cycle, best_score=score,
                record_kind="best_so_far_checkpoint", candidate_id=None,
                objective_definition="native fitness including any configured regularization",
                engine_identity="Tapered ReaxFF", precision="native_10_decimal_scientific",
                status="ok" if score is not None else "nonfinite",
            ))
    reader.result.progress.sort(key=lambda item: (item["cycle"], item["native_step"]))
    for path in sorted(reader.root.glob("ffield.gbest.*")):
        if re.fullmatch(r"ffield\.gbest\.\d+\.\d+\.\d+", path.name):
            reader.artifact(path, "parameter_only")
    for path in sorted(reader.root.glob("results.out.*")):
        if re.fullmatch(r"results\.out\.\d+", path.name):
            reader.artifact(path)
    reader.artifact(reader.root / "log.flocky")
    reader.result.warnings.extend([
        "History contains sampled best-so-far values per cycle, not every candidate evaluation",
        "results.out files may be overwritten; candidate association and parity extraction remain unverified",
        "Tapered ReaxFF objective is not assumed equivalent to AMS, JAX or LAMMPS objectives",
    ])
    return reader.result
