"""Read the supplied Fortran ffsav unit-83 force-field stream."""

import re

from ..base import NUMBER, Reader
from ...models import progress


def convert(root, options=None):
    reader = Reader(root, options)
    path = reader.root / reader.options.get("progress_file", "fort.83")
    completed = 0
    if path.is_file():
        for line_number, line in reader.lines(path):
            match = re.match(r"Error force field:\s*(" + NUMBER + r")", line, re.I)
            if match:
                completed += 1
                reader.result.progress.append(progress(
                    reader.relative(path), line_number, match[1],
                    native_step=completed, step_kind="parameter_row",
                    candidate_id=f"unit83-record:{completed}", precision="native_4_decimals",
                ))
        reader.artifact(path)
        reader.result.warnings.append("Unit 83 contains concatenated force fields; last field may be incomplete and is not claimed restartable")
    return reader.result
