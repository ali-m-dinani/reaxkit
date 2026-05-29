"""Internal helper modules for the ReaxFF engine adapter.

This package keeps `adapter.py` focused on API-facing orchestration while
isolating path resolution, timing, and write-side formatting logic.

**Usage context**

- Adapter wiring: Imported by `ReaxFFAdapter` methods.
- Internal APIs: Not part of the public `reaxkit.engine` contract.
- Refactoring seam: Enables responsibility-based module splits.
"""

from .io_paths import (
    _quick_n_frames,
    _quick_n_frames_from_control,
    _quick_n_frames_from_geo_xmol,
    _resolve_against_run_dir,
    _resolve_reaxff_path,
)
from .timing import _build_handler, _emit_load_timing, _time_source
from .writers import _write_control_data, _write_trajectory_data

__all__ = [
    "_build_handler",
    "_emit_load_timing",
    "_quick_n_frames",
    "_quick_n_frames_from_control",
    "_quick_n_frames_from_geo_xmol",
    "_resolve_against_run_dir",
    "_resolve_reaxff_path",
    "_time_source",
    "_write_control_data",
    "_write_trajectory_data",
]
