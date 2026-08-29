"""Path and frame-probe helpers for the ReaxFF adapter.

This module centralizes argument-to-path resolution and lightweight frame-count
probes used by metadata flows and Web UI preloads.

**Usage context**

- Path normalization: Resolve canonical file paths from heterogeneous CLI/workflow args.
- Metadata probing: Estimate frame counts without full parser initialization.
- Adapter internals: Consumed by `ReaxFFAdapter` class/static methods.
"""

from __future__ import annotations

from pathlib import Path


def _resolve_reaxff_path(args: dict, *keys: str, default: str) -> Path:
    """Resolve a ReaxFF input/output path from argument aliases."""
    for key in keys:
        raw = args.get(key)
        if raw:
            path = Path(raw)
            return path / default if path.is_dir() else path

    run_dir = args.get("run_dir")
    if run_dir:
        return Path(run_dir) / default

    input_path = args.get("input")
    if input_path:
        path = Path(input_path)
        return path / default if path.is_dir() else path

    return Path(default)


def _resolve_against_run_dir(args: dict, path: Path) -> Path:
    """Resolve a relative path against run_dir when the candidate exists."""
    run_dir = args.get("run_dir")
    if run_dir and not path.is_absolute():
        candidate = Path(run_dir) / path
        if candidate.exists():
            return candidate
    return path


def _quick_n_frames_from_control(control_path: Path) -> int | None:
    """Estimate frame count from control `nmdit` and `iout2` parameters."""
    if not control_path.exists() or not control_path.is_file():
        return None
    nmdit: int | None = None
    iout2: int | None = None
    try:
        with open(control_path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                # ReaxFF control commonly uses "<value> <key>" but accept both orders.
                for key, value in ((parts[1].lower(), parts[0]), (parts[0].lower(), parts[1])):
                    if key not in {"nmdit", "iout2"}:
                        continue
                    try:
                        parsed = int(float(value))
                    except Exception:
                        continue
                    if key == "nmdit":
                        nmdit = parsed
                    else:
                        iout2 = parsed
                if nmdit is not None and iout2 is not None and iout2 > 0:
                    break
    except Exception:
        return None
    if nmdit is None or iout2 is None or iout2 <= 0:
        return None
    return max(1, int(nmdit / iout2) + 1)


def _quick_n_frames_from_geo_xmol(geo_path: Path, xmol_path: Path) -> int | None:
    """Count xmolout frame-start lines without parsing frame contents."""
    _ = geo_path  # Retained for compatibility with the existing helper API.
    if not xmol_path.exists() or not xmol_path.is_file():
        return None
    count = 0
    try:
        with open(xmol_path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                # Each xmolout frame starts with a line containing only n_atoms.
                if raw.strip().isdigit():
                    count += 1
    except Exception:
        return None
    return count


def _quick_n_frames(args: dict) -> int | None:
    """Run fast frame-count probes for metadata-aware UI and workflows."""
    geo_raw = args.get("geo") or args.get("geometry") or args.get("run_dir") or args.get("input") or "geo"
    geo_path = Path(geo_raw)
    geo_path = geo_path / "geo" if geo_path.is_dir() else geo_path
    geo_path = _resolve_against_run_dir(args, geo_path)

    xmol_path = _resolve_reaxff_path(args, "xmolout", default="xmolout")
    xmol_path = _resolve_against_run_dir(args, xmol_path)
    n_from_xmolout = _quick_n_frames_from_geo_xmol(geo_path, xmol_path)
    if n_from_xmolout is not None:
        return n_from_xmolout

    # A control-file estimate is useful only when no trajectory is available.
    control_path = _resolve_reaxff_path(args, "control", "control_file", default="control")
    control_path = _resolve_against_run_dir(args, control_path)
    return _quick_n_frames_from_control(control_path)
