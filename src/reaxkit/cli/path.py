"""
Path utilities and storage maintenance commands for ReaxKit.

This module provides:
- `resolve_output_path(...)` for analysis exports
- direct-command hooks for `reaxkit free-up ...`
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from reaxkit.core.storage_layout import ReaxkitStorageLayout, normalize_storage_args

DEFAULT_DATA_RAW_ROOT = Path("reaxkit_workkspace/data/raw")
DEFAULT_ALT_DATA_RAW_ROOT = Path("data/raw")
FREE_UP_COMMAND = "free-up"


def resolve_output_path(
    user_value: str,
    workflow: str,
    *,
    run_id: str | None = None,
    project_root: str | Path = ".",
    analysis_id: str | None = None,
) -> Path:
    """
    Resolve the output path for a workflow result.

    If the user provides only a bare filename, the file is written under
    ``reaxkit_outputs/<workflow>/``. If the user provides an absolute path
    or a path containing directories, that path is used directly.
    """
    p = Path(user_value)

    # If user gave an absolute path or a relative path with dirs,
    # respect it exactly.
    if p.is_absolute() or p.parent != Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    norm = normalize_storage_args(
        {
            "run_id": run_id,
            "project_root": str(project_root),
            "analysis_id": analysis_id,
        }
    )
    layout = ReaxkitStorageLayout(project_root=Path(norm["project_root"]))
    layout.ensure_base_layout()
    effective_run_id = str(norm["run_id"])
    scoped = layout.analysis_root / workflow / str(analysis_id or effective_run_id)
    scoped.mkdir(parents=True, exist_ok=True)
    return scoped / p.name


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be an integer.") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return parsed


def _resolve_default_raw_root() -> Path:
    if DEFAULT_DATA_RAW_ROOT.exists():
        return DEFAULT_DATA_RAW_ROOT
    if DEFAULT_ALT_DATA_RAW_ROOT.exists():
        return DEFAULT_ALT_DATA_RAW_ROOT
    return DEFAULT_DATA_RAW_ROOT


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".tar.gz") or name.endswith(".tar.zst") or name.endswith(".gz") or name.endswith(".zst")


def _raw_entries(raw_root: Path) -> list[Path]:
    if not raw_root.exists():
        return []
    entries: list[Path] = []
    for item in raw_root.iterdir():
        if item.name.startswith("."):
            continue
        if _is_archive(item):
            continue
        if item.is_dir() or item.is_file():
            entries.append(item)
    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return entries


def _delete_entry(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _archive_path(path: Path, compression: str) -> Path:
    suffix = ".tar.gz" if compression == "gz" else ".tar.zst"
    return path.with_name(f"{path.name}{suffix}")


def _compress_to_archive(source: Path, archive: Path, *, compression: str) -> None:
    if compression == "gz":
        with tarfile.open(archive, mode="w:gz") as tar:
            tar.add(source, arcname=source.name)
        return

    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError(
            "zstd compression requested but optional dependency `zstandard` is not installed. "
            "Install it or use --format gz."
        ) from exc

    with archive.open("wb") as out_f:
        compressor = zstd.ZstdCompressor(level=3)
        with compressor.stream_writer(out_f) as zstd_stream:
            with tarfile.open(fileobj=zstd_stream, mode="w|") as tar:
                tar.add(source, arcname=source.name)


def free_up_keep_last(raw_root: Path, keep: int, *, dry_run: bool = False) -> list[Path]:
    entries = _raw_entries(raw_root)
    victims = entries[keep:]
    for victim in victims:
        _delete_entry(victim, dry_run=dry_run)
    return victims


def free_up_compress_old(raw_root: Path, keep: int, *, compression: str = "gz", dry_run: bool = False) -> tuple[list[Path], list[Path]]:
    entries = _raw_entries(raw_root)
    victims = entries[keep:]
    archives: list[Path] = []
    skipped: list[Path] = []
    for victim in victims:
        archive = _archive_path(victim, compression)
        if archive.exists():
            skipped.append(victim)
            continue
        if not dry_run:
            _compress_to_archive(victim, archive, compression=compression)
        archives.append(archive)
        _delete_entry(victim, dry_run=dry_run)
    return archives, skipped


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Manage storage under raw run outputs.\n\n"
        "Examples:\n"
        "  reaxkit free-up --last 5\n"
        "  reaxkit free-up --compress 5\n"
        "  reaxkit free-up --compress 5 --format zst\n"
        "  reaxkit free-up --last 3 --raw-root reaxkit_workkspace/data/raw --dry-run"
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--last", type=_positive_int, metavar="N", help="Keep only the latest N raw runs and delete older ones.")
    mode_group.add_argument("--compress", type=_positive_int, metavar="N", help="Compress all but the latest N raw runs.")
    parser.add_argument("--raw-root", default=None, help="Path to raw run directory (default: auto-detect data/raw).")
    parser.add_argument("--format", choices=("gz", "zst"), default="gz", help="Archive format for --compress.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without writing/deleting files.")
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    _ = command
    raw_root = Path(args.raw_root) if args.raw_root else _resolve_default_raw_root()
    if not raw_root.exists():
        print(f"[Info] Raw directory not found: {raw_root}")
        return 0

    if args.last is not None:
        deleted = free_up_keep_last(raw_root, keep=args.last, dry_run=args.dry_run)
        action = "Would delete" if args.dry_run else "Deleted"
        print(f"{action} {len(deleted)} raw run(s) from {raw_root} (kept latest {args.last}).")
        for path in deleted:
            print(f"  - {path}")
        return 0

    archives, skipped = free_up_compress_old(
        raw_root,
        keep=args.compress,
        compression=args.format,
        dry_run=args.dry_run,
    )
    action = "Would create" if args.dry_run else "Created"
    suffix = ".tar.gz" if args.format == "gz" else ".tar.zst"
    print(f"{action} {len(archives)} archive(s) in {raw_root} using {suffix} (kept latest {args.compress}).")
    for archive in archives:
        print(f"  - {archive}")
    if skipped:
        print(f"[Info] Skipped {len(skipped)} run(s) because archive already exists:")
        for path in skipped:
            print(f"  - {path}")
    return 0
