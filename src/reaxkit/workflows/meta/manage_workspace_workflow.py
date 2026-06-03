"""Workspace cleanup and archiving commands.

This module implements CLI workflow orchestration for its command family, including argument parsing, request construction, execution dispatch, and result presentation handoff.

**Usage context**

- Command routing: Resolve CLI aliases and normalized command names.
- Task execution: Build request objects and invoke registered tasks.
- Output handling: Forward results to table, plot, export, or report flows.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from reaxkit.core.storage.storage_layout import default_project_root

MANAGE_WORKSPACE_COMMAND = "manage-workspace"
FREE_UP_COMMAND = "free-up"
ALL_COMMANDS = (MANAGE_WORKSPACE_COMMAND,)
ALL_LEGACY_COMMANDS = (FREE_UP_COMMAND, "manage_workspace", "free_up")
DEFAULT_ALT_WORKSPACE = Path("reaxkit_workspace")


def _positive_int(value: str) -> int:
    """Positive int."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be an integer.") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return parsed


def _is_archive(path: Path) -> bool:
    """Is archive."""
    name = path.name.lower()
    return name.endswith(".tar.gz") or name.endswith(".tar.zst") or name.endswith(".gz") or name.endswith(".zst")


def _human_size(num_bytes: int) -> str:
    """Human size."""
    value = float(max(0, int(num_bytes)))
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.1f}{units[idx]}"


def _entry_size(path: Path) -> int:
    """Entry size."""
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += int(child.stat().st_size)
    return total


def _resolve_workspace_root(workspace_root: str | None) -> Path:
    """Resolve workspace root."""
    if workspace_root:
        root = Path(workspace_root)
        return root

    candidates = [
        default_project_root(),
        DEFAULT_ALT_WORKSPACE,
        Path.cwd() / default_project_root(),
        Path.cwd() / DEFAULT_ALT_WORKSPACE,
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return default_project_root()


def _resolve_target_folder(*, workspace_root: Path, folder: str) -> Path:
    """Resolve target folder."""
    raw = Path(str(folder).strip())
    if raw.is_absolute():
        return raw
    return workspace_root / raw


def _collect_entries(target: Path, *, include_archives: bool = False) -> list[Path]:
    """Collect entries."""
    if not target.exists() or not target.is_dir():
        return []
    entries: list[Path] = []
    for item in target.iterdir():
        if item.name.startswith("."):
            continue
        if not include_archives and _is_archive(item):
            continue
        if item.is_dir() or item.is_file():
            entries.append(item)
    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return entries


def _delete_entry(path: Path, *, dry_run: bool) -> None:
    """Delete entry."""
    if dry_run:
        return
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _archive_path(path: Path, compression: str) -> Path:
    """Archive path."""
    suffix = ".tar.gz" if compression == "gz" else ".tar.zst"
    return path.with_name(f"{path.name}{suffix}")


def _compress_to_archive(source: Path, archive: Path, *, compression: str) -> None:
    """Compress to archive."""
    if compression == "gz":
        with tarfile.open(archive, mode="w:gz") as tar:
            tar.add(source, arcname=source.name)
        return

    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError(
            "zst compression requested but optional dependency `zstandard` is not installed. "
            "Install it or use --format gz."
        ) from exc

    with archive.open("wb") as out_f:
        compressor = zstd.ZstdCompressor(level=3)
        with compressor.stream_writer(out_f) as zstd_stream:
            with tarfile.open(fileobj=zstd_stream, mode="w|") as tar:
                tar.add(source, arcname=source.name)


def _print_list(target: Path) -> int:
    """Print list."""
    entries = _collect_entries(target, include_archives=True)
    if not entries:
        print(f"[Info] No entries found under {target}")
        return 0
    total = 0
    print(f"[Info] Entries under {target}:")
    for item in entries:
        size = _entry_size(item)
        total += size
        stamp = item.stat().st_mtime
        print(f"  - {item.name}  size={_human_size(size)}  mtime={stamp:.0f}")
    print(f"[Info] Total entries: {len(entries)} | total size: {_human_size(total)}")
    return 0


def _delete_policy(entries: list[Path], keep_last: int) -> list[Path]:
    """Delete policy."""
    if keep_last <= 0:
        return list(entries)
    return entries[keep_last:]


def _run_manage_workspace(args: argparse.Namespace) -> int:
    """Run manage workspace."""
    workspace_root = _resolve_workspace_root(args.workspace_root)
    target = _resolve_target_folder(workspace_root=workspace_root, folder=args.folder)
    if not target.exists():
        print(f"[Info] Target folder not found: {target}")
        return 0

    action = str(args.action or "delete").strip().lower()
    dry_run = bool(args.dry_run)
    keep_n = int(args.number or 0)

    if action == "list":
        return _print_list(target)

    entries = _collect_entries(target, include_archives=False)
    if not entries:
        print(f"[Info] No entries found under {target}")
        return 0

    if action in {"delete", "keep"}:
        victims = _delete_policy(entries, keep_last=(keep_n if action == "keep" else keep_n))
        freed = sum(_entry_size(v) for v in victims)
        for victim in victims:
            _delete_entry(victim, dry_run=dry_run)
        verb = "Would delete" if dry_run else "Deleted"
        print(f"{verb} {len(victims)} item(s) from {target}.")
        print(f"Estimated reclaimed size: {_human_size(freed)}")
        for victim in victims:
            print(f"  - {victim}")
        return 0

    if action == "archive":
        victims = _delete_policy(entries, keep_last=keep_n)
        archives: list[Path] = []
        skipped: list[Path] = []
        reclaimed = 0
        for victim in victims:
            archive = _archive_path(victim, args.format)
            if archive.exists():
                skipped.append(victim)
                continue
            if not dry_run:
                _compress_to_archive(victim, archive, compression=args.format)
            archives.append(archive)
            reclaimed += _entry_size(victim)
            _delete_entry(victim, dry_run=dry_run)
        verb = "Would archive+delete" if dry_run else "Archived+deleted"
        print(f"{verb} {len(archives)} item(s) from {target}.")
        print(f"Estimated reclaimed size: {_human_size(reclaimed)}")
        for archive in archives:
            print(f"  - {archive}")
        if skipped:
            print(f"[Info] Skipped {len(skipped)} item(s); archive already exists:")
            for victim in skipped:
                print(f"  - {victim}")
        return 0

    raise ValueError(f"Unsupported action: {action}")


def _run_free_up_legacy(args: argparse.Namespace) -> int:
    """Run free up legacy."""
    workspace_root = _resolve_workspace_root(args.workspace_root)
    if args.raw_root:
        target = Path(args.raw_root)
    else:
        target = workspace_root / "data" / "raw"
    if not target.exists():
        print(f"[Info] Raw directory not found: {target}")
        return 0

    if args.last is not None:
        entries = _collect_entries(target, include_archives=False)
        victims = _delete_policy(entries, keep_last=int(args.last))
        freed = sum(_entry_size(v) for v in victims)
        for victim in victims:
            _delete_entry(victim, dry_run=bool(args.dry_run))
        verb = "Would delete" if args.dry_run else "Deleted"
        print(f"{verb} {len(victims)} raw run(s) from {target} (kept latest {args.last}).")
        print(f"Estimated reclaimed size: {_human_size(freed)}")
        for victim in victims:
            print(f"  - {victim}")
        return 0

    entries = _collect_entries(target, include_archives=False)
    victims = _delete_policy(entries, keep_last=int(args.compress))
    archives: list[Path] = []
    skipped: list[Path] = []
    reclaimed = 0
    for victim in victims:
        archive = _archive_path(victim, args.format)
        if archive.exists():
            skipped.append(victim)
            continue
        if not args.dry_run:
            _compress_to_archive(victim, archive, compression=args.format)
        archives.append(archive)
        reclaimed += _entry_size(victim)
        _delete_entry(victim, dry_run=bool(args.dry_run))
    verb = "Would archive+delete" if args.dry_run else "Archived+deleted"
    print(f"{verb} {len(archives)} raw run(s) in {target} (kept latest {args.compress}).")
    print(f"Estimated reclaimed size: {_human_size(reclaimed)}")
    for archive in archives:
        print(f"  - {archive}")
    if skipped:
        print(f"[Info] Skipped {len(skipped)} run(s) because archive already exists:")
        for victim in skipped:
            print(f"  - {victim}")
    return 0


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Build parser.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    parser : Any
        Function argument.
    command : Any
        Function argument.

    Returns
    -----
    argparse.ArgumentParser
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
    parser.formatter_class = argparse.RawTextHelpFormatter

    if command == MANAGE_WORKSPACE_COMMAND:
        parser.description = (
            "Manage workspace storage by listing, deleting, keeping, or archiving entries.\n"
            "Use this command to control disk usage under selected workspace folders such as\n"
            "`data/raw` or `cache`, while optionally preserving the newest N entries.\n\n"
            "Examples:\n"
            "  1. Delete all entries in a target folder (default action):\n"
            "   reaxkit manage-workspace --folder data/raw\n\n"
            "  2. Keep only the latest 5 entries and delete older ones:\n"
            "   reaxkit manage-workspace --folder data/raw --action keep --number 5\n\n"
            "  3. Archive older entries (then delete originals) and keep latest 5:\n"
            "   reaxkit manage-workspace --folder data/raw --action archive --number 5 --format zst\n\n"
            "  4. List entries and sizes without deleting anything:\n"
            "   reaxkit manage-workspace --folder cache --action list"
        )
        parser.add_argument(
            "--folder",
            required=True,
            help="Target folder inside workspace. Example: --folder data/raw, which selects the raw-data folder for management.",
        )
        parser.add_argument(
            "--action",
            choices=("list", "delete", "keep", "archive"),
            default="delete",
            help="Action to perform. Example: --action archive, which compresses targets then removes originals when archive succeeds.",
        )
        parser.add_argument(
            "--number",
            type=_positive_int,
            default=0,
            help="Retention count N. Example: --number 5, which keeps the 5 most recent entries and applies action to older ones.",
        )
        parser.add_argument(
            "--format",
            choices=("gz", "zst"),
            default="gz",
            help="Archive format for --action archive. Example: --format zst, which writes .tar.zst archives.",
        )
        parser.add_argument(
            "--workspace-root",
            default=None,
            help="Workspace root path (auto-detected by default). Example: --workspace-root /path/to/reaxkit_workspace, which forces root resolution to that location.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview only; do not write/delete files. Example: --dry-run, which shows what would change without modifying disk.",
        )
        return parser

    if command == FREE_UP_COMMAND:
        parser.description = (
            "Legacy raw-data cleanup command for `data/raw` style run folders.\n"
            "This command supports two exclusive modes: keep latest N by deleting older runs,\n"
            "or archive+delete older runs while keeping latest N unchanged.\n\n"
            "Examples:\n"
            "  1. Keep latest 5 raw runs and delete older ones:\n"
            "   reaxkit free-up --last 5\n\n"
            "  2. Archive+delete older runs while keeping latest 5:\n"
            "   reaxkit free-up --compress 5 --format zst\n\n"
            "  3. Preview cleanup actions without changing files:\n"
            "   reaxkit free-up --last 3 --dry-run"
        )
        mode_group = parser.add_mutually_exclusive_group(required=True)
        mode_group.add_argument(
            "--last",
            type=_positive_int,
            metavar="N",
            help="Keep only the latest N raw runs. Example: --last 5, which deletes runs older than the newest five.",
        )
        mode_group.add_argument(
            "--compress",
            type=_positive_int,
            metavar="N",
            help="Archive+delete all but latest N runs. Example: --compress 5, which archives older runs and keeps five newest unarchived.",
        )
        parser.add_argument(
            "--raw-root",
            default=None,
            help="Raw root path (default: <workspace>/data/raw). Example: --raw-root /tmp/raw_runs, which targets that folder directly.",
        )
        parser.add_argument(
            "--workspace-root",
            default=None,
            help="Workspace root path (auto-detected by default). Example: --workspace-root /path/to/reaxkit_workspace, which sets the base for default raw-root resolution.",
        )
        parser.add_argument(
            "--format",
            choices=("gz", "zst"),
            default="gz",
            help="Archive format for --compress. Example: --format gz, which creates .tar.gz archives.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview only; do not write/delete files. Example: --dry-run, which reports planned actions without modifying files.",
        )
        return parser

    raise KeyError(f"Unsupported workspace command: {command!r}.")


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run main.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    command : Any
        Function argument.
    args : Any
        Function argument.

    Returns
    -----
    int
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
    if command == MANAGE_WORKSPACE_COMMAND:
        return _run_manage_workspace(args)
    if command == FREE_UP_COMMAND:
        return _run_free_up_legacy(args)
    raise KeyError(f"Unsupported workspace command: {command!r}.")
