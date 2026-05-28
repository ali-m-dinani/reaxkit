"""Run MkDocs with a fresh single-instance dev server."""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import tempfile
import time
import webbrowser
from pathlib import Path


def _lock_path(port: int) -> Path:
    return Path(tempfile.gettempdir()) / f"reaxkit_mkdocs_{port}.pid"


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
        pid = int(text)
        return pid if pid > 0 else None
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _terminate_pid(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        os.kill(pid, signal.SIGTERM)


def _ensure_single_instance(port: int) -> Path:
    lock = _lock_path(port)
    current_pid = os.getpid()
    existing_pid = _read_pid(lock)
    if existing_pid and existing_pid != current_pid and _pid_alive(existing_pid):
        _terminate_pid(existing_pid)
        deadline = time.time() + 4.0
        while time.time() < deadline:
            if not _pid_alive(existing_pid):
                break
            time.sleep(0.1)
    lock.write_text(str(current_pid), encoding="utf-8")

    def _cleanup() -> None:
        try:
            if _read_pid(lock) == current_pid:
                lock.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_cleanup)
    return lock


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    port = int(os.environ.get("REAXKIT_MKDOCS_PORT", "8010"))
    _ensure_single_instance(port)

    env = os.environ.copy()
    env.setdefault("WATCHDOG_USE_POLLING", "true")

    url = f"http://127.0.0.1:{port}/?v={int(time.time())}"
    webbrowser.open(url)

    cmd = [
        sys.executable,
        "-m",
        "mkdocs",
        "serve",
        "--dirtyreload",
        "--dev-addr",
        f"127.0.0.1:{port}",
    ]
    proc = subprocess.run(cmd, env=env, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
