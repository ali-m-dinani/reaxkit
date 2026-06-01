"""MkDocs hook script for pre-build doc generation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def on_pre_build(config, **kwargs):  # noqa: D401
    """Generate docs that are derived from source before site build."""
    mkdocs_file = Path(config.config_file_path).resolve()
    repo_root = mkdocs_file.parent
    scripts_dir = repo_root / "docs" / "scripts"
    scripts = sorted(scripts_dir.glob("*.py"))
    for script in scripts:
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(repo_root))
