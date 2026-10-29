"""Compatibility access to the packaged stylesheet; Dash loads the CSS asset directly."""
from pathlib import Path
_CSS = (Path(__file__).resolve().parents[2] / "assets" / "workspace.css").read_text(encoding="utf-8")
