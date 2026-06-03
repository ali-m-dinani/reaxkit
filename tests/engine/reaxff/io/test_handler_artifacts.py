from __future__ import annotations

from pathlib import Path

import pytest

from .handler_artifacts import EXAMPLES_DIR, HANDLER_SPECS, generate_handler_artifacts


def test_generate_handler_artifacts(tmp_path: Path):
    if not EXAMPLES_DIR.exists():
        pytest.skip(f"Example input directory not available: {EXAMPLES_DIR}")

    index = generate_handler_artifacts(tmp_path)

    assert set(index) == {spec.name for spec in HANDLER_SPECS}
    assert (tmp_path / "index.json").exists()

    for spec in HANDLER_SPECS:
        handler_dir = tmp_path / spec.name
        assert handler_dir.exists()
        assert (handler_dir / "metadata.json").exists()
        exported = [Path(path) for path in index[spec.name]["exported_files"]]
        assert exported
        assert any(path.exists() for path in exported)
