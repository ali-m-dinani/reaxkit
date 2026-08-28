from pathlib import Path

import pytest

from reaxkit.engine.reaxff.io.fort83_handler import (
    DEFAULT_OPTIMIZED_FFIELD_NAME,
    Fort83Handler,
    extract_optimized_ffield,
)


def test_get_optimized_ffield_returns_block_after_last_marker(tmp_path: Path) -> None:
    fort83 = tmp_path / "fort.83"
    fort83.write_bytes(
        b"iteration 1\n"
        b"Error force field: 12.0\n"
        b"old ffield\n"
        b"iteration 2\n"
        b"Error force field: 3.5\n"
        b"Reactive MD-force field\n"
        b"39 ! Number of general parameters\n"
    )

    result = Fort83Handler(fort83).get_optimized_ffield()

    assert result == (
        "Reactive MD-force field\n"
        "39 ! Number of general parameters\n"
    )


def test_write_optimized_ffield_uses_requested_path(tmp_path: Path) -> None:
    fort83 = tmp_path / "fort.83"
    fort83.write_bytes(b"Error force field\noptimized force field\n")
    output = tmp_path / "results" / DEFAULT_OPTIMIZED_FFIELD_NAME

    returned_path = extract_optimized_ffield(fort83, output)

    assert returned_path == output
    assert output.read_bytes() == b"optimized force field\n"


def test_get_optimized_ffield_rejects_file_without_marker(tmp_path: Path) -> None:
    fort83 = tmp_path / "fort.83"
    fort83.write_text("no force-field marker here\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Error force field"):
        Fort83Handler(fort83).get_optimized_ffield()
