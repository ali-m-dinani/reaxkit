"""Validation shared by ReaxFF frame-based text readers."""

from __future__ import annotations

from pathlib import Path

from reaxkit.core.platform.exceptions import ParseError


def validate_geometry_name(
    name: str,
    *,
    file_kind: str,
    path: str | Path,
    frame_index: int,
    line_number: int | None,
    header: str,
) -> str:
    """Reject geometry-name fields that contain shifted lattice metadata."""
    invalid = (
        not name
        or any(character.isspace() for character in name)
        or any(character in name for character in ('=', '"', "'"))
        or name.casefold().startswith("lattice")
    )
    if not invalid:
        return name

    location = f"frame {frame_index}"
    if line_number is not None:
        location += f", line {line_number}"
    raise ParseError(
        f"Malformed {file_kind} frame header in '{Path(path)}' ({location}): "
        f"the geometry name {name!r} is invalid. It looks like lattice metadata "
        "was shifted into the geometry-name field, usually because an invalid or "
        "overlong geometry name was truncated during the ReaxFF run. Use a short "
        "geometry name without spaces, quotes, or '=' and regenerate xmolout and "
        f"fort.7. Header: {header.strip()!r}"
    )


__all__ = ["validate_geometry_name"]
