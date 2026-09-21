"""Stream atom identities from xmolout without parsing coordinate values."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from reaxkit.engine.reaxff.io.xmolout_handler import _parse_xmolout_header


def iter_xmolout_atom_identities(
    path: str | Path,
    *,
    frame_indices: Sequence[int] | None = None,
    reporter=None,
) -> Iterator[dict[str, Any]]:
    """Yield atom numbers and element labels for every requested xmolout frame.

    Atom numbers are the one-based row positions used by xmolout. Coordinate
    tokens and optional per-atom numeric fields are skipped without conversion.
    """

    source = Path(path)
    requested = set(int(value) for value in frame_indices) if frame_indices is not None else None
    max_requested = max(requested, default=-1) if requested is not None else None
    source_index = -1
    emitted = 0
    with source.open("r", encoding="utf-8") as handle:
        while True:
            count_line = next((line.strip() for line in handle if line.strip()), None)
            if count_line is None:
                break
            values = count_line.split()
            if len(values) != 1 or not values[0].isdigit():
                continue
            source_index += 1
            if max_requested is not None and source_index > max_requested:
                break
            atom_count = int(values[0])
            header = next((line.strip() for line in handle if line.strip()), None)
            if header is None:
                raise ValueError(f"xmolout frame {source_index} has no header: {source}")
            _, iteration, _ = _parse_xmolout_header(
                header,
                path=source,
                frame_index=source_index,
                line_number=None,
            )
            selected = requested is None or source_index in requested
            elements: list[str] = []
            for _ in range(atom_count):
                atom_line = next((line.strip() for line in handle if line.strip()), None)
                if atom_line is None:
                    raise ValueError(f"xmolout frame {source_index} atom block is truncated: {source}")
                if selected:
                    elements.append(atom_line.split(None, 1)[0])
            if not selected:
                continue
            emitted += 1
            if callable(reporter):
                total = len(requested) if requested is not None else 0
                reporter("stream", emitted, total, "Streaming xmolout atom identities")
            yield {
                "source_index": source_index,
                "iter": iteration,
                "atom_ids": list(range(1, atom_count + 1)),
                "elements": elements,
            }


def load_xmolout_atom_identities(path: str | Path) -> dict[int, tuple[list[int], list[str]]]:
    """Materialize lightweight identity records for every xmolout frame."""

    return {
        int(record["source_index"]): (list(record["atom_ids"]), list(record["elements"]))
        for record in iter_xmolout_atom_identities(path)
    }


__all__ = ["iter_xmolout_atom_identities", "load_xmolout_atom_identities"]
