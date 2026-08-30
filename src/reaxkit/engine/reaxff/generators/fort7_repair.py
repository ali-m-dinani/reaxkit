"""Utilities to repair corrupted ReaxFF ``fort.7`` atom rows.

**Usage context**

- Template generation: Produce canonical text payloads for ReaxFF artifacts.
- File writing: Persist generated outputs to disk with stable formatting.
- Workflow integration: Support higher-level ReaxKit workflow commands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, TextIO

from tqdm.auto import tqdm

_BONDS_RE = re.compile(r"#Bonds:\s*(\d+)")
_ITERATION_RE = re.compile(r"Iteration:\s*(\d+)")
_LEGACY_REPAIRED_ROW_RE = re.compile(r"^\d+ \d+ ")
_MAX_DIGITS = 5

__all__ = ["repair_fort7"]


def _first_float_index(parts: list[str]) -> int:
    """First float index."""
    for i, part in enumerate(parts):
        if "." in part or "e" in part.lower():
            return i
    return -1


def _is_valid_compact_row(raw_tokens: list[str], n_bonds: int) -> bool:
    """Is valid compact row."""
    if len(raw_tokens) != n_bonds + 1:
        return False
    neighbors = [int(x) for x in raw_tokens[:-1]]
    seen_zero = False
    previous = 0
    for value in neighbors:
        if value == 0:
            seen_zero = True
        else:
            if seen_zero or value <= previous or len(str(value)) > _MAX_DIGITS:
                return False
            previous = value
    return True


def _trailing_zero_count(tokens: list[str]) -> int:
    """Trailing zero count."""
    zeros = 0
    for token in reversed(tokens):
        if token == "0":
            zeros += 1
        else:
            break
    return zeros


def _score_seq(seq: tuple[int, ...]) -> int:
    """Score seq."""
    return sum(0 if len(str(x)) >= 4 else (4 - len(str(x))) * 3 for x in seq)


def _positive_prefix_candidates(
    prefix_tokens: list[str],
    required: int,
    *,
    trust_existing_fields: bool = True,
) -> list[tuple[int, ...]]:
    """Return every increasing split of compact neighbor-id fields."""
    if (
        trust_existing_fields
        and len(prefix_tokens) == required
        and all(1 <= len(t) <= _MAX_DIGITS for t in prefix_tokens)
    ):
        values = [int(t) for t in prefix_tokens]
        if all(value > previous for previous, value in zip([0, *values[:-1]], values)):
            return [tuple(values)]

    compact = "".join(prefix_tokens)
    n_chars = len(compact)
    if required == 0:
        return [()] if n_chars == 0 else []
    if n_chars < required or n_chars > required * _MAX_DIGITS:
        return []

    @lru_cache(None)
    def _rec(position: int, used: int, previous: int) -> list[tuple[int, ...]]:
        """Rec."""
        if used == required:
            return [()] if position == n_chars else []
        remaining_numbers = required - used
        remaining_chars = n_chars - position
        if remaining_chars < remaining_numbers or remaining_chars > remaining_numbers * _MAX_DIGITS:
            return []
        solutions: list[tuple[int, ...]] = []
        max_len = min(_MAX_DIGITS, n_chars - position)
        for chunk_len in range(1, max_len + 1):
            remaining_after = n_chars - (position + chunk_len)
            if remaining_after < (remaining_numbers - 1) or remaining_after > (remaining_numbers - 1) * _MAX_DIGITS:
                continue
            chunk = compact[position: position + chunk_len]
            if len(chunk) > 1 and chunk[0] == "0":
                continue
            value = int(chunk)
            if value <= previous:
                continue
            for tail in _rec(position + chunk_len, used + 1, value):
                solutions.append((value,) + tail)
        return solutions

    return _rec(0, 0, 0)


@dataclass(frozen=True)
class _GeometryFrame:
    """Coordinates and periodic cell for one xmolout frame."""

    iteration: int
    coordinates: tuple[tuple[float, float, float], ...]
    cell: tuple[tuple[float, float, float], ...] | None


def _cell_matrix(
    lengths: tuple[float, float, float],
    angles: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...] | None:
    """Build conventional triclinic cell vectors from lengths and angles."""
    a, b, c = lengths
    alpha, beta, gamma = (radians(value) for value in angles)
    sin_gamma = sin(gamma)
    if min(a, b, c) <= 0.0 or abs(sin_gamma) < 1.0e-12:
        return None
    bx = b * cos(gamma)
    by = b * sin_gamma
    cx = c * cos(beta)
    cy = c * (cos(alpha) - cos(beta) * cos(gamma)) / sin_gamma
    cz_squared = c * c - cx * cx - cy * cy
    if cz_squared <= 0.0:
        return None
    return ((a, 0.0, 0.0), (bx, by, 0.0), (cx, cy, sqrt(cz_squared)))


def _minimum_image_distance_squared(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    cell: tuple[tuple[float, float, float], ...] | None,
) -> float:
    """Squared Cartesian distance, applying the minimum-image convention."""
    dx, dy, dz = (second[i] - first[i] for i in range(3))
    if cell is None:
        return dx * dx + dy * dy + dz * dz

    (ax, _, _), (bx, by, _), (cx, cy, cz) = cell
    if min(abs(ax), abs(by), abs(cz)) < 1.0e-12:
        return dx * dx + dy * dy + dz * dz

    # Solve delta = fractional[0] * a + fractional[1] * b + fractional[2] * c.
    fz = dz / cz
    fy = (dy - fz * cy) / by
    fx = (dx - fy * bx - fz * cx) / ax
    centered = (fx - round(fx), fy - round(fy), fz - round(fz))

    # Component-wise wrapping alone can miss the nearest image in a skewed
    # triclinic cell.  Check the 27 images surrounding the wrapped position.
    best = float("inf")
    for ox, oy, oz in product((-1, 0, 1), repeat=3):
        ux, uy, uz = centered[0] + ox, centered[1] + oy, centered[2] + oz
        x = ux * ax + uy * bx + uz * cx
        y = uy * by + uz * cy
        z = uz * cz
        best = min(best, x * x + y * y + z * z)
    return best


def _geometry_score(atom_index: int, neighbors: tuple[int, ...], frame: _GeometryFrame) -> float | None:
    """Score a candidate neighbor list by its total periodic bond length."""
    if not (1 <= atom_index <= len(frame.coordinates)):
        return None
    if any(neighbor == atom_index or not 1 <= neighbor <= len(frame.coordinates) for neighbor in neighbors):
        return None
    center = frame.coordinates[atom_index - 1]
    return sum(
        _minimum_image_distance_squared(center, frame.coordinates[neighbor - 1], frame.cell)
        for neighbor in neighbors
    )


def _choose_positive_prefix(
    prefix_tokens: list[str],
    required: int,
    atom_index: int,
    frame: _GeometryFrame | None,
    *,
    trust_existing_fields: bool = True,
) -> list[int] | None:
    """Choose a compact-id split, preferring the geometrically closest list."""
    solutions = _positive_prefix_candidates(
        prefix_tokens,
        required,
        trust_existing_fields=trust_existing_fields,
    )
    if not solutions:
        return None
    if len(solutions) > 1:
        # An ambiguous digit string cannot be repaired reliably from its
        # pattern alone.  In particular, ``1 6 120619202`` is biased by the
        # legacy digit-length score toward ``1 61 2061 9202`` even though the
        # actual nearby atoms are ``1 6 1206 19202``.  Never silently use that
        # heuristic when geometry is unavailable.
        if frame is None:
            return None
        scored = [(score, seq) for seq in solutions if (score := _geometry_score(atom_index, seq, frame)) is not None]
        if not scored:
            return None
        return list(min(scored, key=lambda item: (item[0], _score_seq(item[1]), item[1]))[1])
    return list(solutions[0])


def _format_data_line(integer_tokens: list[str], float_tokens: list[str]) -> str:
    """Format a repaired row using the fixed-width ``fort.7`` columns."""
    # ReaxFF's traditional integer columns are five characters wide.  Keep
    # those starts for ordinary (<=4 digit) values, but reserve an explicit
    # separator before every field so a five-digit atom id cannot fuse with
    # its predecessor again.
    integers = f"{int(integer_tokens[0]):5d}" + "".join(
        f" {int(token):>4d}" for token in integer_tokens[1:]
    )
    floats = "".join(f" {token:>6}" for token in float_tokens)
    return integers + floats + "\n"


def _fix_data_line(line: str, n_bonds: int, frame: _GeometryFrame | None = None) -> tuple[str, str]:
    """Fix data line."""
    parts = line.split()
    float_index = _first_float_index(parts)
    if float_index == -1:
        return line, "skipped"
    left = parts[:float_index]
    right = parts[float_index:]
    if len(left) < 3:
        return line, "skipped"

    atom_index, atom_type = left[0], left[1]
    raw = left[2:]

    atom_type_was_fused = False
    if len(atom_type) > 1 and atom_type.isdigit():
        raw = [atom_type[1:]] + raw if atom_type[1:] else raw
        atom_type = atom_type[0]
        atom_type_was_fused = True

    if _is_valid_compact_row(raw, n_bonds):
        # Older versions of this repairer rewrote damaged rows with a simple
        # single-space join.  That made an incorrect guess look syntactically
        # valid on the next run, so it was copied unchanged forever.  Such
        # rows are recognizable by their normalized ``"atom type ..."``
        # prefix; discard their existing separators and resolve the compact
        # neighbor digit stream again using xmolout geometry.
        if not atom_type_was_fused and _LEGACY_REPAIRED_ROW_RE.match(line):
            body = raw[:-1]
            zeros_count = _trailing_zero_count(body)
            prefix = body[: len(body) - zeros_count]
            zeros = body[len(body) - zeros_count:]
            required = n_bonds - zeros_count
            try:
                parsed_atom_index = int(atom_index)
            except ValueError:
                return line, "unresolved"
            values = _choose_positive_prefix(
                prefix,
                required,
                parsed_atom_index,
                frame,
                trust_existing_fields=False,
            )
            if values is None:
                return line, "unresolved"
            rebuilt_left = [atom_index, atom_type] + [str(v) for v in values] + zeros + [raw[-1]]
            return _format_data_line(rebuilt_left, right), "fixed"

        # ``raw`` may only have become valid because the original atom-type
        # token contained both the one-digit type and the first neighbor id
        # (for example ``120594`` -> type 1, neighbor 20594).  Returning the
        # original line here used to discard that split and left the repaired
        # file unparseable even though it was reported as ``unchanged``.
        if atom_type_was_fused:
            rebuilt_left = [atom_index, atom_type] + raw
            return _format_data_line(rebuilt_left, right), "fixed"
        return line, "unchanged"

    if len(raw) < 1:
        return line, "unresolved"

    spacer = raw[-1]
    body = raw[:-1]
    zeros_count = _trailing_zero_count(body)
    prefix = body[: len(body) - zeros_count]
    zeros = body[len(body) - zeros_count:]

    required = n_bonds - zeros_count
    if required < 0:
        return line, "unresolved"

    try:
        parsed_atom_index = int(atom_index)
    except ValueError:
        return line, "unresolved"
    values = _choose_positive_prefix(prefix, required, parsed_atom_index, frame)
    if values is None:
        return line, "unresolved"

    rebuilt_left = [atom_index, atom_type] + [str(v) for v in values] + zeros + [spacer]
    return _format_data_line(rebuilt_left, right), "fixed"


class _XmoloutReader:
    """Small streaming reader used to avoid loading a large trajectory."""

    def __init__(self, source: TextIO) -> None:
        self.source = source
        self.buffered: _GeometryFrame | None = None

    def _next_frame(self) -> _GeometryFrame | None:
        for line in self.source:
            stripped = line.strip()
            if stripped.isdigit():
                n_atoms = int(stripped)
                break
        else:
            return None

        header = self.source.readline().split()
        if len(header) < 9:
            raise ValueError("Malformed xmolout frame header")
        iteration = int(header[1])
        lengths = tuple(float(value) for value in header[3:6])
        angles = tuple(float(value) for value in header[6:9])
        coordinates: list[tuple[float, float, float]] = []
        for _ in range(n_atoms):
            atom_line = self.source.readline()
            fields = atom_line.split()
            if len(fields) < 4:
                raise ValueError("Malformed or truncated xmolout atom block")
            coordinates.append((float(fields[1]), float(fields[2]), float(fields[3])))
        return _GeometryFrame(iteration, tuple(coordinates), _cell_matrix(lengths, angles))

    def frame_for_iteration(self, iteration: int) -> _GeometryFrame | None:
        """Return the matching frame, skipping older xmolout frames."""
        frame = self.buffered or self._next_frame()
        while frame is not None and frame.iteration < iteration:
            frame = self._next_frame()
        self.buffered = frame
        if frame is not None and frame.iteration == iteration:
            self.buffered = None
            return frame
        return None


def repair_fort7(
    input_file: str | Path = "fort.7",
    output_file: str | Path = "fort7_fixed",
    *,
    xmolout_file: str | Path = "xmolout",
    progress_every: int = 5000,
) -> dict[str, Any]:
    """Repair fort7.

    Parameters
    ----------
    input_file : str | Path, optional
        Input parameter.
    output_file : str | Path, optional
        Input parameter.
    xmolout_file : str | Path, optional
        Required trajectory coordinates and periodic cells used to resolve
        ambiguous atom-id splits. Defaults to ``xmolout`` in the current
        working directory.
    progress_every : int, optional
        Keyword-only parameter.

    Returns
    -------
    dict[str, Any]
        Return value.

    Examples
    --------
    ```python
    # Example
    repair_fort7(...)
    ```
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xmolout_path = Path(xmolout_file)
    if not xmolout_path.is_file():
        raise FileNotFoundError(f"required xmolout file not found: {xmolout_path}")

    current_bonds: int | None = None
    geometry_frame: _GeometryFrame | None = None
    lines = frames = fixed = unchanged = unresolved = skipped = 0

    xmolout_source = xmolout_path.open("r", encoding="utf-8")
    xmolout_reader = _XmoloutReader(xmolout_source)
    progress = tqdm(
        total=input_path.stat().st_size,
        desc="Repairing fort.7",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
        disable=False,
    )
    pending_progress_bytes = 0
    completed = False
    try:
        source = input_path.open("r", encoding="utf-8")
        sink = output_path.open("w", encoding="utf-8")
        with source, sink:
            for line in source:
                lines += 1
                pending_progress_bytes += len(line.encode("utf-8"))
                bonds_match = _BONDS_RE.search(line)
                if bonds_match:
                    current_bonds = int(bonds_match.group(1))
                    frames += 1
                    iteration_match = _ITERATION_RE.search(line)
                    if iteration_match is None:
                        raise ValueError(f"fort.7 frame {frames - 1} has no readable iteration number")
                    iteration = int(iteration_match.group(1))
                    geometry_frame = xmolout_reader.frame_for_iteration(iteration)
                    if geometry_frame is None:
                        raise ValueError(
                            f"xmolout has no frame matching fort.7 iteration {iteration}; "
                            "refusing to repair ambiguous atom ids without coordinates"
                        )
                    try:
                        fort7_n_atoms = int(line.split(maxsplit=1)[0])
                    except (ValueError, IndexError) as exc:
                        raise ValueError(f"fort.7 iteration {iteration} has no readable atom count") from exc
                    if fort7_n_atoms != len(geometry_frame.coordinates):
                        raise ValueError(
                            f"atom-count mismatch at iteration {iteration}: fort.7 has {fort7_n_atoms}, "
                            f"xmolout has {len(geometry_frame.coordinates)}"
                        )
                    sink.write(line)
                elif current_bonds is None:
                    skipped += 1
                    sink.write(line)
                else:
                    new_line, status = _fix_data_line(line, current_bonds, geometry_frame)
                    sink.write(new_line)
                    if status == "fixed":
                        fixed += 1
                    elif status == "unchanged":
                        unchanged += 1
                    elif status == "unresolved":
                        unresolved += 1
                    else:
                        skipped += 1

                if lines % max(1, progress_every) == 0:
                    progress.update(pending_progress_bytes)
                    pending_progress_bytes = 0
        completed = True
    finally:
        progress.update(pending_progress_bytes)
        if completed and progress.total is not None and progress.n < progress.total:
            # Universal-newline decoding can make character counts slightly
            # smaller than the on-disk byte total on Windows.
            progress.update(progress.total - progress.n)
        progress.close()
        xmolout_source.close()

    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "xmolout_file": str(xmolout_path),
        "lines": lines,
        "frames": frames,
        "fixed": fixed,
        "unchanged": unchanged,
        "unresolved": unresolved,
        "skipped": skipped,
    }
