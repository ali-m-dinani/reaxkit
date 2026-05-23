"""Utilities to repair corrupted ReaxFF ``fort.7`` atom rows."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any


_BONDS_RE = re.compile(r"#Bonds:\s*(\d+)")
_MAX_DIGITS = 5

__all__ = ["repair_fort7"]


def _first_float_index(parts: list[str]) -> int:
    for i, part in enumerate(parts):
        if "." in part or "e" in part.lower():
            return i
    return -1


def _is_valid_compact_row(raw_tokens: list[str], n_bonds: int) -> bool:
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
    zeros = 0
    for token in reversed(tokens):
        if token == "0":
            zeros += 1
        else:
            break
    return zeros


def _score_seq(seq: tuple[int, ...]) -> int:
    return sum(0 if len(str(x)) >= 4 else (4 - len(str(x))) * 3 for x in seq)


def _parse_positive_prefix(prefix_tokens: list[str], required: int) -> list[int] | None:
    if len(prefix_tokens) == required and all(1 <= len(t) <= _MAX_DIGITS for t in prefix_tokens):
        values = [int(t) for t in prefix_tokens]
        valid = True
        previous = 0
        for value in values:
            if value <= previous:
                valid = False
                break
            previous = value
        if valid:
            return values

    compact = "".join(prefix_tokens)
    n_chars = len(compact)
    if required == 0:
        return [] if n_chars == 0 else None
    if n_chars < required or n_chars > required * _MAX_DIGITS:
        return None

    @lru_cache(None)
    def _rec(position: int, used: int, previous: int) -> list[tuple[int, ...]]:
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
            chunk = compact[position : position + chunk_len]
            if len(chunk) > 1 and chunk[0] == "0":
                continue
            value = int(chunk)
            if value <= previous:
                continue
            for tail in _rec(position + chunk_len, used + 1, value):
                solutions.append((value,) + tail)
        return solutions

    solutions = _rec(0, 0, 0)
    if not solutions:
        return None
    best = min(solutions, key=lambda seq: (_score_seq(seq), seq))
    return list(best)


def _fix_data_line(line: str, n_bonds: int) -> tuple[str, str]:
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

    if len(atom_type) > 1 and atom_type.isdigit():
        raw = [atom_type[1:]] + raw if atom_type[1:] else raw
        atom_type = atom_type[0]

    if _is_valid_compact_row(raw, n_bonds):
        return line, "unchanged"

    if len(raw) < 1:
        return line, "unresolved"

    spacer = raw[-1]
    body = raw[:-1]
    zeros_count = _trailing_zero_count(body)
    prefix = body[: len(body) - zeros_count]
    zeros = body[len(body) - zeros_count :]

    required = n_bonds - zeros_count
    if required < 0:
        return line, "unresolved"

    values = _parse_positive_prefix(prefix, required)
    if values is None:
        return line, "unresolved"

    rebuilt_left = [atom_index, atom_type] + [str(v) for v in values] + zeros + [spacer]
    return " ".join(rebuilt_left + right) + "\n", "fixed"


def repair_fort7(
    input_file: str | Path = "fort.7",
    output_file: str | Path = "fort7_fixed",
    *,
    progress_every: int = 5000,
) -> dict[str, Any]:
    """Repair line-format corruption in ``fort.7`` atom rows and write a corrected file."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    current_bonds: int | None = None
    lines = frames = fixed = unchanged = unresolved = skipped = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for line in source:
            lines += 1
            bonds_match = _BONDS_RE.search(line)
            if bonds_match:
                current_bonds = int(bonds_match.group(1))
                frames += 1
                sink.write(line)
            elif current_bonds is None:
                skipped += 1
                sink.write(line)
            else:
                new_line, status = _fix_data_line(line, current_bonds)
                sink.write(new_line)
                if status == "fixed":
                    fixed += 1
                elif status == "unchanged":
                    unchanged += 1
                elif status == "unresolved":
                    unresolved += 1
                else:
                    skipped += 1

            _ = progress_every

    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "lines": lines,
        "frames": frames,
        "fixed": fixed,
        "unchanged": unchanged,
        "unresolved": unresolved,
        "skipped": skipped,
    }
