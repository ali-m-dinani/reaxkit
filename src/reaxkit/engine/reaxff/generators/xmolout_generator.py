"""
XMOL trajectory file generators.

This module provides utilities for generating new ReaxFF ``xmolout`` files
from in-memory trajectory data or from an existing ``XmoloutHandler``.
Generated files are fully compatible with downstream ReaxKit analyses.

**Usage context**

- Template generation: Produce canonical text payloads for ReaxFF artifacts.
- File writing: Persist generated outputs to disk with stable formatting.
- Workflow integration: Support higher-level ReaxKit workflow commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler


FrameSel = Optional[Union[Sequence[int], range, slice]]
AtomSel = Optional[Union[Sequence[int], slice]]


__all__ = [
    "FrameSel",
    "AtomSel",
    "XmoloutFromHandlerSpec",
    "XmoloutFromFramesSpec",
    "XMOL_GENERATOR_REGISTRY",
    "trim_xmolout",
]


@dataclass(frozen=True)
class XmoloutFromHandlerSpec:
    """Represent XmoloutFromHandlerSpec.

    Public class used by ReaxFF generator components.

    Fields
    ------
    xh : XmoloutHandler
        Dataclass field.
    frames : FrameSel
        Dataclass field.
    atoms : AtomSel
        Dataclass field.
    atom_types : Optional[Sequence[str]]
        Dataclass field.
    simulation_name : Optional[str]
        Dataclass field.
    precision : int
        Dataclass field.
    include_extras : Union[bool, Sequence[str], str]
        Dataclass field.
    """
    xh: XmoloutHandler
    frames: FrameSel = None
    atoms: AtomSel = None
    atom_types: Optional[Sequence[str]] = None
    simulation_name: Optional[str] = None
    precision: int = 6
    include_extras: Union[bool, Sequence[str], str] = False


@dataclass(frozen=True)
class XmoloutFromFramesSpec:
    """Represent XmoloutFromFramesSpec.

    Public class used by ReaxFF generator components.

    Fields
    ------
    frames : tuple[Dict[str, Any], ...]
        Dataclass field.
    simulation_name : str
        Dataclass field.
    precision : int
        Dataclass field.
    defaults : Dict[str, float]
        Dataclass field.
    """
    frames: tuple[Dict[str, Any], ...]
    simulation_name: str = "MD"
    precision: int = 6
    defaults: Dict[str, float] = field(
        default_factory=lambda: {
            "E_pot": 0.0,
            "a": 1.0,
            "b": 1.0,
            "c": 1.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        }
    )


def _normalize_frames_for_write(xh: XmoloutHandler, frames: FrameSel) -> list[int]:
    """Normalize frames for write."""
    n = xh.n_frames()
    if frames is None:
        return list(range(n))
    if isinstance(frames, slice):
        return list(range(*frames.indices(n)))
    return [int(i) for i in frames if 0 <= int(i) < n]


def _normalize_atoms_for_write(
    frame_dict: Dict[str, Any],
    atoms: AtomSel,
    atom_types: Optional[Sequence[str]],
) -> list[int]:
    """Normalize atoms for write."""
    n_atoms = frame_dict["coords"].shape[0]
    if atoms is not None:
        if isinstance(atoms, slice):
            return list(range(*atoms.indices(n_atoms)))
        return [int(a) for a in atoms if 0 <= int(a) < n_atoms]
    if atom_types:
        tset = {str(t) for t in atom_types}
        return [j for j, t in enumerate(frame_dict["atom_types"]) if str(t) in tset]
    return list(range(n_atoms))


def _format_header_line(
    sim_name: str,
    iteration: int,
    energy_pot: float,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    precision: int,
) -> str:
    """Format header line."""
    fmt = f"{{:.{precision}f}}"
    return (
        f"{sim_name} {iteration} "
        f"{fmt.format(energy_pot)} {fmt.format(a)} {fmt.format(b)} {fmt.format(c)} "
        f"{fmt.format(alpha)} {fmt.format(beta)} {fmt.format(gamma)}\n"
    )


def _safe_get(row: pd.Series, key: str, default: float = 0.0) -> float:
    """Safe get."""
    return float(row[key]) if (isinstance(row, pd.Series) and key in row and pd.notna(row[key])) else float(default)


def _get_frame_table(xh: XmoloutHandler, index: int) -> pd.DataFrame:
    """Get frame table."""
    if hasattr(xh, "_frames") and index < len(xh._frames):
        return xh._frames[index]
    frame = xh.frame(index)
    return pd.DataFrame(
        {
            "atom_type": frame["atom_types"],
            "x": frame["coords"][:, 0],
            "y": frame["coords"][:, 1],
            "z": frame["coords"][:, 2],
        }
    )


def _format_atom_line_extended(row: pd.Series, precision: int, extra_order: list[str] | None) -> str:
    """Format atom line extended."""
    fmt = f"{{:.{precision}f}}"
    atom_type = str(row["atom_type"])
    x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
    parts = [f"{atom_type:<3}", fmt.format(x), fmt.format(y), fmt.format(z)]
    if extra_order:
        for col in extra_order:
            val = row.get(col, np.nan)
            if isinstance(val, (int, float, np.floating, np.integer)) or pd.isna(val):
                parts.append(fmt.format(float(val)) if pd.notna(val) else "nan")
            else:
                parts.append(str(val))
    return " ".join(parts) + "\n"


def _generate_xmolout_from_handler(spec: XmoloutFromHandlerSpec) -> str:
    """
    Generate filtered ``xmolout`` text from an existing ``XmoloutHandler``.
    """
    xh = spec.xh
    df = xh.dataframe()
    sim_name = spec.simulation_name or getattr(xh, "simulation_name", None) or "MD"
    frame_indices = _normalize_frames_for_write(xh, spec.frames)
    lines: list[str] = []

    for i in frame_indices:
        frame = xh.frame(i)
        selected_atoms = _normalize_atoms_for_write(frame, spec.atoms, spec.atom_types)

        row = df.iloc[i] if i < len(df) else pd.Series()
        iteration = int(frame.get("iter", int(row["iter"]) if "iter" in row else i))
        energy_pot = _safe_get(row, "E_pot", 0.0)
        a = _safe_get(row, "a", 1.0)
        b = _safe_get(row, "b", 1.0)
        c = _safe_get(row, "c", 1.0)
        alpha = _safe_get(row, "alpha", 90.0)
        beta = _safe_get(row, "beta", 90.0)
        gamma = _safe_get(row, "gamma", 90.0)

        lines.append(f"{len(selected_atoms)}\n")
        lines.append(
            _format_header_line(
                sim_name,
                iteration,
                energy_pot,
                a,
                b,
                c,
                alpha,
                beta,
                gamma,
                spec.precision,
            )
        )

        frame_tbl = _get_frame_table(xh, i).reset_index(drop=True)
        base_cols = ["atom_type", "x", "y", "z"]
        present_cols = frame_tbl.columns.tolist()
        extra_cols_default = [col for col in present_cols if col not in base_cols]

        if spec.include_extras is True or (isinstance(spec.include_extras, str) and spec.include_extras.lower() == "all"):
            extra_order = extra_cols_default
        elif isinstance(spec.include_extras, (list, tuple)):
            extra_order = [col for col in spec.include_extras if col in present_cols and col not in base_cols]
        else:
            extra_order = []

        needed_cols = base_cols + [col for col in extra_order if col not in base_cols]
        subset = frame_tbl.loc[selected_atoms, [col for col in needed_cols if col in frame_tbl.columns]].copy()
        for col in needed_cols:
            if col not in subset.columns:
                subset[col] = np.nan

        for _, row_data in subset[base_cols + extra_order].iterrows():
            lines.append(_format_atom_line_extended(row_data, spec.precision, extra_order))

    return "".join(lines)


def _write_xmolout(
    out_path: str | Path,
    text: str,
) -> Path:
    """
    Write generated ``xmolout`` text to disk.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _write_xmolout_from_handler(
    xh: XmoloutHandler,
    out_path: str | Path,
    *,
    frames: FrameSel = None,
    atoms: AtomSel = None,
    atom_types: Optional[Sequence[str]] = None,
    simulation_name: Optional[str] = None,
    precision: int = 6,
    include_extras: Union[bool, Sequence[str], str] = False,
) -> Path:
    """
    Backward-compatible wrapper for writing filtered ``xmolout`` from a handler.
    """
    spec = XmoloutFromHandlerSpec(
        xh=xh,
        frames=frames,
        atoms=atoms,
        atom_types=atom_types,
        simulation_name=simulation_name,
        precision=precision,
        include_extras=include_extras,
    )
    return _write_xmolout(out_path, _generate_xmolout_from_handler(spec))


def trim_xmolout(
    input_file: str | Path = "xmolout",
    out_path: str | Path = "xmolout_trimmed",
    *,
    frames: FrameSel = None,
    atoms: AtomSel = None,
    atom_types: Optional[Sequence[str]] = None,
    simulation_name: Optional[str] = None,
    precision: int = 6,
) -> Path:
    """Trim xmolout.

    Parameters
    ----------
    input_file : str | Path, optional
        Input parameter.
    out_path : str | Path, optional
        Input parameter.
    frames : FrameSel, optional
        Keyword-only parameter.
    atoms : AtomSel, optional
        Keyword-only parameter.
    atom_types : Optional[Sequence[str]], optional
        Keyword-only parameter.
    simulation_name : Optional[str], optional
        Keyword-only parameter.
    precision : int, optional
        Keyword-only parameter.

    Returns
    -------
    Path
        Return value.

    Examples
    --------
    ```python
    # Example
    trim_xmolout(...)
    ```
    """
    xh = XmoloutHandler(input_file)
    return _write_xmolout_from_handler(
        xh,
        out_path,
        frames=frames,
        atoms=atoms,
        atom_types=atom_types,
        simulation_name=simulation_name,
        precision=precision,
        include_extras=False,
    )


def _generate_xmolout_from_frames(spec: XmoloutFromFramesSpec) -> str:
    """
    Generate ``xmolout`` text from explicit per-frame dictionaries.
    """
    lines: list[str] = []
    for frame in spec.frames:
        coords = np.asarray(frame["coords"], dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("Each frame['coords'] must be an (n,3) array-like.")

        atom_types = [str(atom_type) for atom_type in frame["atom_types"]]
        if len(atom_types) != coords.shape[0]:
            raise ValueError("Length of frame['atom_types'] must match number of coordinate rows.")

        iteration = int(frame["iter"])
        energy_pot = float(frame.get("E_pot", spec.defaults["E_pot"]))
        a = float(frame.get("a", spec.defaults["a"]))
        b = float(frame.get("b", spec.defaults["b"]))
        c = float(frame.get("c", spec.defaults["c"]))
        alpha = float(frame.get("alpha", spec.defaults["alpha"]))
        beta = float(frame.get("beta", spec.defaults["beta"]))
        gamma = float(frame.get("gamma", spec.defaults["gamma"]))

        extras_dict: Dict[str, Any] = dict(frame.get("extras", {}))
        extra_keys = list(extras_dict.keys())
        extras_cols: list[np.ndarray] = []
        for key in extra_keys:
            arr = np.asarray(extras_dict[key])
            if arr.shape[0] != coords.shape[0]:
                raise ValueError(f"extras['{key}'] length must match number of atoms.")
            extras_cols.append(arr)

        lines.append(f"{coords.shape[0]}\n")
        lines.append(
            _format_header_line(
                spec.simulation_name,
                iteration,
                energy_pot,
                a,
                b,
                c,
                alpha,
                beta,
                gamma,
                spec.precision,
            )
        )

        fmt = f"{{:.{spec.precision}f}}"
        for idx, (atom_type, (x, y, z)) in enumerate(zip(atom_types, coords)):
            parts = [f"{atom_type:<3}", fmt.format(x), fmt.format(y), fmt.format(z)]
            for col in extras_cols:
                value = col[idx]
                if isinstance(value, (int, float, np.floating, np.integer)) or (
                    isinstance(value, str) and value.replace(".", "", 1).isdigit()
                ):
                    parts.append(fmt.format(float(value)))
                else:
                    parts.append(str(value))
            lines.append(" ".join(parts) + "\n")

    return "".join(lines)


def _write_xmolout_from_frames(
    frames: Iterable[Dict[str, Any]],
    out_path: str | Path,
    *,
    simulation_name: str = "MD",
    precision: int = 6,
    defaults: Dict[str, float] | None = None,
) -> Path:
    """
    Backward-compatible wrapper for writing ``xmolout`` from explicit frames.
    """
    spec = XmoloutFromFramesSpec(
        frames=tuple(frames),
        simulation_name=simulation_name,
        precision=precision,
        defaults=defaults
        or {
            "E_pot": 0.0,
            "a": 1.0,
            "b": 1.0,
            "c": 1.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
    )
    return _write_xmolout(out_path, _generate_xmolout_from_frames(spec))


XMOL_GENERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    "xmolout_from_handler": {
        "label": "XMOL Trajectory From Handler",
        "default_filename": "xmolout",
        "spec_type": XmoloutFromHandlerSpec,
        "generate": _generate_xmolout_from_handler,
        "write": _write_xmolout_from_handler,
    },
    "xmolout_from_frames": {
        "label": "XMOL Trajectory From Frames",
        "default_filename": "xmolout",
        "spec_type": XmoloutFromFramesSpec,
        "generate": _generate_xmolout_from_frames,
        "write": _write_xmolout_from_frames,
    },
}
