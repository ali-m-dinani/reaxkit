from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reaxkit.engine.reaxff.io.control_handler import ControlHandler
from reaxkit.engine.reaxff.io.eregime_handler import EregimeHandler
from reaxkit.engine.reaxff.io.ffield_handler import FFieldHandler
from reaxkit.engine.reaxff.io.fort13_handler import Fort13Handler
from reaxkit.engine.reaxff.io.fort57_handler import Fort57Handler
from reaxkit.engine.reaxff.io.fort73_handler import Fort73Handler
from reaxkit.engine.reaxff.io.fort74_handler import Fort74Handler
from reaxkit.engine.reaxff.io.fort76_handler import Fort76Handler
from reaxkit.engine.reaxff.io.fort78_handler import Fort78Handler
from reaxkit.engine.reaxff.io.fort79_handler import Fort79Handler
from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler
from reaxkit.engine.reaxff.io.fort99_handler import Fort99Handler
from reaxkit.engine.reaxff.io.geo_handler import GeoHandler
from reaxkit.engine.reaxff.io.molfra_handler import MolFraHandler
from reaxkit.engine.reaxff.io.params_handler import ParamsHandler
from reaxkit.engine.reaxff.io.summary_handler import SummaryHandler
from reaxkit.engine.reaxff.io.trainset_handler import TrainsetHandler
from reaxkit.engine.reaxff.io.vels_handler import VelsHandler
from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler


REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLES_DIR = REPO_ROOT / "examples_to_test"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "artifacts" / "reaxff_io"


@dataclass(frozen=True)
class HandlerSpec:
    name: str
    handler_cls: type
    sample_path: Path


HANDLER_SPECS = [
    HandlerSpec("control", ControlHandler, EXAMPLES_DIR / "control"),
    HandlerSpec("eregime", EregimeHandler, EXAMPLES_DIR / "reaxkit_generated_inputs" / "eregime_func.in"),
    HandlerSpec("ffield", FFieldHandler, EXAMPLES_DIR / "ffield"),
    HandlerSpec("fort13", Fort13Handler, EXAMPLES_DIR / "fort.13"),
    HandlerSpec("fort57", Fort57Handler, EXAMPLES_DIR / "fort.57"),
    HandlerSpec("fort73", Fort73Handler, EXAMPLES_DIR / "fort.73"),
    HandlerSpec("fort74", Fort74Handler, EXAMPLES_DIR / "fort.74"),
    HandlerSpec("fort76", Fort76Handler, EXAMPLES_DIR / "fort.76"),
    HandlerSpec("fort78", Fort78Handler, EXAMPLES_DIR / "fort.78"),
    HandlerSpec("fort79", Fort79Handler, EXAMPLES_DIR / "fort.79"),
    HandlerSpec("fort7", Fort7Handler, EXAMPLES_DIR / "fort.7"),
    HandlerSpec("fort99", Fort99Handler, EXAMPLES_DIR / "fort.99"),
    HandlerSpec("geo", GeoHandler, EXAMPLES_DIR / "geo"),
    HandlerSpec("molfra", MolFraHandler, EXAMPLES_DIR / "molfra.out"),
    HandlerSpec("params", ParamsHandler, EXAMPLES_DIR / "params"),
    HandlerSpec("summary", SummaryHandler, EXAMPLES_DIR / "summary.txt"),
    HandlerSpec("trainset", TrainsetHandler, EXAMPLES_DIR / "trainset.in"),
    HandlerSpec("vels", VelsHandler, EXAMPLES_DIR / "vels"),
    HandlerSpec("xmolout", XmoloutHandler, EXAMPLES_DIR / "xmolout"),
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return {
            "rows": int(len(value)),
            "columns": [str(col) for col in value.columns],
        }
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_table(df: pd.DataFrame, out_base: Path) -> Path:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    try:
        xlsx_path = out_base.with_suffix(".xlsx")
        df.to_excel(xlsx_path, index=False)
        return xlsx_path
    except Exception:
        csv_path = out_base.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return path


def _frame_to_table(frame: Any) -> pd.DataFrame | None:
    if isinstance(frame, pd.DataFrame):
        return frame
    if isinstance(frame, dict):
        if "coords" in frame and "atom_types" in frame:
            coords = np.asarray(frame["coords"], dtype=float)
            atom_types = list(frame["atom_types"])
            if coords.ndim == 2 and coords.shape[1] == 3 and len(atom_types) == coords.shape[0]:
                data = {
                    "atom_type": atom_types,
                    "x": coords[:, 0],
                    "y": coords[:, 1],
                    "z": coords[:, 2],
                }
                extra_cols = {
                    key: value
                    for key, value in frame.items()
                    if key not in {"coords", "atom_types"}
                    and not isinstance(value, (dict, list, tuple, np.ndarray, pd.DataFrame, pd.Series))
                }
                for key, value in extra_cols.items():
                    data[str(key)] = [value] * coords.shape[0]
                return pd.DataFrame(data)
        flat = {
            str(key): value
            for key, value in frame.items()
            if not isinstance(value, (dict, list, tuple, np.ndarray, pd.DataFrame, pd.Series))
        }
        if flat:
            return pd.DataFrame([flat])
    return None


def _export_sections(handler: Any, out_dir: Path) -> list[str]:
    outputs: list[str] = []
    sections = getattr(handler, "sections", None)
    if not isinstance(sections, dict):
        return outputs
    for name, value in sections.items():
        if isinstance(value, pd.DataFrame):
            path = _write_table(value, out_dir / f"section_{name}")
            outputs.append(str(path))
    return outputs


def _export_named_tables(handler: Any, out_dir: Path) -> list[str]:
    outputs: list[str] = []
    table_methods = {
        "totals": "totals",
        "coordinates": "coordinates",
    }
    for file_stem, method_name in table_methods.items():
        method = getattr(handler, method_name, None)
        if callable(method):
            try:
                value = method()
            except Exception:
                continue
            if isinstance(value, pd.DataFrame):
                path = _write_table(value, out_dir / file_stem)
                outputs.append(str(path))
    return outputs


def _export_named_metadata(handler: Any, out_dir: Path) -> list[str]:
    outputs: list[str] = []
    metadata_methods = {
        "cell": "cell",
    }
    for file_stem, method_name in metadata_methods.items():
        method = getattr(handler, method_name, None)
        if callable(method):
            try:
                value = method()
            except Exception:
                continue
            if isinstance(value, dict):
                path = _write_json(out_dir / f"{file_stem}.json", value)
                outputs.append(str(path))
    return outputs


def _export_frames(handler: Any, out_dir: Path) -> list[str]:
    outputs: list[str] = []
    n_frames_method = getattr(handler, "n_frames", None)
    frame_method = getattr(handler, "frame", None)
    if not callable(n_frames_method) or not callable(frame_method):
        return outputs

    try:
        n_frames = int(n_frames_method())
    except Exception:
        return outputs
    if n_frames <= 0:
        return outputs

    summary_path = _write_text(out_dir / "frames_summary.txt", f"n_frames={n_frames}\n")
    outputs.append(str(summary_path))

    preview_indices = sorted({0, max(0, n_frames // 2), n_frames - 1})
    for frame_index in preview_indices:
        try:
            frame = frame_method(frame_index)
        except Exception:
            continue
        table = _frame_to_table(frame)
        if table is not None:
            path = _write_table(table, out_dir / f"frame_{frame_index:04d}")
            outputs.append(str(path))
        else:
            path = _write_json(out_dir / f"frame_{frame_index:04d}.json", {"frame": frame})
            outputs.append(str(path))
    return outputs


def generate_handler_artifacts(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, dict[str, Any]] = {}
    for spec in HANDLER_SPECS:
        handler = spec.handler_cls(spec.sample_path)
        handler_dir = output_dir / spec.name
        handler_dir.mkdir(parents=True, exist_ok=True)

        df = handler.dataframe()
        metadata = handler.metadata()

        exported_files = [
            str(_write_table(df, handler_dir / "dataframe")),
            str(_write_json(handler_dir / "metadata.json", metadata)),
        ]
        exported_files.extend(_export_sections(handler, handler_dir))
        exported_files.extend(_export_named_tables(handler, handler_dir))
        exported_files.extend(_export_named_metadata(handler, handler_dir))
        exported_files.extend(_export_frames(handler, handler_dir))

        index[spec.name] = {
            "handler_class": spec.handler_cls.__name__,
            "sample_path": str(spec.sample_path),
            "row_count": int(len(df)),
            "columns": [str(col) for col in df.columns],
            "exported_files": exported_files,
        }

    _write_json(output_dir / "index.json", index)
    return index


if __name__ == "__main__":
    generate_handler_artifacts()
