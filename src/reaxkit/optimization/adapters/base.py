"""Shared readers that preserve source provenance and incomplete output."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

from ..models import Conversion, OptimizationError, finite, prediction, progress
from ..storage import digest

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?|[-+]?(?:inf|nan)"


class Reader:
    def __init__(self, root, options=None):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise OptimizationError(f"Native output directory not found: {self.root}")
        self.options = options or {}
        self.result = Conversion()

    def relative(self, path):
        path = Path(path).resolve()
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError as error:
            raise OptimizationError(f"Output source escapes native directory: {path}") from error

    def read(self, path):
        path = Path(path)
        name = self.relative(path)
        try:
            data = path.read_bytes()
        except OSError as error:
            self.result.warnings.append(f"{name}: unreadable: {error}")
            return b""
        self.result.sources[name] = {
            "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        }
        return data

    def lines(self, path):
        data = self.read(path).decode("utf-8", errors="replace")
        lines = data.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            self.result.warnings.append(f"{self.relative(path)}: trailing partial line deferred")
            lines.pop()
        return [(index, line.rstrip("\r\n")) for index, line in enumerate(lines, 1)]

    def json(self, path):
        try:
            return json.loads(self.read(path))
        except (ValueError, UnicodeError) as error:
            self.result.warnings.append(f"{self.relative(path)}: incomplete/invalid JSON: {error}")
            return None

    def csv(self, path):
        lines = self.lines(path)
        try:
            return list(csv.DictReader(io.StringIO("\n".join(line for _, line in lines))))
        except csv.Error as error:
            self.result.warnings.append(f"{self.relative(path)}: malformed CSV: {error}")
            return []

    def artifact(self, path, restart="neither"):
        if path.is_file():
            name = self.relative(path)
            self.result.artifacts.append({
                "path": name, "sha256": digest(path), "restart": restart,
            })

    def npz(self, path):
        import numpy as np

        try:
            data = self.read(path)
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                return {key: archive[key].tolist() for key in archive.files}
        except (ValueError, OSError, EOFError, zipfile.BadZipFile) as error:
            self.result.warnings.append(f"{self.relative(path)}: incomplete/unsupported NPZ: {error}")
            return None

    def configured_progress(self, default_name, step_kind):
        path = self.root / self.options.get("progress_file", default_name)
        if not path.is_file():
            return
        columns = self.options.get("columns")
        if not columns:
            self.read(path)
            self.result.warnings.append(
                f"{path.name}: supply conversion.columns with zero-based step/score indices for this engine version; columns are not guessed"
            )
            return
        for row, line in self.lines(path):
            if row <= self.options.get("skip_rows", 0) or not line.strip() or line.lstrip().startswith(("#", "%")):
                continue
            values = line.split()
            if max(columns.values()) >= len(values):
                self.result.warnings.append(f"{path.name}:{row}: insufficient columns")
                continue
            selected = {key: finite(values[index]) for key, index in columns.items()}
            if selected["step"] is None:
                self.result.warnings.append(f"{path.name}:{row}: invalid step")
                continue
            self.result.progress.append(progress(
                self.relative(path), row, selected["score"],
                native_step=selected["step"], step_kind=self.options.get("step_kind", step_kind),
                best_score=selected.get("best_score"), elapsed_seconds=selected.get("elapsed_seconds"),
                status="ok" if selected["score"] is not None else "nonfinite",
                raw_record=line,
            ))


def jax_reports(reader, paths):
    pattern = re.compile(r"^(.*?)\s+(" + NUMBER + r")\s+(" + NUMBER + r")\s+(" + NUMBER + r")\s+(" + NUMBER + r")\s+(" + NUMBER + r")\s*$", re.I)
    units = {"ENERGY": "kcal/mol", "CHARGE": "e", "FORCE": "kcal/mol/angstrom",
             "DISTANCE": "angstrom", "ANGLE": "degree", "TORSION": "degree"}
    for path in sorted(paths):
        source = reader.relative(path)
        counts = {}
        for row, line in reader.lines(path):
            match = pattern.match(line)
            if not match or ":" not in match[1]:
                continue
            label = match[1].strip()
            counts[label] = counts.get(label, 0) + 1
            property_name = label.split(":", 1)[0]
            reader.result.predictions.append(prediction(
                source, source, f"{label}#{counts[label]}", match[3], match[4],
                property=property_name, unit=units.get(property_name),
                weight=finite(match[2]), weight_semantics="accuracy_divisor",
                dataset_split="validation" if path.name.startswith("valid_") else "train",
                source_record=row, precision="native_report_2_decimals",
                weighted_squared_error=finite(match[5]),
            ))
