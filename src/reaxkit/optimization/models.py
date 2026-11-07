"""Versioned records shared by converters, workers, and future interfaces."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

from . import SCHEMA_VERSION


class OptimizationError(ValueError):
    """An invalid configuration, unsupported format, or unsafe lifecycle action."""


def finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(str(value).replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def identity(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def progress(source: str, record: Any, score: Any = None, **values: Any) -> dict:
    result = {
        "schema_version": SCHEMA_VERSION,
        "event_id": identity(source, record),
        "source": source,
        "source_record": record,
        "native_step": None,
        "step_kind": "checkpoint",
        "candidate_id": None,
        "candidate_evaluations": None,
        "engine_calls": None,
        "elapsed_seconds": None,
        "score": finite(score),
        "raw_score": str(score) if score is not None and finite(score) is None else None,
        "objectives": None,
        "best_score": None,
        "status": "ok" if finite(score) is not None else "unavailable" if score is None else "nonfinite",
    }
    result.update(values)
    return result


def prediction(source: str, candidate: str, entry: str, reference: Any,
               predicted: Any, **values: Any) -> dict:
    result = {
        "schema_version": SCHEMA_VERSION,
        "event_id": identity(source, candidate, entry),
        "candidate_id": candidate,
        "training_entry_id": identity(entry),
        "identity_scope": "source",
        "entry_label": entry,
        "property": None,
        "unit": None,
        "reference": finite(reference),
        "prediction": finite(predicted),
        "weight": None,
        "weight_semantics": None,
        "dataset_split": "train",
        "source": source,
        "status": "ok" if finite(reference) is not None and finite(predicted) is not None else "nonfinite",
    }
    result.update(values)
    return result


@dataclass
class Conversion:
    progress: list[dict] = field(default_factory=list)
    predictions: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sources: dict[str, dict] = field(default_factory=dict)
