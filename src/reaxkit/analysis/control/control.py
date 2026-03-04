"""Control-parameter analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ControlParametersData

_SECTION_NAMES = ("general", "md", "mm", "ff", "outdated")


def _section_map(data: ControlParametersData) -> dict[str, dict]:
    return {
        "general": data.general,
        "md": data.md,
        "mm": data.mm,
        "ff": data.ff,
        "outdated": data.outdated,
    }


def _get_available_sections(data: ControlParametersData) -> list[str]:
    """List the available control-parameter sections."""
    section_map = _section_map(data)
    return [name for name in _SECTION_NAMES if section_map[name]]


def _get_available_keys(data: ControlParametersData, section: Optional[str] = None) -> list[str]:
    """List available control keys, optionally limited to one section."""
    section_map = _section_map(data)

    if section:
        section = section.lower()
        if section not in section_map:
            raise ValueError(f"Unknown section: {section}")
        return sorted(section_map[section].keys())

    all_keys = set()
    for values in section_map.values():
        all_keys.update(values.keys())
    return sorted(all_keys)


def _get_control_data(
    data: ControlParametersData,
    key: str,
    section: Optional[str] = None,
    default=None,
):
    """Retrieve one control-parameter value by key."""
    key = str(key).lower()
    section_map = _section_map(data)

    if section:
        section = section.lower()
        if section not in section_map:
            raise ValueError(f"Unknown section: {section}")
        return section_map[section].get(key, default)

    for section_name in _SECTION_NAMES:
        values = section_map[section_name]
        if key in values:
            return values[key]
    return default


@dataclass
class ControlValueRequest(BaseRequest):
    """Request for a single control-parameter lookup."""

    key: str
    section: Optional[str] = None
    default: object = None


@dataclass
class ControlValueResult(BaseResult):
    """Single control-parameter lookup result."""

    key: str
    value: object
    section: Optional[str] = None
    found: bool = False


@register_task("control_value")
class ControlValueTask(AnalysisTask):
    """Return the value of a control parameter of interest."""

    required_data = ControlParametersData

    def run(
        self,
        data: ControlParametersData,
        request: ControlValueRequest,
        reporter=None,
    ) -> ControlValueResult:
        missing = object()
        value = _get_control_data(
            data,
            key=request.key,
            section=request.section,
            default=missing,
        )
        return ControlValueResult(
            key=str(request.key),
            value=request.default if value is missing else value,
            section=request.section,
            found=value is not missing,
        )


__all__ = [
    "ControlValueRequest",
    "ControlValueResult",
    "ControlValueTask",
]
