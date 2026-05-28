"""Extract and filter parsed control-parameter sections as analyzer outputs.

This module exposes analyzer tasks over parsed `control` configuration blocks,
including section-level selection and key/value filtering for reporting.
It is intentionally limited to control-parameter content and does not read
raw files directly.

**Usage context**

- Input auditing: Inspect effective control settings used for a run.
- Section-level review: Pull only `general`, `md`, `mm`, `ff`, or `outdated`.
- Reproducibility logs: Export normalized control tables for reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ControlParametersData
from reaxkit.presentation.specs import PresentationSpec

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


def _resolve_key_section(data: ControlParametersData, key: str, section: Optional[str]) -> Optional[str]:
    """Resolve which section contains a key when section is not explicitly requested."""
    if section is not None:
        return str(section).lower()
    key_norm = str(key).lower()
    section_map = _section_map(data)
    for section_name in _SECTION_NAMES:
        if key_norm in section_map[section_name]:
            return section_name
    return None


@dataclass
class ControlParametersTaskRequest(BaseRequest):
    """Request for extracting a control parameter as a table row.

    Fields
    -----
    key : str
        Control key to query (case-insensitive).
    section : Optional[str]
        Optional section to search. If `None`, sections are searched in default
        order: `general -> md -> mm -> ff -> outdated`.

    Examples
    -----
    ```python
    req = ControlParametersTaskRequest(key="nmdit", section="md")
    ```
    Sample output:
    `ControlParametersTaskRequest(...)`
    Meaning:
    The request asks for one control key lookup in a selected section.
    """

    key: str = dc_field(
        metadata={
            "label": "Key",
            "help": (
                "Control key to retrieve (case-insensitive). "
                "Examples: 'nmdit', 'iout2', 'imetho'."
            ),
        },
    )
    section: Optional[str] = dc_field(
        default=None,
        metadata={
            "label": "Section",
            "help": (
                "Optional control section to search. "
                "If omitted, all sections are searched in default order. "
                "Example: 'md'."
            ),
            "choices": ["general", "md", "mm", "ff", "outdated"],
        },
    )


@dataclass
class ControlParametersTaskResult(BaseResult):
    """Result for a control-parameter table extraction.

    Fields
    -----
    table : pd.DataFrame
        One-row table with columns `key`, `value`, `section`, and `found`.
    request : ControlParametersTaskRequest
        Request object used for this lookup.

    Examples
    -----
    ```python
    result = ControlParametersTask().run(data, req)
    result.table
    ```
    Sample output:
    One-row DataFrame (for example: `key='iout2', value=20, section='md', found=True`).
    Meaning:
    The result captures resolved value and provenance for one key lookup.
    """

    table: pd.DataFrame
    request: ControlParametersTaskRequest


@register_task("get_control_data", label="Control Value")
class ControlParametersTask(AnalysisTask):
    """Return one control parameter as a one-row table."""

    required_data = ControlParametersData

    @staticmethod
    def recommended_presentations(
        _result: ControlParametersTaskResult,
        _payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        """Return default table presentation for control lookup outputs.

        Works on
        -----
        Analyzer task output payloads

        Parameters
        -----
        _result : ControlParametersTaskResult
            Analysis result object for the executed task.
        _payload : dict[str, Any]
            Serialized result payload.

        Returns
        -----
        list[PresentationSpec]
            Table presentation specification.

        Examples
        -----
        ```python
        specs = ControlParametersTask.recommended_presentations(result, payload)
        ```
        Sample output:
        `[PresentationSpec(renderer="table", ...)]`
        Meaning:
        Control lookup outputs default to tabular rendering.
        """
        return [PresentationSpec(renderer="table", label="Table", view_type="table")]

    def run(
        self,
        data: ControlParametersData,
        request: ControlParametersTaskRequest,
        reporter=None,
    ) -> ControlParametersTaskResult:
        """Resolve one control key and return a one-row result table.

        Works on
        -----
        `ControlParametersData` plus `ControlParametersTaskRequest` inputs

        Parameters
        -----
        data : ControlParametersData
            Parsed control-parameter sections.
        request : ControlParametersTaskRequest
            Key/section lookup configuration.
        reporter : Any, optional
            Unused progress callback parameter for task API compatibility.

        Returns
        -----
        ControlParametersTaskResult
            One-row lookup result with value, section, and found flag.

        Examples
        -----
        ```python
        result = ControlParametersTask().run(data, ControlParametersTaskRequest(key="iout2"))
        ```
        Sample output:
        `result.table` with columns `key`, `value`, `section`, `found`.
        Meaning:
        A single control entry is normalized into table form.
        """
        missing = object()
        value = _get_control_data(
            data,
            key=request.key,
            section=request.section,
            default=missing,
        )
        found = value is not missing
        resolved_value = None if not found else value
        resolved_section = _resolve_key_section(data, key=request.key, section=request.section)

        table = pd.DataFrame(
            [
                {
                    "key": str(request.key).lower(),
                    "value": resolved_value,
                    "section": resolved_section,
                    "found": bool(found),
                }
            ]
        )
        return ControlParametersTaskResult(
            table=table,
            request=request,
        )


__all__ = [
    "ControlParametersTaskRequest",
    "ControlParametersTaskResult",
    "ControlParametersTask",
]
