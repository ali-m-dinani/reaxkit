"""
Control file analysis utilities.

This module provides helper functions for querying and retrieving
parameters from a parsed ReaxFF control file via `ControlHandler`.

Typical use cases include:

    - discovering available parameter sections
    - listing available control keys
    - retrieving parameter values with safe defaults
"""


from __future__ import annotations
from typing import Optional

from reaxkit.io.handlers.control_handler import ControlHandler


def _get_available_sections(handler: ControlHandler) -> list[str]:
    """List the available parameter sections in a ReaxFF control file.

    Works on
    --------
    ControlHandler (control)

    Parameters
    ----------
    handler : ControlHandler
        Parsed control file handler.

    Returns
    -------
    list[str]
        Section names present in the file (e.g., "general", "md", "mm", "ff", "outdated").

    Examples
    --------
    >>> from reaxkit.io.handlers.control_handler import ControlHandler
    >>> from reaxkit.analysis.per_file.control_analyzer import _get_available_sections
    >>> h = ControlHandler("control")
    >>> sections = _get_available_sections(h)
    """
    sections = []
    if handler.general_parameters:
        sections.append("general")
    if handler.md_parameters:
        sections.append("md")
    if handler.mm_parameters:
        sections.append("mm")
    if handler.ff_parameters:
        sections.append("ff")
    if handler.outdated_parameters:
        sections.append("outdated")
    return sections


def _get_available_keys(handler: ControlHandler, section: Optional[str] = None) -> list[str]:
    """List available control-file parameter keys, optionally within a specific section.

    Works on
    --------
    ControlHandler (control)

    Parameters
    ----------
    handler : ControlHandler
        Parsed control file handler.
    section : str, optional
        Section to query (one of: "general", "md", "mm", "ff", "outdated").
        If None, keys from all sections are combined.

    Returns
    -------
    list[str]
        Sorted list of parameter keys.

    Examples
    --------
    >>> from reaxkit.io.handlers.control_handler import ControlHandler
    >>> from reaxkit.analysis.per_file.control_analyzer import _get_available_keys
    >>> h = ControlHandler("control")
    >>> keys = _get_available_keys(h, section="md")
    """
    section_map = {
        "general": handler.general_parameters,
        "md": handler.md_parameters,
        "mm": handler.mm_parameters,
        "ff": handler.ff_parameters,
        "outdated": handler.outdated_parameters,
    }

    if section:
        section = section.lower()
        if section not in section_map:
            raise ValueError(f"❌ Unknown section: {section}")
        return sorted(section_map[section].keys())

    # all keys from all sections
    all_keys = set()
    for d in section_map.values():
        all_keys.update(d.keys())
    return sorted(all_keys)


def get_control_data(
    handler: ControlHandler,
    key: str,
    section: Optional[str] = None,
    default=None,
):
    """
    Retrieve a control-file parameter value by key, optionally restricted to a section.

    Works on
    --------
    ControlHandler (control)

    Parameters
    ----------
    handler : ControlHandler
        Parsed control file handler.
    key : str
        Parameter name (case-insensitive).
    section : str, optional
        Section to search (one of: "general", "md", "mm", "ff", "outdated").
        If None, all sections are searched in order.
    default : any, optional
        Value to return if the key is not found.

    Returns
    -------
    any
        The parameter value if found; otherwise `default`.

    Examples
    --------
    >>> from reaxkit.io.handlers.control_handler import ControlHandler
    >>> from reaxkit.analysis.per_file.control_analyzer import get_control_data
    >>> h = ControlHandler("control")
    >>> tstep = get_control_data(h, "tstep", section="md", default=None)
    """
    key = key.lower()
    section_map = {
        "general": handler.general_parameters,
        "md": handler.md_parameters,
        "mm": handler.mm_parameters,
        "ff": handler.ff_parameters,
        "outdated": handler.outdated_parameters,
    }

    if section:
        section = section.lower()
        if section not in section_map:
            raise ValueError(f"❌ Unknown section: {section}")
        try:
            return section_map[section][key]
        except:
            return default

    for d in section_map.values():
        if key in d:
            return d[key]
    return default
