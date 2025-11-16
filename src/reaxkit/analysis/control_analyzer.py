"""analyzer for control file"""
from __future__ import annotations
from typing import Optional

from reaxkit.io.control_handler import ControlHandler


def available_sections(handler: ControlHandler) -> list[str]:
    """Return all sections that contain parameters."""
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


def available_keys(handler: ControlHandler, section: Optional[str] = None) -> list[str]:
    """Return list of all keys either for a section or all combined."""
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


def get(
    handler: ControlHandler,
    key: str,
    section: Optional[str] = None,
    default=None,
):
    """Retrieve a control parameter value from a section or all sections.

    Parameters
    ----------
    handler : ControlHandler
        The parsed control handler instance.
    key : str
        Parameter name (case-insensitive).
    section : str, optional
        Section to look in (e.g., 'general', 'md', 'mm', 'ff', 'outdated').
        If None, search all sections.
    default : any, optional
        Value returned if the key is not found.

    Returns
    -------
    any
        Parameter value or `default` if not found.
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
        return section_map[section].get(key, default)

    for d in section_map.values():
        if key in d:
            return d[key]
    return default
