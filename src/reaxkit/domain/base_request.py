"""Base request model for analysis/domain requests.

This module defines the shared base type for request dataclasses used by
analysis and domain-level workflows. It provides a common inheritance anchor
without imposing request fields at the base layer.

**Usage context**

- Inherit from ``BaseRequest`` when defining typed request payloads for analysis tasks.
- Use this base class to keep request model hierarchies consistent across domain modules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseRequest:
    """Base request model.

    This dataclass is the root request container for specialized request
    dataclasses in ReaxKit domain and analysis code.

    Fields
    -----
    None
        ``BaseRequest`` does not define fields directly; subclasses define the
        concrete request payload.
    """
