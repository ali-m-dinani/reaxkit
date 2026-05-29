"""Base result model for analysis/domain outputs.

This module defines the shared base type for result dataclasses produced by
analysis and domain-level workflows. It provides a common inheritance anchor
without imposing output fields at the base layer.

**Usage context**

- Inherit from ``BaseResult`` when defining typed analysis/domain result payloads.
- Use this base class to keep result model hierarchies consistent across modules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseResult:
    """Base result model.

    This dataclass is the root result container for specialized result
    dataclasses in ReaxKit domain and analysis code.

    Fields
    -----
    None
        ``BaseResult`` does not define fields directly; subclasses define the
        concrete output payload.
    """
