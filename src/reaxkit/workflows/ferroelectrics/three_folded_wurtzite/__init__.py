"""Dedicated CLI workflows for three-folded wurtzite analyses."""

from reaxkit.workflows.ferroelectrics.three_folded_wurtzite.neighbors_workflow import (
    ALL_LEGACY_COMMANDS as NEIGHBORS_LEGACY_COMMANDS,
    COMMAND as NEIGHBORS_COMMAND,
)
from reaxkit.workflows.ferroelectrics.three_folded_wurtzite.polarity_trajectory_workflow import (
    ALL_LEGACY_COMMANDS as POLARITY_TRAJECTORY_LEGACY_COMMANDS,
    COMMAND as POLARITY_TRAJECTORY_COMMAND,
)
from reaxkit.workflows.ferroelectrics.three_folded_wurtzite.polarity_workflow import (
    ALL_LEGACY_COMMANDS as POLARITY_LEGACY_COMMANDS,
    COMMAND as POLARITY_COMMAND,
)
from reaxkit.workflows.ferroelectrics.three_folded_wurtzite.polarization_workflow import (
    ALL_LEGACY_COMMANDS as POLARIZATION_LEGACY_COMMANDS,
    COMMAND as POLARIZATION_COMMAND,
)

ALL_COMMANDS = (
    NEIGHBORS_COMMAND,
    POLARITY_COMMAND,
    POLARIZATION_COMMAND,
    POLARITY_TRAJECTORY_COMMAND,
)
ALL_LEGACY_COMMANDS = (
    *NEIGHBORS_LEGACY_COMMANDS,
    *POLARITY_LEGACY_COMMANDS,
    *POLARIZATION_LEGACY_COMMANDS,
    *POLARITY_TRAJECTORY_LEGACY_COMMANDS,
)

__all__ = [
    "ALL_COMMANDS", "ALL_LEGACY_COMMANDS", "NEIGHBORS_COMMAND",
    "POLARITY_COMMAND", "POLARITY_TRAJECTORY_COMMAND",
    "POLARIZATION_COMMAND",
]
