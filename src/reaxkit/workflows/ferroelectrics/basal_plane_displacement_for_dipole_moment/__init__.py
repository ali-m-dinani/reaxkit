"""Workflows for basal-plane displacement dipoles and polarization."""

from reaxkit.workflows.ferroelectrics.basal_plane_displacement_for_dipole_moment.dipole_workflow import (
    ALL_LEGACY_COMMANDS as DIPOLE_LEGACY_COMMANDS,
    COMMAND as DIPOLE_COMMAND,
)
from reaxkit.workflows.ferroelectrics.basal_plane_displacement_for_dipole_moment.polarization_workflow import (
    ALL_LEGACY_COMMANDS as POLARIZATION_LEGACY_COMMANDS,
    COMMAND as POLARIZATION_COMMAND,
)

ALL_COMMANDS = (DIPOLE_COMMAND, POLARIZATION_COMMAND)
ALL_LEGACY_COMMANDS = (*DIPOLE_LEGACY_COMMANDS, *POLARIZATION_LEGACY_COMMANDS)

__all__ = [
    "ALL_COMMANDS", "ALL_LEGACY_COMMANDS", "DIPOLE_COMMAND", "POLARIZATION_COMMAND",
]
