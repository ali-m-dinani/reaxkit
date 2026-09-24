"""Backward-compatible imports for the command catalog."""

from reaxkit.core.registry.command_catalog import (
    COMMAND_REGISTRY,
    CommandSpec,
    command,
    get_registered_commands,
    register_command,
    register_generator_command,
)

# These aliases predate the generated command metadata and remain part of the
# public command-resolution contract.
register_command(
    "msd",
    kind="analysis",
    aliases=("mean-square-displacement", "mean_square_displacement"),
)
register_command(
    "rdf",
    kind="analysis",
    aliases=("radial-distribution-function", "radial_distribution_function"),
)
register_command("rdf_property", kind="analysis", aliases=("rdf-property", "rdfproperty"))

__all__ = [
    "COMMAND_REGISTRY",
    "CommandSpec",
    "command",
    "get_registered_commands",
    "register_command",
    "register_generator_command",
]
