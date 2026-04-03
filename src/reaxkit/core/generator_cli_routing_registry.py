"""Registry for top-level generator commands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorSpec:
    """Metadata for a direct generator command."""

    name: str
    module_path: str


GENERATOR_REGISTRY: dict[str, GeneratorSpec] = {}


def register_generator(name: str, *, module_path: str) -> GeneratorSpec:
    """Register a direct generator command."""
    spec = GeneratorSpec(name=name, module_path=module_path)
    GENERATOR_REGISTRY[name] = spec
    return spec


def get_registered_generators() -> dict[str, GeneratorSpec]:
    """Return all registered generator commands."""
    return dict(GENERATOR_REGISTRY)


register_generator("get-control", module_path="reaxkit.workflows.file_tools.control_workflow")
register_generator("make-control", module_path="reaxkit.workflows.file_tools.control_workflow")
register_generator("write-control", module_path="reaxkit.workflows.file_tools.control_workflow")
register_generator("study", module_path="reaxkit.workflows.meta.study_workflow")
register_generator("add-alias", module_path="reaxkit.workflows.meta.command_alias_workflow")
register_generator("make-eregime", module_path="reaxkit.workflows.file_tools.eregime_workflow")
register_generator("make-addmol", module_path="reaxkit.workflows.file_tools.addmol_workflow")
register_generator("trim-xmolout", module_path="reaxkit.workflows.file_tools.xmolout_workflow")
register_generator("xtob", module_path="reaxkit.workflows.file_tools.geo_workflow")
register_generator("make-geo", module_path="reaxkit.workflows.file_tools.geo_workflow")
register_generator("sort-geo", module_path="reaxkit.workflows.file_tools.geo_workflow")
register_generator("orthogonalize-geo", module_path="reaxkit.workflows.file_tools.geo_workflow")
register_generator("place-geo", module_path="reaxkit.workflows.file_tools.geo_workflow")
register_generator("add-geo-restraint", module_path="reaxkit.workflows.file_tools.geo_workflow")
register_generator("extract-optimized-ffield", module_path="reaxkit.workflows.file_tools.fort83_workflow")
register_generator("export-trainset", module_path="reaxkit.workflows.file_tools.trainset_workflow")
register_generator("make-trainset-settings", module_path="reaxkit.workflows.file_tools.trainset_workflow")
register_generator("make-trainset", module_path="reaxkit.workflows.file_tools.trainset_workflow")
register_generator("make-tregime", module_path="reaxkit.workflows.file_tools.tregime_workflow")
register_generator("make-vregime", module_path="reaxkit.workflows.file_tools.vregime_workflow")
register_generator("free-up", module_path="reaxkit.workflows.meta.workspace_management_workflow")
register_generator("manage-workspace", module_path="reaxkit.workflows.meta.workspace_management_workflow")
