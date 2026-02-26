"""CLI package entrypoint."""

from . import main as _main_mod

main = _main_mod.main
_preinject = _main_mod._preinject
WORKFLOW_MODULES = _main_mod.WORKFLOW_MODULES
DEFAULTABLE = _main_mod.DEFAULTABLE
DEFAULT_TASKS = _main_mod.DEFAULT_TASKS
help_workflow = _main_mod.help_workflow
introspection_workflow = _main_mod.introspection_workflow
