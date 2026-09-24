# cli

## Purpose
Provides command-line entrypoints and bootstrap path setup for invoking ReaxKit workflows and commands.

## What Belongs Here
- CLI startup/entrypoint modules.
- Path/bootstrap helpers required before command dispatch.

## What Does Not Belong Here
- Analysis algorithm implementations.
- Workflow business logic.

## Structure
- `main.py`: main CLI entrypoint.
- `path.py`: path/environment bootstrapping.
- `__init__.py`: package exports.

## Flow
CLI parses and normalizes user command input, then dispatches into workflow/core command resolution.

## Extension Points
- Add new top-level CLI behavior only when it cannot live cleanly in workflows/core registries.

## Help metadata and validation

Register each visible flag's category and short/full visibility in
`help_flag_registry.py`. Use `help_metadata.py`'s parser-path overrides when
the same flag has different roles in different commands. Required arguments
always appear in short help; suppressed internal arguments remain hidden.

After changing parsers or metadata, run these commands from the repository root:

```shell
python tools/cli_help_inventory.py
python docs/scripts/generate_workflow_cli_docs.py
python docs/scripts/generate_workflow_cli_docs.py --check
python -m pytest tests/cli tests/core/test_command_inventory.py
```

The inventory enumerates actual registered parsers and nested tasks, including
late-added shared flags. A rejected route fails generation without replacing
the last good inventory. Generated documentation includes full-help options;
keep examples and figure blocks in the preserved documentation sections.

Use `python tools/compare_cli_namespaces.py REVISION` to compare
representative invocation namespaces against an earlier CLI dispatcher while
holding workflow modules and shared helpers constant.
