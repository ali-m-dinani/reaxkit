"""Study design workflow package.

This package contains the `study` CLI workflow split into focused modules:
the CLI entry surface, help text, and runtime implementation.

**Usage context**

- Command routing: Imported by CLI command registries for the `study` command.
- CLI composition: Keeps parser/help wiring separate from runtime operations.
- Workflow execution: Delegates heavy initialization/run/analyze/aggregate flows.
"""

