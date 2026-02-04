# IO (Input/Output)

This section documents ReaxKit’s **I/O layer**, responsible for reading, parsing, and generating
ReaxFF-related files.

The IO layer is intentionally **analysis-free**:

- It **parses raw files into structured data**
- It **cleans and normalizes formats**
- It **exposes consistent access patterns** for analyzers and workflows

The IO API is split into three parts:

## Handlers
File readers that parse ReaxFF input/output files into pandas DataFrames
and optional per-frame structures.

- One handler per file type (e.g. `xmolout`, `fort.7`, `fort.73`)
- No plotting or physics logic
- Designed to be composable and reusable

➡️ See [handlers/](handlers/index.md)

## Generators
Writers that **create** ReaxFF-compatible input files such as
`control`, `geo`, `eregime`, `tregime`, and related configuration files.

- Deterministic and reproducible file generation
- Often driven by YAML or CLI inputs
- Do not execute simulations

➡️ See [generators/](generators/index.md)

## Base Interfaces
Shared abstractions and conventions used by all IO components.

- Common parsing lifecycle
- Metadata handling
- Error and path management

➡️ See [base.md](base_handler_doc.md)
