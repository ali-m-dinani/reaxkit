# Help Workflow

CLI namespace: `reaxkit help <task> [flags]`

Interactive help and discovery workflow for ReaxKit.

This workflow provides a search-based help system that maps conceptual,
human-language queries (e.g. "electric field", "restraint", "bond order")
to relevant ReaxFF input and output files.

It operates as a kind-level command (`reaxkit help`) rather than a task-based
workflow, and therefore does not define subcommands. Instead, it queries
curated help indices and ranks matches based on relevance.

Optional flags allow users to inspect why a file matched, view example
ReaxKit commands, and explore core, optional, and derived variables
associated with each file.

## Available tasks

## Usage

`reaxkit help <args>`

## Behavior

Run the `reaxkit help` command.

    Behavior:
    -----------
    - If no query is provided, prints a short usage message and exits.
    - If a query is provided, searches curated help indices and prints ranked matches.
    - Use `--top` and `--min-score` to control result count and filtering.
    - Use detail flags (`--why`, `--file_templates`, `--tags`, `--core-vars`, `--optional-vars`,
      `--derived-vars`, `--notes`) to expand what is shown per match.
    - `--all-info` enables all detail flags together.

    Examples
    -----------
    reaxkit help 'restraint'
    reaxkit help 'electric field'

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--top TOP` | Maximum number of matches to display. |
| `--min-score MIN_SCORE` | Minimum score threshold; lower-scoring matches are hidden. |
| `--why` | Show why each file matched the query. |
| `--file_templates` | Show one example ReaxKit command per match. |
| `--tags` | Show tags for each match. |
| `--core-vars` | Show core variables for each match. |
| `--optional-vars` | Show optional variables. |
| `--derived-vars` | Show derived variables. |
| `--notes` | Show notes for each match. |
| `--all-info` | Show why/file_templates/tags/core/optional/derived/notes all at once. |
