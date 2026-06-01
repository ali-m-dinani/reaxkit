# Docstring Inclusion Rules

This page defines what elements should be included in docstrings for files and functions across ReaxKit modules.
For section headings, if anything is mentioned in brackets, use exactly that as the section heading for the sake of consistency. For example, if it says [Parameters], use "Parameters" as the section heading in the docstring. If there is no section heading mentioned, just include the elements without any section headings.

## File-Level Docstring (All Module Types)

Include these elements in file-level docstrings for workflow, analyzer, utility, and other module types:

1. One-line module purpose
2. Short paragraph with scope/boundary
3. [**Usage context**] items under this section should have bullet points with a short description of the context. make sure you put a blank line under the section heading, and also put the starts around the **Usage context** to show it as a bold heading.
4. [Notes] Optional notes on invariants/assumptions (if any). make sure you put a line ----- under the section heading

## Function-Level Docstring (Public Functions, All Module Types)

Include these elements for public functions:

1. One-line summary
2. Extended summary
3. [Works on] (only for analyzer tasks)
4. Optional [Notes] make sure you put a line ----- under the section heading
4. [Parameters] arguments. make sure you put a line ----- under the section heading
5. [Returns] make sure you put a line ----- under the section heading
6. [Examples] make sure you put a line ----- under the section heading
   Under [Examples], include:
   - a runnable code snippet
   - a sample output
   - a short description of what the output means
7. Optional [See Also] make sure you put a blank line under the section heading.
9. Optional [References] make sure you put a blank line under the section heading.

## Function-Level Docstring (Private Functions, All Module Types)

Use a minimal style for private functions:

1. One-line summary
2. Optional [Notes]

## Dataclass Docstring (Analyzer Task Request/Result)

Use this section for analyzer-task dataclasses (typically one `Request` and one
`Result` dataclass per analyzer task).

### Request Dataclass

Include these elements:

1. One-line summary
2. Short paragraph describing what analysis inputs/configuration the request carries
3. [Fields] make sure you put a line ----- under the section heading
4. Optional [Notes] make sure you put a line ----- under the section heading
5. Optional [Examples] make sure you put a line ----- under the section heading
   Under [Examples], include:
   - a sample request payload/object
   - a short description of what the sample represents

Field entries under [Fields] should include:

- field name
- expected type/shape
- meaning/units/allowed range when relevant
- default behavior if optional

### Result Dataclass

Include these elements:

1. One-line summary
2. Short paragraph describing what outputs the analyzer returns
3. [Fields] make sure you put a line ----- under the section heading
4. Optional [Notes] make sure you put a line ----- under the section heading
5. Optional [Examples] make sure you put a line ----- under the section heading
   Under [Examples], include:
   - a sample output payload/object
   - a short description of what the output fields mean
6. Optional [References] make sure you put a blank line under the section heading.

Field entries under [Fields] should include:

- field name
- type/container shape
- semantic meaning (and units when relevant)
- interpretation notes (for derived/normalized values)

## Dataclass Docstring (General Domain/Composite Models)

Use this section for non-analyzer dataclasses such as canonical domain models,
composite bundles, and shared typed containers.

Include these elements:

1. One-line summary
2. Short paragraph describing purpose/scope and intended usage boundary
3. [Fields] make sure you put a line ----- under the section heading
4. Optional [Notes] make sure you put a line ----- under the section heading
5. Optional [Examples] make sure you put a line ----- under the section heading
6. Optional [References] make sure you put a blank line under the section heading.

Field entries under [Fields] should include:

- field name
- expected type/container shape
- semantic meaning (and units when relevant)
- default behavior if optional

Engine-specific source references in these docstrings should use conditional
wording. Example: if you use ReaxFF engine, then the file would be ``fort.99``.
