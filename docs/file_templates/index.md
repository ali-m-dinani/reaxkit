# ReaxKit File Templates

This directory contains reference templates for building new ReaxKit modules.
These files are for contributors to copy and adapt; they are not runtime code.

## Available Templates

### [`template_analyzer.py`](https://github.com/ali-m-dinani/reaxkit/blob/new-features/docs/file_templates/template_analyzer.py)
Template for analysis modules with the ReaxKit request-task-result pattern:
- request dataclass for user inputs
- task class with `recommended_presentations` and `run(...)`
- result dataclass with table/request-oriented outputs

### [`template_workflow.py`](https://github.com/ali-m-dinani/reaxkit/blob/new-features/docs/file_templates/template_workflow.py)
Template for CLI workflow modules:
- argument registration and validation
- task/request construction
- user-facing output and file export flow

### [`template_handler.py`](https://github.com/ali-m-dinani/reaxkit/blob/new-features/docs/file_templates/template_handler.py)
Template for file I/O handlers:
- parse raw file content
- expose structured DataFrame-style accessors
- keep parsing concerns separated from analysis logic

### [`template_generator.py`](https://github.com/ali-m-dinani/reaxkit/blob/new-features/docs/file_templates/template_generator.py)
Template for generator-style modules (e.g., writing ReaxFF input files):
- spec dataclasses
- validation and normalization helpers
- deterministic content generation and save helpers

### [`template_utils.py`](https://github.com/ali-m-dinani/reaxkit/blob/new-features/docs/file_templates/template_utils.py)
Template for utility modules:
- focused reusable functions
- clear input/output contracts
- docstrings aligned with project conventions

## Usage

1. Pick the closest template.
2. Copy it into the target package.
3. Rename classes/functions to domain-specific names.
4. Replace placeholder logic with real implementation.
5. Keep docstrings consistent with [Docstring content and inclusion guidelines](../rules_and_conventions/docstring_content_and_inclusion_guidelines.md)
