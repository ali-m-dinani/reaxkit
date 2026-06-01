"""Template utilities for ReaxKit-style file generators.

This module provides a generic scaffold for implementing deterministic text
generators and file writers used by engine workflows. It demonstrates the
common pattern used in generator modules: template constants, a spec dataclass,
private generate/write helpers, one public convenience function, and a registry
entry consumed by workflow code.

**Usage context**

- Template generation: Produce canonical text payloads from a typed spec.
- File writing: Persist generated outputs to stable filesystem paths.
- Workflow integration: Register generator metadata for command dispatch.

Notes
-----
Replace placeholder names/content with your target file type and rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

TEMPLATE_MAIN_TEXT = """# TEMPLATE FILE
key_a value_a
key_b value_b
"""

TEMPLATE_AUX_TEXT = """# TEMPLATE AUXILIARY FILE
aux_key aux_value
"""

__all__ = [
    "TemplateGeneratorSpec",
    "DEFAULT_TEMPLATE_SPEC",
    "TEMPLATE_GENERATOR_REGISTRY",
    "gen_template_file",
]


@dataclass(frozen=True)
class TemplateGeneratorSpec:
    """Configuration payload for template-based generator outputs.

    Holds text fragments and file naming defaults used by generation and write
    helpers.

    Fields
    ------
    main_text : str
        Full text payload for the primary generated file.
    aux_text : str
        Full text payload for the optional auxiliary file written alongside the
        primary file.
    aux_filename : str
        Output filename for the auxiliary artifact.

    Examples
    ------
    Sample spec payload/object:
    `TemplateGeneratorSpec(main_text="...", aux_text="...", aux_filename="template.aux")`
    This sample defines both primary and auxiliary text templates and the aux
    filename to be emitted next to the main file.
    """

    main_text: str = TEMPLATE_MAIN_TEXT
    aux_text: str = TEMPLATE_AUX_TEXT
    aux_filename: str = "template.aux"


DEFAULT_TEMPLATE_SPEC = TemplateGeneratorSpec()


def _gen_template_main_text(spec: TemplateGeneratorSpec = DEFAULT_TEMPLATE_SPEC) -> str:
    """Return primary template text for the configured generator spec."""
    return spec.main_text


def _gen_template_aux_text(spec: TemplateGeneratorSpec = DEFAULT_TEMPLATE_SPEC) -> str:
    """Return auxiliary template text for the configured generator spec."""
    return spec.aux_text


def _write_template_main(
    out_path: str | Path = "template.in",
    spec: TemplateGeneratorSpec = DEFAULT_TEMPLATE_SPEC,
) -> Path:
    """Write the primary generated file and return its resolved path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_gen_template_main_text(spec), encoding="utf-8")
    return out_path


def _write_template_aux(
    out_path: str | Path,
    spec: TemplateGeneratorSpec = DEFAULT_TEMPLATE_SPEC,
) -> Path:
    """Write the auxiliary generated file and return its resolved path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_gen_template_aux_text(spec), encoding="utf-8")
    return out_path


def gen_template_file(
    out_path: str | Path = "template.in",
    spec: TemplateGeneratorSpec = DEFAULT_TEMPLATE_SPEC,
) -> Path:
    """Generate template artifacts and write them to disk.

    Writes the primary file to `out_path` and writes an auxiliary file in the
    same directory using `spec.aux_filename`.

    Parameters
    -----
    out_path : str | Path
        Destination path for the primary generated file.
    spec : TemplateGeneratorSpec
        Spec object containing text payloads and auxiliary filename settings.

    Returns
    -----
    Path
        Path to the written primary file.

    Examples
    -----
    ```python
    path = gen_template_file("outputs/template.in")
    print(path)
    ```
    Sample output:
    `outputs/template.in`
    Meaning:
    The primary file is written to the requested location, and an auxiliary
    file is written beside it using `spec.aux_filename`.
    """
    main_path = _write_template_main(out_path=out_path, spec=spec)
    aux_path = Path(main_path).with_name(spec.aux_filename)
    _write_template_aux(out_path=aux_path, spec=spec)
    return main_path


TEMPLATE_GENERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    "template_generator": {
        "label": "Template Generator Artifact",
        "default_filename": "template.in",
        "spec_type": TemplateGeneratorSpec,
        "generate": _gen_template_main_text,
        "write": gen_template_file,
    }
}
