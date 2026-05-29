"""Generate engine API markdown pages with per-function/class indented sections.

This generator is intentionally simple:
- it does not reinterpret function docstring contents
- it renders function docs "as-is" via mkdocstrings
- it wraps each function/class section in a markdown-enabled indented container
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EngineModuleDocTarget:
    py_file: Path
    module_import: str
    output_md: Path
    title: str


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "reaxkit").exists():
            return candidate
    raise RuntimeError("Could not locate repository root containing src/reaxkit.")


def _to_title_from_stem(stem: str) -> str:
    parts = [p for p in stem.strip().split("_") if p]
    return " ".join(p[:1].upper() + p[1:] for p in parts) if parts else stem


def _iter_engine_modules(engine_dir: Path, *, include_private_modules: bool) -> list[Path]:
    out: list[Path] = []
    for py in sorted(engine_dir.rglob("*.py")):
        if py.name == "__init__.py":
            continue
        if not include_private_modules and py.name.startswith("_"):
            continue
        out.append(py)
    return out


def _module_import_from_file(src_root: Path, py_file: Path) -> str:
    rel = py_file.relative_to(src_root)
    return ".".join(rel.with_suffix("").parts)


def _output_md_for_file(engine_dir: Path, py_file: Path, docs_root: Path) -> Path:
    rel = py_file.relative_to(engine_dir)
    rel = rel.with_name(f"{rel.stem}_doc.md")
    return docs_root / rel


def _public_top_level_functions(py_file: Path, *, include_private_functions: bool) -> list[str]:
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not include_private_functions and node.name.startswith("_"):
                continue
            names.append(node.name)
    return names


def _public_top_level_classes(py_file: Path, *, include_private_classes: bool) -> list[str]:
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if not include_private_classes and node.name.startswith("_"):
                continue
            names.append(node.name)
    return names


def _class_method_names(py_file: Path, *, include_private_methods: bool) -> dict[str, list[str]]:
    """Return top-level class -> method names defined on that class.

    Includes only public methods by default (skips names starting with ``_``,
    including ``__init__``). When ``include_private_methods`` is true, private
    and dunder methods are included.
    """
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods: list[str] = []
        for sub in node.body:
            if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = sub.name
            if not include_private_methods and name.startswith("_"):
                continue
            methods.append(name)
        out[node.name] = methods
    return out


def _extract_preserved_item_blocks(existing_md: str) -> dict[str, str]:
    """Extract custom per-function/class markdown blocks from an existing page.

    The preserved block for each item is any content between the
    ``mkdocstrings`` options block and the closing ``</div>`` of that section.
    Keys are stored as ``function:<name>`` or ``class:<name>``.
    """
    lines = existing_md.splitlines()
    out: dict[str, str] = {}
    i = 0
    current_class: str | None = None
    while i < len(lines):
        heading = lines[i].strip()
        m = re.match(r"^## (Function|Class): `(.+)`$", heading)
        mm = re.match(r"^### Method: `(.+)`$", heading)

        if mm and current_class:
            item_key = f"method:{current_class}.{mm.group(1)}"

            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("::: "):
                if re.match(r"^### Method: `(.+)`$", lines[j].strip()) or re.match(
                    r"^## (Function|Class): `(.+)`$", lines[j].strip()
                ):
                    break
                j += 1
            if j >= len(lines) or re.match(r"^### Method: `(.+)`$", lines[j].strip()) or re.match(
                r"^## (Function|Class): `(.+)`$", lines[j].strip()
            ):
                i = j
                continue

            j += 1
            while j < len(lines) and lines[j].startswith("    "):
                j += 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1

            start = j
            section_end = len(lines)
            k = j
            while k < len(lines):
                if re.match(r"^### Method: `(.+)`$", lines[k].strip()) or re.match(
                    r"^## (Function|Class): `(.+)`$", lines[k].strip()
                ):
                    section_end = k
                    break
                k += 1

            close_idx = None
            for k in range(section_end - 1, start - 1, -1):
                if lines[k].strip() == "</div>":
                    close_idx = k
                    break

            end = close_idx if close_idx is not None else section_end
            preserved = "\n".join(lines[start:end]).strip("\n")
            if preserved.strip():
                out[item_key] = preserved
            i = section_end
            continue

        if not m:
            i += 1
            continue
        kind = m.group(1).lower()
        item_name = m.group(2)
        current_class = item_name if kind == "class" else None
        item_key = f"{kind}:{item_name}"

        # Find the mkdocstrings directive in this item section.
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith("::: "):
            if re.match(r"^## (Function|Class): `(.+)`$", lines[j].strip()):
                break
            j += 1
        if j >= len(lines) or re.match(r"^## (Function|Class): `(.+)`$", lines[j].strip()):
            i = j
            continue

        # Skip directive + its options block.
        j += 1
        while j < len(lines) and lines[j].startswith("    "):
            j += 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1

        # Capture custom block until just before the outer section closing div.
        # Use the *last* </div> in this section so nested divs in custom content
        # are preserved.
        start = j
        section_end = len(lines)
        k = j
        while k < len(lines):
            if re.match(r"^## (Function|Class): `(.+)`$", lines[k].strip()):
                section_end = k
                break
            k += 1

        # For class-level preservation, do not capture generated method blocks.
        # Keep only custom content before the first "### Method: ..." heading.
        if kind == "class":
            for k in range(start, section_end):
                if re.match(r"^### Method: `(.+)`$", lines[k].strip()):
                    section_end = k
                    break

        close_idx = None
        for k in range(section_end - 1, start - 1, -1):
            if lines[k].strip() == "</div>":
                close_idx = k
                break

        end = close_idx if close_idx is not None else section_end
        preserved = "\n".join(lines[start:end]).strip("\n")
        if preserved.strip():
            out[item_key] = preserved
        i = section_end
    return out


def _render_module_page(
    target: EngineModuleDocTarget,
    function_names: list[str],
    class_names: list[str],
    class_methods_by_class: dict[str, list[str]],
    preserved_blocks: dict[str, str] | None = None,
) -> str:
    preserved_blocks = preserved_blocks or {}
    lines: list[str] = []
    lines.append("<!-- AUTO-GENERATED by docs/scripts/generate_engine_function_docs.py -->")
    lines.append(f"# {target.title} Engine Utility")
    lines.append("")
    lines.append(f"::: {target.module_import}")
    lines.append("    options:")
    lines.append("      show_root_heading: false")
    lines.append("      show_root_full_path: false")
    lines.append("      members: []")
    lines.append("")

    if not function_names and not class_names:
        lines.append("_No top-level functions or classes found in this module._")
        lines.append("")
        return "\n".join(lines)

    for cls in class_names:
        lines.append(f"## Class: `{cls}`")
        lines.append("")
        lines.append('<div class="analysis-section-indent" markdown="1">')
        lines.append("")
        lines.append(f"::: {target.module_import}.{cls}")
        lines.append("    options:")
        lines.append("      show_root_heading: false")
        lines.append("      show_root_full_path: false")
        lines.append("      members: []")
        lines.append("")

        preserved = preserved_blocks.get(f"class:{cls}", "").strip("\n")
        if preserved:
            lines.append(preserved)
            lines.append("")

        if cls in class_methods_by_class and class_methods_by_class[cls]:
            for method in class_methods_by_class[cls]:
                lines.append(f"### Method: `{method}`")
                lines.append("")
                lines.append('<div class="analysis-section-indent" markdown="1">')
                lines.append("")
                lines.append(f"::: {target.module_import}.{cls}.{method}")
                lines.append("    options:")
                lines.append("      show_root_heading: false")
                lines.append("      show_root_full_path: false")
                lines.append("")

                preserved_method = preserved_blocks.get(f"method:{cls}.{method}", "").strip("\n")
                if preserved_method:
                    lines.append(preserved_method)
                    lines.append("")

                lines.append("</div>")
                lines.append("")

        lines.append("</div>")
        lines.append("")

    for fn in function_names:
        lines.append(f"## Function: `{fn}`")
        lines.append("")
        lines.append('<div class="analysis-section-indent" markdown="1">')
        lines.append("")
        lines.append(f"::: {target.module_import}.{fn}")
        lines.append("    options:")
        lines.append("      show_root_heading: false")
        lines.append("      show_root_full_path: false")
        lines.append("")

        preserved = preserved_blocks.get(f"function:{fn}", "").strip("\n")
        if preserved:
            lines.append(preserved)
            lines.append("")

        lines.append("</div>")
        lines.append("")

    return "\n".join(lines)


def generate_engine_function_docs(
    *,
    src_root: Path,
    docs_root: Path,
    include_private_modules: bool = False,
    include_private_functions: bool = False,
) -> list[Path]:
    engine_dir = src_root / "reaxkit" / "engine"
    if not engine_dir.exists():
        raise FileNotFoundError(f"Could not find engine directory: {engine_dir}")

    targets: list[EngineModuleDocTarget] = []
    for py in _iter_engine_modules(engine_dir, include_private_modules=include_private_modules):
        module_import = _module_import_from_file(src_root, py)
        out_md = _output_md_for_file(engine_dir, py, docs_root)
        targets.append(
            EngineModuleDocTarget(
                py_file=py,
                module_import=module_import,
                output_md=out_md,
                title=_to_title_from_stem(py.stem),
            )
        )

    written: list[Path] = []
    for target in targets:
        fn_names = _public_top_level_functions(
            target.py_file,
            include_private_functions=include_private_functions,
        )
        cls_names = _public_top_level_classes(
            target.py_file,
            include_private_classes=include_private_functions,
        )
        # Skip modules that expose no public top-level API.
        if not fn_names and not cls_names:
            if target.output_md.exists():
                try:
                    existing = target.output_md.read_text(encoding="utf-8")
                except Exception:
                    existing = ""
                if existing.startswith("<!-- AUTO-GENERATED by docs/scripts/generate_engine_function_docs.py -->"):
                    target.output_md.unlink(missing_ok=True)
            continue

        class_methods = _class_method_names(
            target.py_file, include_private_methods=include_private_functions
        )
        class_methods_by_class = {
            cls: methods
            for cls, methods in class_methods.items()
            if cls in cls_names and methods
        }
        preserved_blocks: dict[str, str] = {}
        if target.output_md.exists():
            existing_md = target.output_md.read_text(encoding="utf-8")
            preserved_blocks = _extract_preserved_item_blocks(existing_md)

        md = _render_module_page(
            target,
            fn_names,
            cls_names,
            class_methods_by_class,
            preserved_blocks=preserved_blocks,
        )
        target.output_md.parent.mkdir(parents=True, exist_ok=True)
        target.output_md.write_text(md, encoding="utf-8")
        written.append(target.output_md)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate per-function engine API markdown pages with indented content blocks."
    )
    parser.add_argument("--src-root", default=None, help="Path to src/ (auto-detected if omitted).")
    parser.add_argument("--docs-root", default=None, help="Output docs/api/engine directory (auto-detected if omitted).")
    parser.add_argument(
        "--include-private-modules",
        action="store_true",
        help="Include engine modules whose filename starts with '_'.",
    )
    parser.add_argument(
        "--include-private-functions",
        action="store_true",
        help="Include top-level functions whose name starts with '_'.",
    )
    args = parser.parse_args()

    repo_root = _find_repo_root(Path(__file__).resolve())
    src_root = Path(args.src_root).resolve() if args.src_root else (repo_root / "src")
    docs_root = Path(args.docs_root).resolve() if args.docs_root else (repo_root / "docs" / "api" / "engine")

    written = generate_engine_function_docs(
        src_root=src_root,
        docs_root=docs_root,
        include_private_modules=bool(args.include_private_modules),
        include_private_functions=bool(args.include_private_functions),
    )

    print(f"[generated] {len(written)} engine doc pages under: {docs_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
