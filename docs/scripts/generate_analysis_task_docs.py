"""Generate structured analysis task docs (AST-based) for selected modules."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re


@dataclass(frozen=True)
class AnalysisDocTarget:
    analysis_file: Path
    module_import: str
    output_md: Path
    title: str


@dataclass
class FieldRow:
    name: str
    type_text: str
    default: str
    help_text: str
    choices: str


@dataclass
class PreservedBlock:
    section_kind: str  # request:<Class> | task:<Class> | result:<Class>
    context_heading: str  # nearest preceding ###/#### heading or "__section_start__"
    content: str


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "reaxkit").exists():
            return candidate
    raise RuntimeError("Could not locate repository root containing src/reaxkit.")


def _literal(node: ast.AST) -> Any | None:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    return repr(value)


def _doc_sections(doc: str) -> tuple[str, dict[str, str]]:
    lines = doc.splitlines()
    summary_lines: list[str] = []
    sections: dict[str, list[str]] = {}

    i = 0
    current_section: str | None = None
    saw_section = False
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and nxt.strip().startswith("-----"):
            current_section = line.strip()
            sections[current_section] = []
            saw_section = True
            i += 2
            continue
        if not saw_section:
            summary_lines.append(line)
        else:
            if current_section is not None:
                sections[current_section].append(line)
        i += 1

    cleaned = {k: "\n".join(v).strip() for k, v in sections.items()}
    return "\n".join(summary_lines).strip(), cleaned


def _class_fields(cls: ast.ClassDef) -> list[FieldRow]:
    rows: list[FieldRow] = []
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        name = node.target.id
        type_text = ast.unparse(node.annotation) if node.annotation is not None else ""
        default_text = ""
        help_text = ""
        choices_text = ""

        if node.value is not None:
            if isinstance(node.value, ast.Call):
                # dataclasses field(...) alias path
                for kw in node.value.keywords:
                    if kw.arg == "default":
                        default_text = _as_text(_literal(kw.value))
                    elif kw.arg == "default_factory":
                        default_text = f"default_factory={ast.unparse(kw.value)}"
                    elif kw.arg == "metadata":
                        metadata = _literal(kw.value)
                        if isinstance(metadata, dict):
                            help_text = str(metadata.get("help", "")) if metadata.get("help", "") is not None else ""
                            choices_text = _as_text(metadata.get("choices", ""))
            else:
                default_text = _as_text(_literal(node.value))

        rows.append(
            FieldRow(
                name=name,
                type_text=type_text,
                default=default_text,
                help_text=help_text,
                choices=choices_text,
            )
        )
    return rows


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _render_field_table(rows: list[FieldRow]) -> list[str]:
    out = [
        "| Field | Type | Default | Help | Choices |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| `{_escape_cell(r.name)}` | `{_escape_cell(r.type_text)}` | "
            f"{_escape_cell(r.default)} | {_escape_cell(r.help_text)} | {_escape_cell(r.choices)} |"
        )
    return out


def _find_task_classes(classes: list[ast.ClassDef]) -> list[ast.ClassDef]:
    out: list[ast.ClassDef] = []
    for cls in classes:
        if any(
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Name)
            and dec.func.id == "register_task"
            for dec in cls.decorator_list
        ):
            out.append(cls)
            continue
        if any(isinstance(base, ast.Name) and base.id == "AnalysisTask" for base in cls.bases):
            out.append(cls)
    return out


def _format_method_signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    params: list[str] = []
    a = fn.args

    def _fmt_arg(arg: ast.arg, default_node: ast.AST | None = None) -> str:
        if arg.annotation is not None:
            base = f"{arg.arg}: {ast.unparse(arg.annotation)}"
        else:
            base = arg.arg
        if default_node is not None:
            base = f"{base}={ast.unparse(default_node)}"
        return base

    ordered = list(a.posonlyargs) + list(a.args)
    defaults = list(a.defaults)
    defaults_start = len(ordered) - len(defaults)
    for idx, arg in enumerate(ordered):
        if idx == 0 and arg.arg in {"self", "cls"}:
            continue
        default_node = defaults[idx - defaults_start] if idx >= defaults_start and defaults else None
        params.append(_fmt_arg(arg, default_node))

    if a.vararg is not None:
        params.append("*" + _fmt_arg(a.vararg))
    elif a.kwonlyargs:
        params.append("*")

    for kwarg, default_node in zip(a.kwonlyargs, a.kw_defaults):
        params.append(_fmt_arg(kwarg, default_node))

    if a.kwarg is not None:
        params.append("**" + _fmt_arg(a.kwarg))

    return f"{fn.name}({', '.join(params)})"


def _method_doc_body(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return ast.get_docstring(fn) or ""


def _method_sections(doc: str) -> tuple[str, list[tuple[str, str]]]:
    """Split method docstring into intro + ordered setext sections."""
    lines = doc.splitlines()
    intro: list[str] = []
    sections: list[tuple[str, str]] = []
    i = 0
    current_title: str | None = None
    current_lines: list[str] = []
    saw_section = False
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and nxt.strip() and set(nxt.strip()) == {"-"} and len(nxt.strip()) >= 3:
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines).strip()))
                current_lines = []
            current_title = line.strip()
            saw_section = True
            i += 2
            continue
        if saw_section and current_title is not None:
            current_lines.append(line)
        else:
            intro.append(line)
        i += 1

    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return "\n".join(intro).strip(), sections


def _parse_param_like_rows(
    body: str,
    *,
    allowed_names: set[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Parse numpydoc-style `name : type` blocks into rows."""
    lines = body.splitlines()
    rows: list[tuple[str, str, str]] = []
    i = 0
    pat = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$")
    while i < len(lines):
        raw = lines[i]
        m = pat.match(raw)
        if not m:
            i += 1
            continue
        name = m.group(1).strip()
        if allowed_names is not None and name not in allowed_names:
            i += 1
            continue
        type_text = m.group(2).strip()
        desc_lines: list[str] = []
        i += 1
        while i < len(lines):
            nxt = lines[i]
            m_next = pat.match(nxt)
            if m_next:
                if allowed_names is None or m_next.group(1).strip() in allowed_names:
                    break
            if nxt.strip():
                desc_lines.append(nxt.strip())
            i += 1
        rows.append((name, type_text, " ".join(desc_lines).strip()))
    return rows


def _parse_returns_rows(body: str) -> list[tuple[str, str]]:
    """Parse returns section into (type, description) rows."""
    param_like = _parse_param_like_rows(body)
    if param_like:
        return [(f"{name} : {typ}", desc) for name, typ, desc in param_like]

    lines = [ln.rstrip() for ln in body.splitlines()]
    blocks: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        if not ln.strip():
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(ln)
    if cur:
        blocks.append(cur)

    out: list[tuple[str, str]] = []
    for block in blocks:
        typ = block[0].strip()
        desc = " ".join(ln.strip() for ln in block[1:]).strip()
        out.append((typ, desc))
    return out


def _parse_name_only_param_rows(body: str, *, allowed_names: set[str]) -> list[tuple[str, str]]:
    """Parse sections where parameter names are standalone lines followed by description."""
    lines = body.splitlines()
    rows: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        name = lines[i].strip()
        if name not in allowed_names:
            i += 1
            continue
        i += 1
        desc_lines: list[str] = []
        while i < len(lines):
            nxt = lines[i].strip()
            if nxt in allowed_names:
                break
            if nxt:
                desc_lines.append(nxt)
            i += 1
        rows.append((name, " ".join(desc_lines).strip()))
    return rows


def _render_method_section(title: str, body: str) -> list[str]:
    lines: list[str] = [f"#### {title}", ""]
    normalized = title.strip().lower()

    if normalized == "parameters":
        rows = _parse_param_like_rows(body)
        if rows:
            lines.append("| Name | Type | Description |")
            lines.append("|---|---|---|")
            for name, typ, desc in rows:
                lines.append(
                    f"| `{name.replace('|', '\\|')}` | `{typ.replace('|', '\\|')}` | {desc.replace('|', '\\|')} |"
                )
            lines.append("")
            return lines

    if normalized == "returns":
        rows = _parse_returns_rows(body)
        if rows:
            lines.append("| Type | Description |")
            lines.append("|---|---|")
            for typ, desc in rows:
                lines.append(f"| `{typ.replace('|', '\\|')}` | {desc.replace('|', '\\|')} |")
            lines.append("")
            return lines

    if normalized == "examples":
        if "```" in body:
            lines.append(body.rstrip())
        else:
            lines.append("```text")
            lines.append(body.rstrip())
            lines.append("```")
        lines.append("")
        return lines

    # For Notes / Works on / fallback sections:
    if body.strip():
        lines.append(body.strip())
        lines.append("")
    else:
        lines.append("_No details provided._")
        lines.append("")
    return lines


def _render_doc_section(
    title: str,
    body: str,
    *,
    allowed_param_names: set[str] | None = None,
) -> list[str]:
    """Render a generic numpydoc-style section with consistent formatting."""
    lines: list[str] = [f"### {title}", ""]
    normalized = title.strip().lower()

    if normalized == "parameters":
        rows = _parse_param_like_rows(body, allowed_names=allowed_param_names)
        if rows:
            lines.append("| Name | Type | Description |")
            lines.append("|---|---|---|")
            for name, typ, desc in rows:
                lines.append(
                    f"| `{name.replace('|', '\\|')}` | `{typ.replace('|', '\\|')}` | {desc.replace('|', '\\|')} |"
                )
            lines.append("")
            return lines
        if allowed_param_names:
            rows2 = _parse_name_only_param_rows(body, allowed_names=allowed_param_names)
            if rows2:
                for name, desc in rows2:
                    lines.append(f"- `{name}`: {desc}")
                lines.append("")
                return lines

    if normalized == "returns":
        rows = _parse_returns_rows(body)
        if rows:
            lines.append("| Type | Description |")
            lines.append("|---|---|")
            for typ, desc in rows:
                lines.append(f"| `{typ.replace('|', '\\|')}` | {desc.replace('|', '\\|')} |")
            lines.append("")
            return lines

    if normalized == "examples":
        _append_examples_or_text(lines, body)
        return lines

    if body.strip():
        lines.append(body.strip())
        lines.append("")
    else:
        lines.append("_No details provided._")
        lines.append("")
    return lines


def _section_kind_from_heading(line: str) -> str | None:
    stripped = line.strip()
    m = re.match(r"^## Request: `(.+)`$", stripped)
    if m:
        return f"request:{m.group(1)}"
    m = re.match(r"^## Task: `(.+)`$", stripped)
    if m:
        return f"task:{m.group(1)}"
    m = re.match(r"^## Result: `(.+)`$", stripped)
    if m:
        return f"result:{m.group(1)}"
    return None


def _extract_preserved_blocks(existing_md: str) -> list[PreservedBlock]:
    """Extract developer-added anchor/figure blocks from an existing generated page.

    The preserved content starts at an anchor line (`<a id="..."></a>`) and
    captures the surrounding explanatory text + nested div block.
    """
    lines = existing_md.splitlines()
    out: list[PreservedBlock] = []

    i = 0
    current_section: str | None = None
    while i < len(lines):
        line = lines[i]
        sec = _section_kind_from_heading(line)
        if sec is not None:
            current_section = sec
            i += 1
            continue

        if current_section is None:
            i += 1
            continue

        if line.strip().startswith("<a id="):
            context = "__section_start__"
            k = i - 1
            while k >= 0:
                prev = lines[k].strip()
                if prev.startswith("#### ") or prev.startswith("### "):
                    context = prev
                    break
                if prev.startswith("## "):
                    break
                k -= 1

            j = i
            div_depth = 0
            saw_div = False
            while j < len(lines):
                s = lines[j].strip()
                if s.startswith("## "):
                    break
                # Track nested <div> blocks commonly used for figures.
                if "<div" in s and not s.startswith("</div>"):
                    div_depth += s.count("<div")
                    saw_div = True
                if "</div>" in s:
                    div_depth -= s.count("</div>")
                    if div_depth < 0:
                        div_depth = 0

                if j > i:
                    next_is_section = (j + 1 < len(lines) and lines[j + 1].strip().startswith("## "))
                    if saw_div:
                        if div_depth == 0 and (next_is_section or (j + 1 < len(lines) and lines[j + 1].strip() == "")):
                            j += 1
                            break
                    else:
                        if s == "" and (j + 1 < len(lines) and lines[j + 1].strip() == ""):
                            break
                j += 1

            content = "\n".join(lines[i:j]).strip("\n")
            if content:
                out.append(PreservedBlock(section_kind=current_section, context_heading=context, content=content))
            i = j
            continue

        i += 1

    return out


def _append_preserved_at_context(
    lines: list[str],
    blocks: list[PreservedBlock],
    section_kind: str,
    context_heading: str,
    consumed: set[int],
) -> None:
    for idx, block in enumerate(blocks):
        if idx in consumed:
            continue
        if block.section_kind != section_kind:
            continue
        if block.context_heading != context_heading:
            continue
        lines.append(block.content)
        lines.append("")
        consumed.add(idx)


def _append_remaining_preserved(
    lines: list[str],
    blocks: list[PreservedBlock],
    section_kind: str,
    consumed: set[int],
) -> None:
    for idx, block in enumerate(blocks):
        if idx in consumed:
            continue
        if block.section_kind != section_kind:
            continue
        lines.append(block.content)
        lines.append("")
        consumed.add(idx)


def _class_stem(name: str) -> str:
    for suffix in ("Request", "Result", "Task"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _append_examples_or_text(lines: list[str], body: str) -> None:
    text = body.strip()
    if not text:
        return
    if "```" in text:
        lines.append(text)
        lines.append("")
        return
    lines.append("```text")
    lines.append(text)
    lines.append("```")
    lines.append("")


def _write_page(target: AnalysisDocTarget) -> Path:
    source = target.analysis_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]

    request_classes = [c for c in classes if c.name.endswith("Request")]
    result_classes = [c for c in classes if c.name.endswith("Result")]
    task_classes = _find_task_classes(classes)

    request_by_stem = {_class_stem(c.name): c for c in request_classes}
    result_by_stem = {_class_stem(c.name): c for c in result_classes}
    task_by_stem = {_class_stem(c.name): c for c in task_classes}
    stems_in_order = []
    for cls in classes:
        stem = _class_stem(cls.name)
        if stem in request_by_stem or stem in task_by_stem or stem in result_by_stem:
            if stem not in stems_in_order:
                stems_in_order.append(stem)

    preserved_blocks: list[PreservedBlock] = []
    if target.output_md.exists():
        preserved_blocks = _extract_preserved_blocks(target.output_md.read_text(encoding="utf-8"))
    consumed_blocks: set[int] = set()

    lines: list[str] = []
    lines.append("<!-- AUTO-GENERATED by docs/scripts/generate_analysis_task_docs.py -->")
    lines.append(f"# {target.title}")
    lines.append("")
    lines.append(f"::: {target.module_import}")
    lines.append("    options:")
    lines.append("      show_root_heading: false")
    lines.append("      show_root_full_path: false")
    lines.append("      members: []")
    lines.append("")

    for stem in stems_in_order:
        request_cls = request_by_stem.get(stem)
        task_cls = task_by_stem.get(stem)
        result_cls = result_by_stem.get(stem)

        if request_cls is not None:
            section_key = f"request:{request_cls.name}"
            lines.append(f"## Request: `{request_cls.name}`")
            lines.append("")
            lines.append('<div class="analysis-section-indent" markdown="1">')
            lines.append("")
            _append_preserved_at_context(lines, preserved_blocks, section_key, "__section_start__", consumed_blocks)
            req_doc = ast.get_docstring(request_cls) or ""
            req_summary, req_sections = _doc_sections(req_doc)
            req_fields = _class_fields(request_cls)
            req_field_names = {row.name for row in req_fields}
            if req_summary:
                lines.append(req_summary)
                lines.append("")
            lines.append("### Fields")
            lines.append("")
            lines.extend(_render_field_table(req_fields))
            lines.append("")
            for section_name, body in req_sections.items():
                if not body.strip():
                    continue
                if section_name.strip().lower() == "fields":
                    continue
                lines.extend(
                    _render_doc_section(
                        section_name,
                        body,
                        allowed_param_names=req_field_names if section_name.strip().lower() == "parameters" else None,
                    )
                )
                _append_preserved_at_context(lines, preserved_blocks, section_key, f"### {section_name}", consumed_blocks)
            _append_remaining_preserved(lines, preserved_blocks, section_key, consumed_blocks)
            lines.append("</div>")
            lines.append("")

        if task_cls is not None:
            section_key = f"task:{task_cls.name}"
            lines.append(f"## Task: `{task_cls.name}`")
            lines.append("")
            lines.append('<div class="analysis-section-indent" markdown="1">')
            lines.append("")
            _append_preserved_at_context(lines, preserved_blocks, section_key, "__section_start__", consumed_blocks)
            task_doc = ast.get_docstring(task_cls) or ""
            if task_doc:
                lines.append(task_doc.strip())
                lines.append("")

            methods = [n for n in task_cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            wanted = [m for m in methods if m.name in {"recommended_presentations", "run"}]
            for method in wanted:
                method_heading = f"### Method: `{_format_method_signature(method)}`"
                lines.append(method_heading)
                lines.append("")
                lines.append('<div class="analysis-method-indent" markdown="1">')
                lines.append("")
                _append_preserved_at_context(lines, preserved_blocks, section_key, method_heading, consumed_blocks)
                doc_body = _method_doc_body(method).strip()
                if doc_body:
                    intro, sections = _method_sections(doc_body)
                    if intro:
                        lines.append(intro)
                        lines.append("")
                    for sec_title, sec_body in sections:
                        rendered = _render_method_section(sec_title, sec_body)
                        lines.extend(rendered)
                        _append_preserved_at_context(lines, preserved_blocks, section_key, f"#### {sec_title}", consumed_blocks)
                else:
                    lines.append("_No docstring available._")
                    lines.append("")
                lines.append("</div>")
                lines.append("")
            _append_remaining_preserved(lines, preserved_blocks, section_key, consumed_blocks)
            lines.append("</div>")
            lines.append("")

        if result_cls is not None:
            section_key = f"result:{result_cls.name}"
            lines.append(f"## Result: `{result_cls.name}`")
            lines.append("")
            lines.append('<div class="analysis-section-indent" markdown="1">')
            lines.append("")
            _append_preserved_at_context(lines, preserved_blocks, section_key, "__section_start__", consumed_blocks)
            res_doc = ast.get_docstring(result_cls) or ""
            res_summary, res_sections = _doc_sections(res_doc)
            res_fields = _class_fields(result_cls)
            res_field_names = {row.name for row in res_fields}
            if res_summary:
                lines.append(res_summary)
                lines.append("")
            lines.append("### Fields")
            lines.append("")
            lines.extend(_render_field_table(res_fields))
            lines.append("")
            for section_name, body in res_sections.items():
                if not body.strip():
                    continue
                if section_name.strip().lower() == "fields":
                    continue
                lines.extend(
                    _render_doc_section(
                        section_name,
                        body,
                        allowed_param_names=res_field_names if section_name.strip().lower() == "parameters" else None,
                    )
                )
                _append_preserved_at_context(lines, preserved_blocks, section_key, f"### {section_name}", consumed_blocks)
            _append_remaining_preserved(lines, preserved_blocks, section_key, consumed_blocks)
            lines.append("</div>")
            lines.append("")

    target.output_md.parent.mkdir(parents=True, exist_ok=True)
    target.output_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target.output_md


def _is_task_module(py_file: Path) -> bool:
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if not classes:
        return False

    def _has_dataclass(cls: ast.ClassDef) -> bool:
        for d in cls.decorator_list:
            try:
                s = ast.unparse(d)
            except Exception:
                s = ""
            if "dataclass" in s:
                return True
        return False

    has_request_dataclass = any(c.name.endswith("Request") and _has_dataclass(c) for c in classes)
    has_result_dataclass = any(c.name.endswith("Result") and _has_dataclass(c) for c in classes)
    has_task_dataclass = any(c.name.endswith("Task") and _has_dataclass(c) for c in classes)
    has_task_cls = bool(_find_task_classes(classes))

    # Include any module that defines an analysis task class (decorated with
    # register_task or inheriting AnalysisTask). Some task modules keep their
    # Request/Result dataclasses in a separate shared models module.
    if has_task_cls:
        return True

    # Backward compatibility: retain support for modules that currently expose
    # only dataclass payloads in this file.
    if has_task_dataclass:
        return True
    if has_request_dataclass or has_result_dataclass:
        return True
    return False


def _analysis_source_root(repo_root: Path) -> tuple[Path, str]:
    analyzer_root = repo_root / "src" / "reaxkit" / "analyzer"
    if analyzer_root.exists():
        return analyzer_root, "reaxkit.analyzer"
    # Backward-compatible fallback for repositories that still use "analysis".
    analysis_root = repo_root / "src" / "reaxkit" / "analysis"
    if analysis_root.exists():
        return analysis_root, "reaxkit.analysis"
    raise FileNotFoundError("Could not find src/reaxkit/analyzer or src/reaxkit/analysis.")


def _default_targets(repo_root: Path) -> list[AnalysisDocTarget]:
    source_root, import_root = _analysis_source_root(repo_root)
    docs_root = repo_root / "docs" / "api" / "analysis"
    targets: list[AnalysisDocTarget] = []

    for py in sorted(source_root.rglob("*.py")):
        if py.name == "__init__.py":
            continue
        if py.name.startswith("_"):
            continue
        if not _is_task_module(py):
            continue

        rel = py.relative_to(source_root)
        module_stem = rel.with_suffix("")
        module_import = f"{import_root}." + ".".join(module_stem.parts)

        out_rel = rel.with_name(f"{rel.stem}_analysis.md")
        output_md = docs_root / out_rel

        title = f"{' '.join(p.capitalize() for p in rel.stem.split('_'))} Analysis"
        targets.append(
            AnalysisDocTarget(
                analysis_file=py,
                module_import=module_import,
                output_md=output_md,
                title=title,
            )
        )
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate structured analysis task markdown docs from source.")
    parser.add_argument("--analysis-file", default=None, help="Path to analysis module .py file.")
    parser.add_argument("--module-import", default=None, help="Python import path for mkdocstrings module directive.")
    parser.add_argument("--output-md", default=None, help="Output markdown file path.")
    parser.add_argument("--title", default=None, help="Page title, e.g., 'MSD Analysis'.")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path(__file__).resolve())
    if args.analysis_file or args.module_import or args.output_md or args.title:
        if not (args.analysis_file and args.module_import and args.output_md and args.title):
            raise SystemExit("If any custom arg is used, provide all: --analysis-file --module-import --output-md --title")
        targets = [
            AnalysisDocTarget(
                analysis_file=Path(args.analysis_file).resolve(),
                module_import=str(args.module_import),
                output_md=Path(args.output_md).resolve(),
                title=str(args.title),
            )
        ]
    else:
        targets = _default_targets(repo_root)

    for target in targets:
        out = _write_page(target)
        print(f"[generated] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
