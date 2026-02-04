# docs/_gen_workflow_docs.py
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional


# ----------------------------
# Paths / config
# ----------------------------
# ----------------------------
# Paths / config
# ----------------------------
def _find_repo_root(start: Path) -> Path:
    """
    Walk upward until we find a directory that looks like the repo root.
    """
    for p in [start, *start.parents]:
        if (p / "src" / "reaxkit").exists():
            return p
        if (p / "pyproject.toml").exists() and (p / "src").exists():
            return p
    raise RuntimeError("Could not locate repo root (expected src/reaxkit or pyproject.toml).")

REPO_ROOT = _find_repo_root(Path(__file__).resolve())
SRC_ROOT = REPO_ROOT / "src"
WORKFLOWS_DIR = SRC_ROOT / "reaxkit" / "workflows"

DOCS_OUT_ROOT = REPO_ROOT / "docs" / "api" / "workflows"
CATEGORY_DIRS = {
    "per_file": DOCS_OUT_ROOT / "per_file",
    "meta": DOCS_OUT_ROOT / "meta",
    "composed": DOCS_OUT_ROOT / "composed",
}

SRC_ROOT = REPO_ROOT / "src"
WORKFLOWS_DIR = SRC_ROOT / "reaxkit" / "workflows"

DOCS_OUT_ROOT = REPO_ROOT / "docs" / "api" / "workflows"
CATEGORY_DIRS = {
    "per_file": DOCS_OUT_ROOT / "per_file",
    "meta": DOCS_OUT_ROOT / "meta",
    "composed": DOCS_OUT_ROOT / "composed",
}

# Make sure imports work for src/ layout
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ----------------------------
# Helpers
# ----------------------------
def _module_path_to_import(file_path: Path) -> str:
    """
    Convert: <repo>/src/reaxkit/workflows/per_file/control_workflow.py
    To:      reaxkit.workflows.per_file.control_workflow
    """
    rel = file_path.relative_to(SRC_ROOT)
    return ".".join(rel.with_suffix("").parts)


def _detect_category(file_path: Path) -> str:
    rel = file_path.relative_to(WORKFLOWS_DIR)
    parts = rel.parts
    if parts and parts[0] in CATEGORY_DIRS:
        return parts[0]
    return "meta"


def _get_subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            return a
    raise RuntimeError("No subparsers found. Did the workflow call add_subparsers()?")


def _discover_tasks(module) -> List[Tuple[str, argparse.ArgumentParser]]:
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="task")

    if not hasattr(module, "register_tasks"):
        return []

    module.register_tasks(subparsers)

    sp_action = _get_subparsers_action(parser)
    tasks = sorted(sp_action.choices.items(), key=lambda kv: kv[0])
    return [(name, sp) for name, sp in tasks]


def _title_case_workflow(stem: str) -> str:
    """
    coordination_workflow -> "Coordination Workflow"
    xmolout_workflow -> "Xmolout Workflow"
    """
    s = stem.replace("_", " ").strip()
    return " ".join(w[:1].upper() + w[1:] for w in s.split())


def _workflow_title_from_module(import_path: str) -> str:
    base = import_path.split(".")[-1].replace("_workflow", "")
    return _title_case_workflow(base) + " Workflow"


def _md_escape(s: str) -> str:
    return (s or "").strip()


def _strip_examples_line(text: str) -> str:
    """
    Remove any line that starts with 'Examples' (common in argparse descriptions).
    """
    if not text:
        return ""
    keep: List[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("Examples"):
            continue
        keep.append(line)
    return "\n".join(keep).strip()


def _parse_argparse_help(help_text: str) -> tuple[list[str], list[tuple[str, str]]]:
    """
    From argparse's format_help(), extract:
      - examples: list[str]
      - options: list[(flag, desc)]
    Rules:
      - "usage:" section is ignored completely.
      - "Examples:" becomes a markdown section with bullet list.
      - "options:" becomes a markdown table.
    """
    lines = help_text.splitlines()

    # Find "Examples:" and "options:" headers
    ex_i: Optional[int] = None
    opt_i: Optional[int] = None
    for i, line in enumerate(lines):
        if line.strip() == "Examples:":
            ex_i = i
        if line.strip() in ("options:", "optional arguments:"):
            opt_i = i

    # Collect examples (between Examples: and options: (or end))
    examples: List[str] = []
    if ex_i is not None:
        start = ex_i + 1
        end = opt_i if opt_i is not None else len(lines)
        for line in lines[start:end]:
            s = line.strip()
            if not s:
                continue
            # argparse often indents example commands; keep as commands
            examples.append(s)

    # Collect options (after options: header)
    options: List[tuple[str, str]] = []
    if opt_i is not None:
        current_flag: Optional[str] = None
        current_desc: List[str] = []

        def flush():
            nonlocal current_flag, current_desc
            if current_flag is not None:
                desc = " ".join(d.strip() for d in current_desc if d.strip()).strip()
                options.append((current_flag.strip(), desc))
            current_flag = None
            current_desc = []

        for line in lines[opt_i + 1 :]:
            if not line.strip():
                continue

            stripped = line.lstrip()

            # New option lines usually start with '-' (after indentation)
            if stripped.startswith("-"):
                flush()
                # Split by 2+ spaces into "flags" and "desc"
                parts = re.split(r"\s{2,}", stripped, maxsplit=1)
                current_flag = parts[0].strip()
                current_desc = [parts[1].strip()] if len(parts) > 1 else [""]
            else:
                # Continuation line for previous description
                if current_flag is not None:
                    current_desc.append(stripped.strip())

        flush()

    return examples, options

def _render_options_table(options: list[tuple[str, str]], lines: list[str]) -> None:
    if not options:
        return
    lines.append("#### Options")
    lines.append("")
    lines.append("| Flag | Description |")
    lines.append("|---|---|")
    for flag, desc2 in options:
        desc2 = (desc2 or "").replace("|", "\\|")
        flag = (flag or "").replace("|", "\\|")
        lines.append(f"| `{flag}` | {desc2} |")
    lines.append("")


def _render_examples(examples: list[str], lines: list[str]) -> None:
    if not examples:
        return
    lines.append("#### Examples")
    lines.append("")
    for ex in examples:
        lines.append(f"- `{ex}`")
    lines.append("")


def _render_kind_level_docs(module, workflow_cmd: str, lines: list[str]) -> None:
    """
    Kind-level = no subcommands.
    Strategy:
      - show usage
      - show run_main docstring (if present)
      - show options if module provides build_parser(p)
    """
    # Usage
    lines.append("## Usage")
    lines.append("")
    lines.append(f"`reaxkit {workflow_cmd} <args>`")
    lines.append("")

    # run_main docstring (recommended to add for help/introspection style workflows)
    run_main = getattr(module, "run_main", None)
    rm_doc = _md_escape(getattr(run_main, "__doc__", "") or "") if run_main else ""
    if rm_doc:
        lines.append("## Behavior")
        lines.append("")
        lines.append(rm_doc)
        lines.append("")

    # Options: preferred path is build_parser(p)
    build_parser = getattr(module, "build_parser", None)
    if callable(build_parser):
        p = argparse.ArgumentParser(prog=f"reaxkit {workflow_cmd}", add_help=True)
        build_parser(p)
        help_text = p.format_help().rstrip()
        examples, options = _parse_argparse_help(help_text)
        _render_examples(examples, lines)
        _render_options_table(options, lines)
    else:
        lines.append("## Options")
        lines.append("")
        lines.append(
            "_This command does not declare subcommands. "
            "To list available flags, run:_"
        )
        lines.append("")
        lines.append(f"`reaxkit {workflow_cmd} --help`")
        lines.append("")


def _render_markdown_for_workflow(import_path: str, file_path: Path) -> str:
    module = importlib.import_module(import_path)

    title = _workflow_title_from_module(import_path)
    workflow_cmd = import_path.split(".")[-1].replace("_workflow", "")

    tasks = _discover_tasks(module)

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"CLI namespace: `reaxkit {workflow_cmd} <task> [flags]`")
    lines.append("")

    # Optional module docstring
    doc = _md_escape(getattr(module, "__doc__", "") or "")
    if doc:
        lines.append(doc)
        lines.append("")

    lines.append("## Available tasks")
    lines.append("")

    if not tasks:
        # Kind-level command (no subcommands)
        _render_kind_level_docs(module, workflow_cmd, lines)
    else:
        for task_name, task_parser in tasks:
            lines.append(f"### `{task_name}`")
            lines.append("")

            # one-line (or multi-line) explanation of what this task does
            desc = (getattr(task_parser, "description", "") or "").strip()

            if desc:
                kept_lines: list[str] = []
                for ln in desc.splitlines():
                    if ln.strip().lower().startswith("examples:"):
                        break
                    kept_lines.append(ln.rstrip())

                # trim leading/trailing blank lines
                while kept_lines and not kept_lines[0].strip():
                    kept_lines.pop(0)
                while kept_lines and not kept_lines[-1].strip():
                    kept_lines.pop()

                if kept_lines:
                    # preserve line breaks in Markdown
                    desc_clean = "<br>\n".join(kept_lines)
                    lines.append(desc_clean)
                    lines.append("")

            # Prefer description, but remove any "Examples..." line right after task
            desc = _md_escape(getattr(task_parser, "description", "") or "")
            desc = _strip_examples_line(desc)
            # if desc:
            #     lines.append(desc)
            #     lines.append("")

            help_text = task_parser.format_help().rstrip()
            examples, options = _parse_argparse_help(help_text)

            # Examples section (no "usage:", and bullets under Examples)
            if examples:
                lines.append("#### Examples")
                lines.append("")
                for ex in examples:
                    lines.append(f"- `{ex}`")
                lines.append("")

            # Options section as a table
            if options:
                lines.append("#### Options")
                lines.append("")
                lines.append("| Flag | Description |")
                lines.append("|---|---|")
                for flag, desc2 in options:
                    # escape pipes to keep table intact
                    desc2 = (desc2 or "").replace("|", "\\|")
                    flag = (flag or "").replace("|", "\\|")
                    lines.append(f"| `{flag}` | {desc2} |")
                lines.append("")
            else:
                # Fallback: show raw help (without usage) if options not found
                # (rare; keeps docs from being empty)
                cleaned = []
                in_usage = False
                for ln in help_text.splitlines():
                    if ln.startswith("usage:"):
                        in_usage = True
                        continue
                    if in_usage and ln.strip() == "":
                        in_usage = False
                        continue
                    if not in_usage:
                        cleaned.append(ln)
                cleaned_text = "\n".join(cleaned).strip()
                if cleaned_text:
                    lines.append("```text")
                    lines.append(cleaned_text)
                    lines.append("```")
                    lines.append("")

    # lines.append("## Python API")
    # lines.append("")
    # lines.append(f"::: {import_path}")
    # lines.append("")

    return "\n".join(lines)


def main() -> int:
    for p in CATEGORY_DIRS.values():
        p.mkdir(parents=True, exist_ok=True)

    workflow_files = sorted(WORKFLOWS_DIR.rglob("*_workflow.py"))

    for wf in workflow_files:
        if wf.name.startswith("_") or wf.name == "__init__.py":
            continue

        category = _detect_category(wf)
        import_path = _module_path_to_import(wf)

        md = _render_markdown_for_workflow(import_path, wf)

        out_name = wf.stem.replace("_workflow", "") + "_doc.md"
        out_path = CATEGORY_DIRS[category] / out_name
        out_path.write_text(md, encoding="utf-8")

        print(f"[gen] {import_path} -> {out_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
