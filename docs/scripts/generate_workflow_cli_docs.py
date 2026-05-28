"""Generate structured workflow CLI docs from workflow source (AST-based).

Outputs:
- module mkdocstrings block (file-level docstring is shown by mkdocstrings)
- per-command description and examples (from parser.description)
- per-command argument table (from add_argument calls in command branch)
- common runtime/presentation argument table (from shared helper calls)
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkflowDocTarget:
    workflow_file: Path
    module_import: str
    output_md: Path
    title: str


@dataclass(frozen=True)
class ModuleContext:
    file_path: Path
    module_import: str
    tree: ast.Module
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
    imported_functions: dict[str, tuple[str, str]]


@dataclass
class ArgRow:
    flag: str
    required: str
    default: str
    help_text: str
    choices: str


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


def _literal_str(node: ast.AST) -> str | None:
    value = _literal(node)
    return value if isinstance(value, str) else None


def _literal_bool(node: ast.AST) -> bool | None:
    value = _literal(node)
    return value if isinstance(value, bool) else None


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    return repr(value)


def _module_import_from_path(src_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(src_root)
    parts = list(rel.with_suffix("").parts)
    return ".".join(parts)


def _module_path_from_import(src_root: Path, module_import: str) -> Path | None:
    py_path = src_root / Path(*module_import.split(".")).with_suffix(".py")
    if py_path.exists():
        return py_path
    pkg_init = src_root / Path(*module_import.split(".")) / "__init__.py"
    if pkg_init.exists():
        return pkg_init
    return None


def _build_module_context(src_root: Path, file_path: Path, module_import: str) -> ModuleContext:
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    imported_functions: dict[str, tuple[str, str]] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
        elif isinstance(node, ast.ImportFrom) and node.module:
            # absolute imports only
            if node.level != 0:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname or alias.name
                imported_functions[local_name] = (node.module, alias.name)

    return ModuleContext(
        file_path=file_path,
        module_import=module_import,
        tree=tree,
        functions=functions,
        imported_functions=imported_functions,
    )


class ContextResolver:
    def __init__(self, src_root: Path) -> None:
        self.src_root = src_root
        self._cache_by_path: dict[Path, ModuleContext] = {}

    def context_for(self, file_path: Path, module_import: str | None = None) -> ModuleContext:
        resolved = file_path.resolve()
        if resolved in self._cache_by_path:
            return self._cache_by_path[resolved]
        mod_import = module_import or _module_import_from_path(self.src_root, resolved)
        ctx = _build_module_context(self.src_root, resolved, mod_import)
        self._cache_by_path[resolved] = ctx
        return ctx

    def resolve_imported_function(
        self, current: ModuleContext, local_name: str
    ) -> tuple[ModuleContext, ast.FunctionDef | ast.AsyncFunctionDef] | None:
        if local_name in current.functions:
            return current, current.functions[local_name]
        imported = current.imported_functions.get(local_name)
        if not imported:
            return None
        module_import, func_name = imported
        module_path = _module_path_from_import(self.src_root, module_import)
        if module_path is None:
            return None
        ctx = self.context_for(module_path, module_import=module_import)
        func = ctx.functions.get(func_name)
        if func is None:
            return None
        return ctx, func


def _extract_all_commands(module_node: ast.Module) -> list[str]:
    for node in module_node.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ALL_COMMANDS":
                    value = _literal(node.value)
                    if isinstance(value, (list, tuple)):
                        return [str(v) for v in value]
    return []


def _command_from_test(test: ast.AST) -> str | None:
    if not isinstance(test, ast.Compare):
        return None
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return None
    if len(test.comparators) != 1:
        return None

    left = test.left
    right = test.comparators[0]
    if isinstance(left, ast.Constant) and isinstance(left.value, str):
        return left.value
    if isinstance(right, ast.Constant) and isinstance(right.value, str):
        return right.value
    return None


def _extract_build_parser_fn(ctx: ModuleContext) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return ctx.functions.get("build_parser")


def _extract_build_parser_descriptions(module_node: ast.Module) -> dict[str, str]:
    build_parser: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in module_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build_parser":
            build_parser = node
            break
    if build_parser is None:
        return {}

    out: dict[str, str] = {}

    def walk_if_chain(node: ast.If) -> None:
        command = _command_from_test(node.test)
        if command:
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Assign):
                        for target in sub.targets:
                            if isinstance(target, ast.Attribute) and target.attr == "description":
                                desc = _literal_str(sub.value)
                                if desc is not None:
                                    out[command] = desc
                                    break
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            walk_if_chain(node.orelse[0])

    for stmt in build_parser.body:
        if isinstance(stmt, ast.If):
            walk_if_chain(stmt)
    return out


def _extract_command_branches(build_parser_fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[dict[str, list[ast.stmt]], list[ast.stmt]]:
    command_branches: dict[str, list[ast.stmt]] = {}
    common_stmts: list[ast.stmt] = []

    def add_common(stmt: ast.stmt) -> None:
        # Ignore pure control-flow tail that does not define common args.
        if isinstance(stmt, (ast.Return, ast.Raise)):
            return
        common_stmts.append(stmt)

    branch_if_index: int | None = None
    for idx, stmt in enumerate(build_parser_fn.body):
        if not isinstance(stmt, ast.If):
            continue
        # Identify command dispatch chain by checking whether it yields command keys.
        probe = stmt
        has_command = False
        while isinstance(probe, ast.If):
            if _command_from_test(probe.test):
                has_command = True
                break
            if len(probe.orelse) == 1 and isinstance(probe.orelse[0], ast.If):
                probe = probe.orelse[0]
            else:
                break
        if has_command:
            branch_if_index = idx
            break

    if branch_if_index is None:
        for stmt in build_parser_fn.body:
            add_common(stmt)
        return command_branches, common_stmts

    for idx, stmt in enumerate(build_parser_fn.body):
        if idx == branch_if_index:
            continue
        add_common(stmt)

    chain = build_parser_fn.body[branch_if_index]
    assert isinstance(chain, ast.If)
    cursor: ast.If | None = chain
    while cursor is not None:
        command = _command_from_test(cursor.test)
        if command:
            command_branches[command] = list(cursor.body)
        if len(cursor.orelse) == 1 and isinstance(cursor.orelse[0], ast.If):
            cursor = cursor.orelse[0]
        else:
            cursor = None

    return command_branches, common_stmts


def _arg_row_from_call(call: ast.Call) -> ArgRow | None:
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "add_argument":
        return None
    flags: list[str] = []
    for pos in call.args:
        flag = _literal_str(pos)
        if flag is not None:
            flags.append(flag)
    if not flags:
        return None

    required_val = ""
    default_val = ""
    help_val = ""
    choices_val = ""

    for kw in call.keywords:
        if kw.arg == "required":
            b = _literal_bool(kw.value)
            if b is not None:
                required_val = "Yes" if b else "No"
        elif kw.arg == "default":
            default_val = _display_value(_literal(kw.value))
        elif kw.arg == "help":
            s = _literal_str(kw.value)
            help_val = s or ""
        elif kw.arg == "choices":
            choices_val = _display_value(_literal(kw.value))

    if not required_val:
        required_val = "No"

    return ArgRow(
        flag=", ".join(flags),
        required=required_val,
        default=default_val,
        help_text=help_val,
        choices=choices_val,
    )


def _collect_argument_rows_from_statements(
    stmts: list[ast.stmt],
    ctx: ModuleContext,
    parser_arg_name: str,
    resolver: ContextResolver,
    out_rows: list[ArgRow],
    seen_flags: set[str],
    visited_functions: set[tuple[str, str]],
) -> None:
    def recurse_into_function(target_ctx: ModuleContext, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        fn_key = (str(target_ctx.file_path), fn.name)
        if fn_key in visited_functions:
            return
        visited_functions.add(fn_key)
        fn_parser_arg = fn.args.args[0].arg if fn.args.args else parser_arg_name
        _collect_argument_rows_from_statements(
            list(fn.body),
            target_ctx,
            fn_parser_arg,
            resolver,
            out_rows,
            seen_flags,
            visited_functions,
        )

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            # Do not descend into nested defs.
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            return

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            row = _arg_row_from_call(node)
            if row is not None:
                if row.flag not in seen_flags:
                    seen_flags.add(row.flag)
                    out_rows.append(row)
            else:
                # helper(parser, ...)
                if isinstance(node.func, ast.Name) and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Name) and first.id == parser_arg_name:
                        resolved = resolver.resolve_imported_function(ctx, node.func.id)
                        if resolved is not None:
                            target_ctx, fn = resolved
                            recurse_into_function(target_ctx, fn)
            self.generic_visit(node)

    visitor = Visitor()
    for stmt in stmts:
        visitor.visit(stmt)


def _split_description_and_examples(desc: str) -> tuple[str, str]:
    lines = desc.splitlines()
    idx = None
    for i, ln in enumerate(lines):
        stripped = ln.strip().lower()
        if stripped.startswith("examples:") or stripped.startswith("example:"):
            idx = i
            break
    if idx is None:
        return desc.strip(), ""
    before = "\n".join(lines[:idx]).strip()
    after = "\n".join(lines[idx + 1 :]).rstrip()
    return before, after


def _render_arg_table(rows: list[ArgRow]) -> list[str]:
    lines: list[str] = []
    lines.append("| Flag | Required | Default | Help | Choices |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        flag = row.flag.replace("|", "\\|")
        req = row.required.replace("|", "\\|")
        default = row.default.replace("|", "\\|")
        help_text = row.help_text.replace("|", "\\|")
        choices = row.choices.replace("|", "\\|")
        lines.append(f"| `{flag}` | {req} | {default} | {help_text} | {choices} |")
    return lines


def _render_markdown(
    target: WorkflowDocTarget,
    commands: list[str],
    descriptions: dict[str, str],
    command_args: dict[str, list[ArgRow]],
    common_args: list[ArgRow],
) -> str:
    lines: list[str] = []
    lines.append("<!-- AUTO-GENERATED by docs/scripts/generate_workflow_cli_docs.py -->")
    lines.append(f"# {target.title}")
    lines.append("")
    lines.append(f"::: {target.module_import}")
    lines.append("    options:")
    lines.append("      show_root_heading: false")
    lines.append("      show_root_full_path: false")
    lines.append("      members: []")
    lines.append("")

    for command in commands:
        lines.append(f"## Command: `{command}`")
        lines.append("")
        lines.append('<div class="analysis-section-indent" markdown="1">')
        lines.append("")
        desc = descriptions.get(command, "")
        normal, examples = _split_description_and_examples(desc)
        if normal:
            lines.append(normal)
            lines.append("")
        if examples:
            lines.append("### Examples")
            lines.append("-----")
            lines.append("")
            lines.append("```text")
            lines.append(examples)
            lines.append("```")
            lines.append("")

        rows = command_args.get(command, [])
        lines.append("### Arguments")
        lines.append("")
        if rows:
            lines.extend(_render_arg_table(rows))
        else:
            lines.append("_No command-specific arguments found._")
        lines.append("")
        lines.append("</div>")
        lines.append("")

    lines.append("## Common Runtime and Presentation Arguments")
    lines.append("")
    lines.append('<div class="analysis-section-indent" markdown="1">')
    lines.append("")
    lines.append(
        "These are shared workflow-level CLI flags added before command-specific options, "
        "covering runtime context (engine/input/storage) and output presentation/export behavior."
    )
    lines.append("")
    if common_args:
        lines.extend(_render_arg_table(common_args))
    else:
        lines.append("_No common arguments found._")
    lines.append("")
    lines.append("</div>")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def generate_workflow_cli_docs(target: WorkflowDocTarget, resolver: ContextResolver) -> Path:
    ctx = resolver.context_for(target.workflow_file, module_import=target.module_import)
    commands = _extract_all_commands(ctx.tree)
    descriptions = _extract_build_parser_descriptions(ctx.tree)

    build_parser_fn = _extract_build_parser_fn(ctx)
    command_args: dict[str, list[ArgRow]] = {cmd: [] for cmd in commands}
    common_args: list[ArgRow] = []

    if build_parser_fn is not None:
        parser_arg_name = build_parser_fn.args.args[0].arg if build_parser_fn.args.args else "parser"
        branches, common_stmts = _extract_command_branches(build_parser_fn)

        seen_common: set[str] = set()
        _collect_argument_rows_from_statements(
            common_stmts,
            ctx,
            parser_arg_name,
            resolver,
            common_args,
            seen_common,
            visited_functions=set(),
        )

        for cmd in commands:
            rows: list[ArgRow] = []
            seen_cmd: set[str] = set()
            _collect_argument_rows_from_statements(
                branches.get(cmd, []),
                ctx,
                parser_arg_name,
                resolver,
                rows,
                seen_cmd,
                visited_functions=set(),
            )
            command_args[cmd] = rows

    md = _render_markdown(target, commands, descriptions, command_args, common_args)
    target.output_md.parent.mkdir(parents=True, exist_ok=True)
    target.output_md.write_text(md, encoding="utf-8")
    return target.output_md


def _default_targets(repo_root: Path) -> list[WorkflowDocTarget]:
    return [
        WorkflowDocTarget(
            workflow_file=repo_root / "src" / "reaxkit" / "workflows" / "active_site_workflow.py",
            module_import="reaxkit.workflows.active_site_workflow",
            output_md=repo_root / "docs" / "api" / "workflows" / "alaki_workflow.md",
            title="Active Site Workflow",
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate structured workflow CLI markdown docs from source.")
    parser.add_argument("--workflow-file", default=None, help="Path to *_workflow.py file.")
    parser.add_argument("--module-import", default=None, help="Python import path for mkdocstrings directive.")
    parser.add_argument("--output-md", default=None, help="Output markdown file path.")
    parser.add_argument("--title", default=None, help="Page title, e.g., 'Active Site Workflow'.")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path(__file__).resolve())
    src_root = repo_root / "src"
    resolver = ContextResolver(src_root)

    if args.workflow_file or args.module_import or args.output_md or args.title:
        if not (args.workflow_file and args.module_import and args.output_md and args.title):
            raise SystemExit(
                "If any custom arg is used, provide all: --workflow-file --module-import --output-md --title"
            )
        targets = [
            WorkflowDocTarget(
                workflow_file=Path(args.workflow_file).resolve(),
                module_import=str(args.module_import),
                output_md=Path(args.output_md).resolve(),
                title=str(args.title),
            )
        ]
    else:
        targets = _default_targets(repo_root)

    for target in targets:
        out = generate_workflow_cli_docs(target, resolver)
        print(f"[generated] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
