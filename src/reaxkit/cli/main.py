"""
Top-level command-line interface for ReaxKit.

This module defines the ``reaxkit`` entry point and routes each top-level command
to either:
- a direct analysis/generator command module, or
- a workflow command module (command-level or task-subcommand based).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import textwrap
from importlib import import_module

from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.core.registry.command_catalog import get_registered_commands
from reaxkit.core.platform.exceptions import AnalysisError, ParseError
from reaxkit.core.registry.generator_cli_routing_registry import get_registered_generators
from reaxkit.core.registry.workflow_cli_routing_registry import get_registered_workflows


class _ReaxKitArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with clearer UX for unknown command/flag cases."""

    def __init__(self, *args, selected_command: str | None = None, known_commands: set[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._selected_command = selected_command
        self._known_commands = known_commands or set()

    @staticmethod
    def _term_width() -> int:
        return max(100, min(shutil.get_terminal_size(fallback=(120, 40)).columns, 180))

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text or "").split())

    @staticmethod
    def _format_flags(action: argparse.Action) -> str:
        flags = ", ".join(action.option_strings)
        if action.nargs == 0:
            return flags
        metavar = action.metavar or action.dest.upper()
        if isinstance(metavar, tuple):
            metavar = " ".join(str(v) for v in metavar)
        return f"{flags} {metavar}"

    @staticmethod
    def _format_default(action: argparse.Action) -> str:
        default = action.default
        if default in (None, argparse.SUPPRESS):
            return "-"
        if isinstance(default, bool):
            return "true" if default else "false"
        txt = str(default)
        return txt if txt else "-"

    @staticmethod
    def _format_choices(action: argparse.Action) -> str:
        choices = getattr(action, "choices", None)
        if not choices:
            return "-"
        return ", ".join(str(v) for v in choices)

    def _commands_rows(self) -> list[tuple[str, str]]:
        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction):
                rows: list[tuple[str, str]] = []
                choice_actions = getattr(action, "_choices_actions", [])
                for name, subparser in sorted(action.choices.items()):
                    help_text = "-"
                    if isinstance(choice_actions, list):
                        for choice_action in choice_actions:
                            if getattr(choice_action, "dest", None) == name:
                                help_text = self._normalize(getattr(choice_action, "help", "") or "")
                                break
                    if not help_text:
                        help_text = "-"
                    rows.append((name, help_text))
                return rows
        return []

    @staticmethod
    def _render_table(headers: list[str], rows: list[list[str]], width: int, wrap_cols: set[int]) -> str:
        if not rows:
            return ""
        n = len(headers)
        max_col = [len(h) for h in headers]
        for row in rows:
            for i in range(n):
                max_col[i] = max(max_col[i], len(row[i]))

        sep_size = 3 * (n - 1)
        natural = sum(max_col) + sep_size
        col_widths = list(max_col)

        if natural > width:
            fixed_cols = [i for i in range(n) if i not in wrap_cols]
            fixed_total = sum(col_widths[i] for i in fixed_cols)
            wrap_total_min = sum(16 for _ in wrap_cols)
            budget = max(width - sep_size - fixed_total, wrap_total_min)
            current_wrap_total = sum(col_widths[i] for i in wrap_cols)
            for i in wrap_cols:
                if current_wrap_total <= 0:
                    col_widths[i] = 16
                else:
                    share = int(budget * (col_widths[i] / current_wrap_total))
                    col_widths[i] = max(16, share)

        def _wrap_cell(text: str, col: int) -> list[str]:
            if col not in wrap_cols:
                return [text]
            return textwrap.wrap(text, width=col_widths[col], break_long_words=False, break_on_hyphens=False) or [""]

        lines: list[str] = []
        lines.append(" | ".join(headers[i].ljust(col_widths[i]) for i in range(n)))
        row_sep = "-+-".join("-" * col_widths[i] for i in range(n))
        lines.append(row_sep)

        for row in rows:
            wrapped = [_wrap_cell(row[i], i) for i in range(n)]
            max_lines = max(len(cell_lines) for cell_lines in wrapped)
            for ln in range(max_lines):
                parts: list[str] = []
                for i in range(n):
                    cell = wrapped[i][ln] if ln < len(wrapped[i]) else ""
                    parts.append(cell.ljust(col_widths[i]))
                lines.append(" | ".join(parts))
            lines.append(row_sep)
        return "\n".join(lines)

    def format_help(self) -> str:
        width = self._term_width()
        out: list[str] = [""]

        if self.description:
            out.append(self.description.rstrip())
            out.append("")

        commands = self._commands_rows()
        if commands:
            out.append("Commands")
            cmd_table = self._render_table(
                headers=["Command", "Help"],
                rows=[[self._normalize(cmd), self._normalize(help_text)] for cmd, help_text in commands],
                width=width,
                wrap_cols={1},
            )
            if cmd_table:
                out.append(cmd_table)
                out.append("")

        option_rows: list[list[str]] = []
        for action in self._actions:
            if not action.option_strings:
                continue
            option_rows.append(
                [
                    self._normalize(self._format_flags(action)),
                    "yes" if bool(getattr(action, "required", False)) else "no",
                    self._normalize(self._format_default(action)),
                    self._normalize(action.help or "-"),
                    self._normalize(self._format_choices(action)),
                ]
            )
        if option_rows:
            out.append("Options")
            opt_table = self._render_table(
                headers=["Flag", "Required", "Default", "Help", "Choices"],
                rows=option_rows,
                width=width,
                wrap_cols={3, 4},
            )
            if opt_table:
                out.append(opt_table)
                out.append("")

        return "\n".join(out)

    def error(self, message: str) -> None:
        if message.startswith("argument command: invalid choice: "):
            bad_command = None
            parts = message.split("'")
            if len(parts) >= 2:
                bad_command = parts[1]
            bad_command = bad_command or str(self._selected_command or "").strip() or "<unknown>"
            print(
                f"There is no command {bad_command}. Please run 'reaxkit help \"query\"' "
                f"where query can be {bad_command} to see if any relevant command exists or not.",
                file=sys.stderr,
            )
            raise SystemExit(2)

        if message.startswith("unrecognized arguments:") and self._selected_command in self._known_commands:
            unknown_args = message.split(":", 1)[1].strip().split()
            bad_flag = next((token for token in unknown_args if token.startswith("-")), unknown_args[0] if unknown_args else "")
            print(
                f"There is no flag {bad_flag} for command {self._selected_command}. "
                f"Please run 'reaxkit {self._selected_command} -h' to see the list of appropriate flags.",
                file=sys.stderr,
            )
            raise SystemExit(2)

        super().error(message)


def _print_error_with_hints(title: str, message: str, *, hints: list[str] | None = None) -> None:
    print(f"[{title}] {message}", file=sys.stderr)
    for hint in hints or []:
        print(f"  - {hint}", file=sys.stderr)


def _hints_for_filenotfound(exc: FileNotFoundError, args: argparse.Namespace) -> list[str]:
    msg = str(exc)
    command = str(getattr(args, "command", "") or "")
    hints: list[str] = []
    if "Study file not found:" in msg and command == "study":
        hints.append("Ensure the YAML path is correct and relative to your current directory.")
        hints.append(r"Example: reaxkit study --init .\study.yaml --root .")
        hints.append(r"Create template first if needed: reaxkit study --make-yaml study.yaml")
    return hints


def _hints_for_oserror(exc: OSError, args: argparse.Namespace) -> list[str]:
    msg = str(exc)
    lower = msg.lower()
    if getattr(exc, "winerror", None) == 206 or "filename or extension is too long" in lower:
        command = str(getattr(args, "command", "") or "")
        hints = [
            r"Use a short project root for analysis cache/output paths, e.g. add: --project-root C:\rk",
            r"Shorten case folder names in a study: reaxkit study --manage <study_root> --action rename-cases --target case-names",
            "Optionally enable Windows long paths policy and restart.",
        ]
        if command and command != "study":
            hints.insert(0, f"Re-run with a shorter root, e.g.: reaxkit {command} ... --project-root C:\\rk")
        return hints
    return []


def _command_help_text(name: str, fallback: str) -> str:
    """Return help text for a top-level command."""
    spec = get_registered_commands(include_analysis_tasks=True).get(name)
    return spec.help_text or fallback if spec is not None else fallback


def _intspec_runner(args: argparse.Namespace) -> int:
    """Run the ``intspec`` command-level workflow."""
    workflow = import_module("reaxkit.workflows.meta.introspection_workflow")
    return workflow.run_main(
        getattr(args, "file", None),
        getattr(args, "folder", None),
    )


def _canonicalize_direct_command(argv: list[str]) -> list[str]:
    """Rewrite direct-command aliases to canonical names before parsing."""
    out = list(argv)
    if len(out) < 2:
        return out

    direct_commands = {
        **get_registered_analysis_commands(),
        **get_registered_generators(),
    }
    try:
        out[1] = resolve_command_name(out[1], task_names=direct_commands.keys())
    except KeyError:
        pass
    return out


def _direct_command_runner(module, command: str):
    """Create an argparse runner for direct-command modules."""

    def _runner(args: argparse.Namespace) -> int:
        return module.run_main(command, args)

    return _runner


def main() -> int:
    """
    Build and execute the ``reaxkit`` CLI dispatcher.

    Examples
    --------
    - reaxkit connection_list --fort7 fort.7 --export connections.csv
    - reaxkit help "fort.7"
    - reaxkit intspec --folder workflows
    """
    sys_argv = _canonicalize_direct_command(sys.argv)

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("command", nargs="?")
    command_ns, _ = probe.parse_known_args(sys_argv[1:])
    selected_command = getattr(command_ns, "command", None)

    direct_commands = {
        **get_registered_analysis_commands(),
        **get_registered_generators(),
    }
    workflow_commands = get_registered_workflows()
    known_commands = set(direct_commands) | set(workflow_commands)

    parser = _ReaxKitArgumentParser(
        "reaxkit CLI",
        selected_command=selected_command,
        known_commands=known_commands,
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print per-task timing to console (timing is always persisted to logs/timing.log).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Enable progress reporting for supported handlers and analysis tasks.",
    )
    parser.add_argument(
        "--log-in-terminal",
        action="store_true",
        help="Show [ReaxKit] runtime log lines in terminal output.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    for command, spec in direct_commands.items():
        cp = sub.add_parser(command, help=_command_help_text(command, f"{command} command"))

        if command != selected_command:
            continue

        module = import_module(spec.module_path)
        if hasattr(module, "build_parser"):
            module.build_parser(cp, command=command)
        cp.set_defaults(_run=_direct_command_runner(module, command))

    for command, spec in workflow_commands.items():
        wp = sub.add_parser(command, help=_command_help_text(command, f"{command} workflows"))

        if command != selected_command:
            continue

        module = import_module(spec.module_path)

        if spec.dispatch_mode == "intspec_runner":
            if hasattr(module, "build_parser"):
                module.build_parser(wp)
            wp.set_defaults(_run=_intspec_runner)
        elif spec.dispatch_mode == "kind_runner":
            if hasattr(module, "build_parser"):
                module.build_parser(wp)
            wp.set_defaults(_run=module.run_main)
        else:
            tasks = wp.add_subparsers(dest="task", required=True)
            module.register_tasks(tasks)

    args = parser.parse_args(sys_argv[1:])
    try:
        return args._run(args)
    except ParseError as exc:
        print(f"[Parse error] {exc}", file=sys.stderr)
        return 2
    except AnalysisError as exc:
        print(f"[Analysis error] {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        _print_error_with_hints("File not found", str(exc), hints=_hints_for_filenotfound(exc, args))
        return 4
    except OSError as exc:
        _print_error_with_hints("OS error", str(exc), hints=_hints_for_oserror(exc, args))
        return 5
    except Exception as exc:
        _print_error_with_hints("Error", str(exc))
        return 1
