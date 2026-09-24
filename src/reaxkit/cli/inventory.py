"""Read-only enumeration of actual CLI parsers, including nested tasks."""

import argparse

from reaxkit.cli.main import build_parser
from reaxkit.cli.help_metadata import metadata_for_action


def _children(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            yield from action.choices.items()


def _walk(parser, path):
    yield path, parser
    for name, child in _children(parser):
        yield from _walk(child, f"{path} {name}")


def iter_parsers():
    """Build each registered route exactly once; do not enumerate placeholders."""
    root = build_parser()
    yield "reaxkit", root
    for name, _ in _children(root):
        selected_root = build_parser(name)
        selected = dict(_children(selected_root))[name]
        yield from _walk(selected, f"reaxkit {name}")


def parser_rows(path, parser):
    rows = []
    seen = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction) or action.help == argparse.SUPPRESS:
            continue
        flags = action.option_strings or [action.dest]
        overlap = seen.intersection(flags)
        if overlap:
            raise ValueError(f"Duplicate options on {path}: {overlap}")
        seen.update(flags)
        category, visibility = metadata_for_action(action, parser)
        if action.required and visibility != "short":
            raise ValueError(f"Required action omitted from short help: {path} {flags}")
        rows.append({
            "parser": path, "flags": flags, "required": action.required,
            "default": repr(action.default),
            "choices": list(action.choices) if action.choices is not None else None,
            "help": action.help, "category": category, "visibility": visibility,
        })
    return rows


def inventory():
    rows = [row for path, parser in iter_parsers() for row in parser_rows(path, parser)]
    return sorted(rows, key=lambda row: (row["parser"], row["flags"][0]))
