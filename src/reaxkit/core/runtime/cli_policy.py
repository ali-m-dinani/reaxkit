"""Advanced execution options shared by every CLI command."""

import argparse

from reaxkit.presentation.plot_styles import add_plot_style_argument


def automatic_positive_int(value):
    if str(value).lower() == "auto":
        return 0
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use auto or a positive integer.") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("Use auto or a positive integer (0 also means auto).")
    return number


def add_execution_arguments(parser, *, inherit=False):
    add_plot_style_argument(parser, inherit=inherit)
    options = parser._option_string_actions
    definitions = (
        ("--execution", dict(choices=("auto", "serial", "threads", "processes"), default="auto",
                             help="Execution backend; unsupported backends fall back to serial with a logged reason. Example: --execution serial, which processes work sequentially.")),
        ("--workers", dict(type=automatic_positive_int, default=0, help="Frame workers: auto or N (default: auto). Example: --workers 4, which requests four frame workers.")),
        ("--chunk-size", dict(type=automatic_positive_int, default=0, help="Maximum in-flight frames: auto or N. Example: --chunk-size 8, which limits concurrent frame processing to eight frames.")),
        ("--detail-format", dict(choices=("parquet", "csv"), default=None, help="Optional detail format (default: Parquet; legacy: CSV). Example: --detail-format csv, which writes requested detail tables as CSV.")),
        ("--output-profile", dict(choices=("standard", "minimal", "full", "legacy"), default="standard",
                                  help="Artifact profile (default: standard). Example: --output-profile full, which includes declared optional detail artifacts.")),
    )
    for flag, kwargs in definitions:
        if inherit:
            kwargs["default"] = argparse.SUPPRESS
        if flag not in options:
            parser.add_argument(flag, **kwargs)
        elif flag in {"--workers", "--chunk-size"}:
            options[flag].type = automatic_positive_int
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                add_execution_arguments(child, inherit=True)
