from __future__ import annotations

import argparse

from reaxkit.cli.main import _ReaxKitArgumentParser


def test_option_table_wraps_long_defaults_and_hides_suppressed_options(monkeypatch):
    monkeypatch.setattr(_ReaxKitArgumentParser, "_term_width", staticmethod(lambda: 100))
    parser = _ReaxKitArgumentParser("reaxkit")
    parser.add_argument(
        "--reference",
        default="C:\\" + "very_long_directory\\" * 10 + "reference.cif",
        help="Select the reference structure.",
    )
    parser.add_argument("--compatibility-option", help=argparse.SUPPRESS)

    rendered = parser.format_help()

    assert max(map(len, rendered.splitlines())) <= 100
    assert "--reference" in rendered
    assert "--compatibility-option" not in rendered
