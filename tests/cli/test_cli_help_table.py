from __future__ import annotations

import argparse

import pytest

from reaxkit.cli.main import _ReaxKitArgumentParser
from reaxkit.workflows.file_tools.trainset_workflow import build_parser


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


@pytest.mark.parametrize("help_flag", ["-h", "--help", "--help-all", "--all-flags"])
@pytest.mark.parametrize("command", ["make-trainset-elastic", "gen_elastic_trainset"])
def test_elastic_help_preserves_description_before_tables(help_flag, command, capsys, monkeypatch):
    monkeypatch.setattr(_ReaxKitArgumentParser, "_term_width", staticmethod(lambda: 80))
    parser = build_parser(_ReaxKitArgumentParser(f"reaxkit CLI {command}"), command=command)

    with pytest.raises(SystemExit) as exc:
        parser.parse_args([help_flag])

    assert exc.value.code == 0
    rendered = capsys.readouterr().out
    description = parser.description.rstrip()
    assert "[NOTE]" in description
    assert "Examples:" in description
    assert "[Note]" in description
    assert description in rendered
    assert rendered.index(description) + len(description) < rendered.index("Flag")
