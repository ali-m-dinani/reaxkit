import argparse

import pytest

from reaxkit.cli.inventory import iter_parsers, parser_rows


def test_all_public_parser_options_have_help_metadata():
    rows = [row for path, parser in iter_parsers() for row in parser_rows(path, parser)]
    assert rows


@pytest.mark.parametrize("suffix", ["", " (default: %(default)s)"])
def test_boolean_help_inventory_is_python_version_independent(suffix):
    parser = argparse.ArgumentParser(add_help=False)
    action = parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=False)
    action.help = "Enable diagnostics." + suffix
    assert parser_rows("reaxkit", parser)[0]["help"] == "Enable diagnostics."


def test_nonboolean_help_preserves_default_suffix():
    parser = argparse.ArgumentParser(add_help=False)
    help_text = "Output file (default: %(default)s)"
    parser.add_argument("--output", help=help_text)
    assert parser_rows("reaxkit", parser)[0]["help"] == help_text
