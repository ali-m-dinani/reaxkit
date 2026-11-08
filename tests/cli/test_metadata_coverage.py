from reaxkit.cli.inventory import iter_parsers, parser_rows


def test_all_public_parser_options_have_help_metadata():
    rows = [row for path, parser in iter_parsers() for row in parser_rows(path, parser)]
    assert rows
