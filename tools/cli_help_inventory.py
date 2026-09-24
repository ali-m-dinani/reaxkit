"""Write the live CLI inventory; exit unsuccessfully on any rejected route.

Run ``python tools/cli_help_inventory.py [output.json]`` from the repository.
"""

import json
import sys
from pathlib import Path

from reaxkit.cli.inventory import iter_parsers, parser_rows


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cli-help-inventory.json")
    rows = []
    errors = {}
    try:
        for path, parser in iter_parsers():
            rows.extend(parser_rows(path, parser))
    except Exception as exc:
        errors["inventory"] = f"{type(exc).__name__}: {exc}"
    error_target = target.with_name(f"{target.stem}.errors.json")
    error_target.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    if errors:
        print(errors, file=sys.stderr)
        return 1
    rows.sort(key=lambda row: (row["parser"], row["flags"][0]))
    target.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} actions to {target}; no rejected routes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
