"""Launch the ReaxKit Dash Web UI.

This module implements CLI workflow orchestration for its command family, including argument parsing, request construction, execution dispatch, and result presentation handoff.

**Usage context**

- Command routing: Resolve CLI aliases and normalized command names.
- Task execution: Build request objects and invoke registered tasks.
- Output handling: Forward results to table, plot, export, or report flows.
"""

from __future__ import annotations

import argparse

ALL_COMMANDS = ("gui",)
ALL_LEGACY_COMMANDS = ()


def build_parser(p: argparse.ArgumentParser) -> None:
    """Define CLI arguments for `reaxkit gui`."""
    p.description = ("Launch the ReaxKit Web UI.\n"
                     "Simply run 'reaxkit gui' to generate a GUI on localhost.\n"
                     "Once you open the GUI, you need to follow these steps:\n"
                     " 1. in the bottom left where you see the settings panel, click on 'Browse...' to locate the ReaxFF files, "
                     "then clik 'Load Dataset' to automatically read the data and detect the engine. "
                     "If the engine is detected, it will show the engine name and number of frames at the bottom of the screen.\n\n"
                     " 2. If the engine was no automatically detected (because it couldn't find the characteristicc files for each engine, such "
                     "as xmolout for ReaxFF Standalone engine, you need to manually select the engine in 'Engine' in the pipeline browser.\n\n"
                     " 3. once the data is loaded and the engine is detected, you can add analyzers by clicking on 'Analysis' in the "
                     "pipeline browser, and then adding the appropriate analyzer.\n\n"
                     " 4. Once the analyzer is added, it shows a list of settings for that analyzer in the settings panel (bottom left of the screen). "
                     "Determine the settings, and start the analyzer. It takes some time to do the calculations and return the results.\n\n"
                     " 5. for each analyzer, a number of default presentations (usually a table view and a plot) are generated. However, you are not "
                     "limited to these presentations, and you can simply click on 'Presentation' to add more presentations.\n\n"
                     " 6. If you want, you can apply some utilities such as signal processing utilities to smooth the data, etc. by clicking "
                     "on 'Utilities' and selecting the approporiate utility. Once the utility is applied on your data, it generates a new column "
                     "to you data, which can be visualized later on by adding presentation layers.\n\n")


def run_main(args: argparse.Namespace) -> int:
    """Run the Web UI server."""
    from reaxkit.webui.app import main as webui_main

    webui_main()
    return 0
