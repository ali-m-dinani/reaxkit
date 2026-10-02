"""Render synthetic EOS and reaction-energy examples with each built-in style."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from reaxkit.presentation.plot import plot
from reaxkit.presentation.plot_styles import PLOT_STYLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("plot_style_examples"))
    args = parser.parse_args()
    volume = np.linspace(26.8, 30.2, 11)
    eos = {
        "plot_type": "single_plot",
        "series": [
            {"x": volume, "y": 0.55 * (volume - 28.5) ** 2,
             "label": "ReaxFF", "marker": "o", "color": "tab:blue"},
            {"x": volume, "y": 0.48 * (volume - 28.5) ** 2,
             "label": "QM/Literature", "marker": "o", "color": "#C0504D"},
        ],
        "title": "EOS bulk_0", "xlabel": "Volume (Å³)",
        "ylabel": "Relative energy (kcal/mol)", "legend": True, "figsize": (6, 5),
    }
    bars = {
        "plot_type": "grouped_bar_plot",
        "labels": ["C_clust", "CH_clust", "CC_clust", "CCH3_clust", "CCH_clust", "CH2_clust"],
        "series": [
            {"values": [-139, -136, -72, -138, -82, -94], "label": "ReaxFF", "color": "tab:blue"},
            {"values": [-147, -145, -140, -134, -99, -95], "label": "QM/Literature", "color": "#C0504D"},
        ],
        "title": "Reaction Energies", "ylabel": "Reaction energy (kcal/mol)",
        "legend": True, "grid": False, "group_width": 0.48,
    }
    for style in PLOT_STYLES:
        for name, payload in (("eos", eos), ("reaction", bars)):
            destination = args.output / f"{name}_{style}.png"
            plot({**payload, "save": destination}, style=style)
            print(destination)


if __name__ == "__main__":
    main()
