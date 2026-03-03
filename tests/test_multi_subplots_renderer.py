from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from reaxkit.presentation.plot.renderers.multi_subplots import MultiSubplotsRenderer


def test_multi_subplots_renderer_accepts_grid_and_hides_unused_axes():
    renderer = MultiSubplotsRenderer()
    original_show = plt.show
    plt.show = lambda: None
    try:
        fig = renderer.render(
            {
                "plot_type": "multi_subplots",
                "subplots": [
                    [{"x": [0, 1], "y": [0, 1], "label": "a"}],
                    [{"x": [0, 1], "y": [1, 2], "label": "b"}],
                    [{"x": [0, 1], "y": [2, 3], "label": "c"}],
                ],
                "grid": "2x2",
                "legend": False,
                "save": None,
            }
        )
    finally:
        plt.show = original_show

    assert len(fig.axes) == 4


def test_multi_subplots_renderer_paginates_when_grid_is_too_small():
    renderer = MultiSubplotsRenderer()
    original_show = plt.show
    plt.show = lambda: None
    try:
        figures = renderer.render(
            {
                "plot_type": "multi_subplots",
                "subplots": [
                    [{"x": [0, 1], "y": [0, 1], "label": "a"}],
                    [{"x": [0, 1], "y": [1, 2], "label": "b"}],
                    [{"x": [0, 1], "y": [2, 3], "label": "c"}],
                    [{"x": [0, 1], "y": [3, 4], "label": "d"}],
                    [{"x": [0, 1], "y": [4, 5], "label": "e"}],
                    [{"x": [0, 1], "y": [5, 6], "label": "f"}],
                    [{"x": [0, 1], "y": [6, 7], "label": "g"}],
                ],
                "grid": "3x2",
                "legend": False,
                "save": None,
                "title": "Paged",
            }
        )
    finally:
        plt.show = original_show

    assert isinstance(figures, list)
    assert len(figures) == 2
    assert len(figures[0].axes) == 6
    assert len(figures[1].axes) == 6
