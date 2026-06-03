"""Callback registration submodules for analysis UI."""

from reaxkit.webui.ui.analysis.callback_sections.canvas_callbacks import (
    register_canvas_callbacks,
)
from reaxkit.webui.ui.analysis.callback_sections.execution_callbacks import (
    register_execution_callbacks,
)
from reaxkit.webui.ui.analysis.callback_sections.pipeline_callbacks import (
    register_pipeline_callbacks,
)
from reaxkit.webui.ui.analysis.callback_sections.properties_callbacks import (
    register_properties_callbacks,
)
from reaxkit.webui.ui.analysis.callback_sections.visualization_callbacks import (
    register_visualization_callbacks,
)

__all__ = [
    "register_pipeline_callbacks",
    "register_execution_callbacks",
    "register_properties_callbacks",
    "register_visualization_callbacks",
    "register_canvas_callbacks",
]
