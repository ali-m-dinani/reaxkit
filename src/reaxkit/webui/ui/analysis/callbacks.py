"""Register analysis callback bindings for the Web UI.

This module is the stable callback entrypoint used by app startup. It wires
responsibility-scoped callback submodules while keeping the public API
(`register_analysis_callbacks`) unchanged.

**Usage context**

- Dash callback registration during app startup.
- Wiring analysis/service actions to UI components.
- Delegating callback blocks to focused registration submodules.
"""

from __future__ import annotations

from typing import Any

import reaxkit.webui.ui.analysis.callback_helpers as _helper
from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.ui.analysis.callback_helpers import (
    _parse_csv_ints,
    _parse_csv_strs,
    _selected_node,
)
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
from reaxkit.webui.ui.analysis.tasks import register_task_callbacks


def register_analysis_callbacks(app: Any, service: WebUIApiService) -> None:
    """Register analysis-focused Dash callbacks.

    Parameters
    -----
    app : Any
        Dash application instance used for callback decoration.
    service : WebUIApiService
        Backend service bridge for pipeline and analysis operations.

    Returns
    -----
    None
        Registers callbacks as a side effect on `app`.

    Examples
    -----
    ```python
    register_analysis_callbacks(app, service)
    ```
    Sample output:
    `None`
    Meaning:
    Callback handlers are attached to the Dash app instance.
    """
    _helper._SERVICE_HANDLE = service

    register_task_callbacks(
        app,
        service,
        selected_node=_selected_node,
        parse_csv_ints=_parse_csv_ints,
        parse_csv_strs=_parse_csv_strs,
    )
    register_pipeline_callbacks(app, service)
    register_execution_callbacks(app, service)
    register_properties_callbacks(app, service)
    register_visualization_callbacks(app, service)
    register_canvas_callbacks(app, service)


__all__ = ["register_analysis_callbacks"]
