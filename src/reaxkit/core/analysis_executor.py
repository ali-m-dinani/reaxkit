"""Module alias for the relocated analysis executor."""

import sys

from reaxkit.core.runtime import analysis_executor as _implementation

sys.modules[__name__] = _implementation
