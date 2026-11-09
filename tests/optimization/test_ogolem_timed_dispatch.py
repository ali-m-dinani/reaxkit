import pytest

from reaxkit.optimization.adapters.ogolem.launch import after_prepare
from reaxkit.optimization.models import OptimizationError


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("existing", ["", "MaxTasksToSubmit=1000\n"])
def test_timed_dispatch_bounds_queued_tasks(tmp_path, workers, existing):
    config = tmp_path / "fit.ogo"
    config.write_text("<ADAPTIVE>\n" + existing + "PopulationSize=4\n</ADAPTIVE>\n")
    (tmp_path / "evaluator.json").write_text("{}")
    request = {
        "launch": {"results_directory": str(tmp_path)},
        "runtime": {"optimization_time_limit_seconds": 120},
        "settings": {"workers": workers, "timeout_seconds": 300, "backend": "ams"},
    }
    after_prepare(request)
    assert config.read_text() == f"<ADAPTIVE>\nMaxTasksToSubmit={workers}\nPopulationSize=4\n</ADAPTIVE>\n"
    after_prepare(request)
    assert config.read_text().count("MaxTasksToSubmit=") == 1


def test_timed_dispatch_rejects_unknown_layout(tmp_path):
    (tmp_path / "fit.ogo").write_text("unknown")
    with pytest.raises(OptimizationError, match="ADAPTIVE"):
        after_prepare({"launch": {"results_directory": str(tmp_path)},
                       "runtime": {"optimization_time_limit_seconds": 120},
                       "settings": {}})
