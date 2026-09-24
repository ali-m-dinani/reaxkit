from __future__ import annotations

import pytest

from reaxkit.engine.reaxff.adapter_parts.loaders_properties import load_electrostatics


class _ElectrostaticsAdapterStub:
    def __init__(self) -> None:
        self.connectivity_loads = 0
        self.charge_args = None

    @staticmethod
    def load_trajectory(args, reporter=None):
        return "trajectory"

    def load_charges(self, args, reporter=None):
        self.charge_args = dict(args)
        return "charges"

    def load_connectivity(self, args, reporter=None):
        self.connectivity_loads += 1
        return "connectivity"

    @staticmethod
    def _resolve_reaxff_path(*args, **kwargs):
        pytest.fail("fort.78 should not be inspected when electric_field is not required")


@pytest.mark.parametrize(
    ("required_fields", "expected_connectivity", "expected_loads", "expected_quick"),
    [
        (("trajectory", "charges"), None, 0, True),
        (("trajectory", "charges", "connectivity"), "connectivity", 1, False),
    ],
)
def test_electrostatics_loader_honors_required_fields(
    required_fields: tuple[str, ...],
    expected_connectivity: str | None,
    expected_loads: int,
    expected_quick: bool,
) -> None:
    adapter = _ElectrostaticsAdapterStub()

    result = load_electrostatics(
        adapter,
        {"_required_data_fields": required_fields},
    )

    assert result.trajectory == "trajectory"
    assert result.charges == "charges"
    assert result.connectivity == expected_connectivity
    assert result.electric_field is None
    assert adapter.connectivity_loads == expected_loads
    assert bool(adapter.charge_args.get("_quick_charge_only")) is expected_quick


def test_selected_electrostatics_frames_use_one_progress_bar_per_file() -> None:
    class ReportingAdapter(_ElectrostaticsAdapterStub):
        @staticmethod
        def load_trajectory(args, reporter=None):
            reporter("load", 0, 3, "Loading selected xmolout frames")
            reporter("load", 3, 3, "Loaded selected xmolout frames")
            reporter("load", 100, 100, "Scanning summary.txt")
            return "trajectory"

        def load_charges(self, args, reporter=None):
            self.charge_args = dict(args)
            reporter("load", 0, 3, "Loading selected fort.7 frames")
            reporter("load", 3, 3, "Loaded selected fort.7 frames")
            return "charges"

    events: list[tuple[str, int, int, str | None]] = []
    result = load_electrostatics(
        ReportingAdapter(),
        {
            "_frame_indices": [0, 2, 4],
            "_required_data_fields": ("trajectory", "charges"),
        },
        reporter=lambda stage, current, total, message=None: events.append(
            (stage, current, total, message)
        ),
    )

    assert result.trajectory == "trajectory"
    assert result.charges == "charges"
    assert events[0] == ("load", 1, 1, "Preparing input files")
    file_events = events[1:]
    assert {stage for stage, *_ in file_events} == {"load xmolout", "load fort.7"}
    assert all(total == 3 for _, _, total, _ in file_events)
    assert file_events[-1] == ("load fort.7", 3, 3, "Reading fort.7 frames")
    assert all("summary" not in str(message).lower() for *_, message in events)
    stages = [stage for stage, *_ in file_events]
    assert stages == ["load xmolout", "load xmolout", "load fort.7", "load fort.7"]
