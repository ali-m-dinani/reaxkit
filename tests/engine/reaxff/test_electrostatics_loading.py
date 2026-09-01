from __future__ import annotations

import pytest

from reaxkit.engine.reaxff.adapter_parts.loaders_properties import load_electrostatics


class _ElectrostaticsAdapterStub:
    def __init__(self) -> None:
        self.connectivity_loads = 0

    @staticmethod
    def load_trajectory(args, reporter=None):
        return "trajectory"

    @staticmethod
    def load_charges(args, reporter=None):
        return "charges"

    def load_connectivity(self, args, reporter=None):
        self.connectivity_loads += 1
        return "connectivity"

    @staticmethod
    def _resolve_reaxff_path(*args, **kwargs):
        pytest.fail("fort.78 should not be inspected when electric_field is not required")


@pytest.mark.parametrize(
    ("required_fields", "expected_connectivity", "expected_loads"),
    [
        (("trajectory", "charges"), None, 0),
        (("trajectory", "charges", "connectivity"), "connectivity", 1),
    ],
)
def test_electrostatics_loader_honors_required_fields(
    required_fields: tuple[str, ...],
    expected_connectivity: str | None,
    expected_loads: int,
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
