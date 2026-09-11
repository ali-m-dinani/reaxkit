from __future__ import annotations

import numpy as np

from reaxkit.domain.data_models import SimulationData
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter


def _summary_simulation() -> SimulationData:
    return SimulationData(
        atom_ids=[],
        iterations=np.asarray([0, 10], dtype=int),
        potential_energy=np.asarray([-1.0, -2.0]),
        temperature=np.asarray([300.0, 301.0]),
    )


def test_load_simulation_skips_xmolout_when_summary_has_requested_fields(monkeypatch) -> None:
    adapter = ReaxFFAdapter()
    summary = _summary_simulation()
    monkeypatch.setattr(
        adapter,
        "_load_simulation_from_summary",
        lambda args, reporter=None: summary,
    )

    def fail_if_xmolout_is_loaded(args, reporter=None):
        raise AssertionError("xmolout should not be loaded for a summary-backed field")

    monkeypatch.setattr(adapter, "_load_simulation_from_xmolout", fail_if_xmolout_is_loaded)

    result = adapter.load_simulation(
        {"_required_data_fields": ("potential_energy", "temperature")}
    )

    assert result is summary


def test_load_simulation_uses_xmolout_only_to_enrich_missing_fields(monkeypatch) -> None:
    adapter = ReaxFFAdapter()
    summary = _summary_simulation()
    xmolout = SimulationData(
        atom_ids=[1],
        iterations=np.asarray([0, 10], dtype=int),
        potential_energy=np.asarray([-9.0, -9.0]),
        cell_lengths=np.asarray([[4.0, 5.0, 6.0], [4.1, 5.1, 6.1]]),
    )
    monkeypatch.setattr(
        adapter,
        "_load_simulation_from_summary",
        lambda args, reporter=None: summary,
    )
    monkeypatch.setattr(
        adapter,
        "_load_simulation_from_xmolout",
        lambda args, reporter=None: xmolout,
    )

    result = adapter.load_simulation({"_required_data_fields": ("a",)})

    assert result.atom_ids == [1]
    assert result.cell_lengths.tolist() == [[4.0, 5.0, 6.0], [4.1, 5.1, 6.1]]
    assert result.potential_energy.tolist() == [-1.0, -2.0]


def test_summary_backed_simulation_fields_only_require_summary_snapshot() -> None:
    adapter = ReaxFFAdapter()

    assert adapter.required_input_files(
        SimulationData,
        {"_required_data_fields": ("potential_energy",)},
    ) == ("summary.txt",)
    assert adapter.required_input_files(
        SimulationData,
        {"_required_data_fields": ("a",)},
    ) == ("summary.txt", "xmolout")
