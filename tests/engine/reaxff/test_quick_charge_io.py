from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reaxkit.core.platform.exceptions import ParseError
from reaxkit.domain.data_models import ChargeData, ElectrostaticsData
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter
from reaxkit.engine.reaxff.adapter_parts import streaming
from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler
from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler
from reaxkit.engine.reaxff.quick_io import (
    iter_charge_data_quick,
    iter_fort7_charge_frames,
    iter_xmolout_atom_identities,
    load_charge_data_quick,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "reaxff_isomer_representatives_detection"
)


def test_xmolout_identity_reader_reads_every_frame_without_coordinates() -> None:
    records = list(iter_xmolout_atom_identities(FIXTURE_DIR / "xmolout"))

    assert len(records) > 1
    assert records[0]["atom_ids"] == list(range(1, 30))
    assert records[0]["elements"][:6] == ["B", "H", "H", "H", "H", "C"]
    assert all("coordinates" not in record and "frame" not in record for record in records)


def test_xmolout_identity_reader_tracks_atoms_added_in_later_frames(tmp_path) -> None:
    xmolout = tmp_path / "xmolout"
    xmolout.write_text(
        """2
sim 0 0 10 10 10 90 90 90
H 0 0 0
O 1 0 0
3
sim 10 0 10 10 10 90 90 90
H 0 0 0
O 1 0 0
C 2 0 0
""",
        encoding="utf-8",
    )

    records = list(iter_xmolout_atom_identities(xmolout))

    assert records[0]["atom_ids"] == [1, 2]
    assert records[0]["elements"] == ["H", "O"]
    assert records[1]["atom_ids"] == [1, 2, 3]
    assert records[1]["elements"] == ["H", "O", "C"]


def test_charge_reader_aligns_each_frames_identity_when_atoms_are_added(tmp_path) -> None:
    xmolout = tmp_path / "xmolout"
    xmolout.write_text(
        """2
sim 0 0 10 10 10 90 90 90
H 0 0 0
O 1 0 0
3
sim 10 0 10 10 10 90 90 90
H 0 0 0
O 1 0 0
C 2 0 0
""",
        encoding="utf-8",
    )
    fort7 = tmp_path / "fort.7"
    fort7.write_text(
        """2 sim Iteration: 0 #Bonds: 0
1 1 1 0.0 0.0 0.1
2 2 1 0.0 0.0 -0.1
0.0 0.0 0.0 0.0
3 sim Iteration: 10 #Bonds: 0
1 1 1 0.0 0.0 0.2
2 2 1 0.0 0.0 -0.2
3 3 1 0.0 0.0 0.3
0.0 0.0 0.0 0.0
""",
        encoding="utf-8",
    )

    frames = list(iter_charge_data_quick(fort7, xmolout_path=xmolout))

    assert frames[0].simulation.elements == ["H", "O"]
    assert frames[1].simulation.elements == ["H", "O", "C"]
    np.testing.assert_allclose(frames[1].charges, [[0.2, -0.2, 0.3]])


def test_quick_fort7_reader_returns_charge_and_type_arrays_without_tables() -> None:
    record = next(iter_fort7_charge_frames(FIXTURE_DIR / "fort.7", frame_indices=[0]))

    assert "frame" not in record
    assert record["charge_atom_ids"].shape == (29,)
    assert record["charge_atom_type_nums"].shape == (29,)
    np.testing.assert_allclose(record["charges"][:2], [0.065, 0.094])
    np.testing.assert_array_equal(record["charge_atom_type_nums"][:2], [4, 2])


def test_quick_materialized_loader_preserves_elements_and_iterations() -> None:
    data = load_charge_data_quick(
        FIXTURE_DIR / "fort.7",
        xmolout_path=FIXTURE_DIR / "xmolout",
        frame_indices=[0, 2],
    )

    assert data.charges.shape == (2, 29)
    assert data.simulation.elements[:2] == ["B", "H"]
    assert data.iterations.tolist() == [0, 200]


def test_charge_stream_never_uses_full_xmolout_frame_parser(monkeypatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Charge-only streaming must not parse xmolout coordinates.")

    monkeypatch.setattr(XmoloutHandler, "stream_file_frames", fail_if_called)
    frames = list(
        ReaxFFAdapter().stream(
            ChargeData,
            {
                "fort7": str(FIXTURE_DIR / "fort.7"),
                "xmolout": str(FIXTURE_DIR / "xmolout"),
                "_frame_indices": [0, 1],
                "progress": False,
            },
        )
    )

    assert len(frames) == 2
    assert frames[0].simulation.elements[:2] == ["B", "H"]
    assert frames[0].metadata["charges_only"] is True


def test_total_electrostatics_always_uses_public_charge_only_quick_io(monkeypatch) -> None:
    original = streaming.iter_fort7_charge_frames
    calls = []

    def tracking_reader(*args, **kwargs):
        calls.append((args, kwargs))
        yield from original(*args, **kwargs)

    monkeypatch.setattr(streaming, "iter_fort7_charge_frames", tracking_reader)
    frames = list(
        ReaxFFAdapter().stream(
            ElectrostaticsData,
            {
                "fort7": str(FIXTURE_DIR / "fort.7"),
                "xmolout": str(FIXTURE_DIR / "xmolout"),
                "scope": "total",
                "_frame_indices": [0, 1],
                "progress": False,
            },
        )
    )

    assert len(calls) == 1
    assert calls[0][1]["include_atom_types"] is False
    assert len(frames) == 2
    assert frames[0].charges.metadata["charges_only"] is True


def test_materialized_total_electrostatics_uses_quick_charge_io(monkeypatch) -> None:
    def fail_full_fort7_parse(*_args, **_kwargs):
        raise AssertionError("Charge-only electrostatics must not fully parse fort.7.")

    monkeypatch.setattr(Fort7Handler, "_parse", fail_full_fort7_parse)
    data = ReaxFFAdapter().load(
        ElectrostaticsData,
        {
            "fort7": str(FIXTURE_DIR / "fort.7"),
            "xmolout": str(FIXTURE_DIR / "xmolout"),
            "_required_data_fields": ("trajectory", "charges"),
            "progress": False,
        },
    )

    assert data.connectivity is None
    assert data.charges.metadata["charges_only"] is True
    assert data.charges.charges.shape[0] > 1


def test_total_electrostatics_quick_reader_skips_fused_atom_type_field(tmp_path) -> None:
    fort7 = tmp_path / "fort.7"
    fort7.write_text(
        """    28880 slab Iteration: 0 #Bonds: 10
10002    2100071001210087115211152311529    0    0    0    0    1  0.426  0.551  0.517  0.342  0.547  0.321  0.000  0.000  0.000  0.000  3.988  0.000  1.177
 0.0 0.0 0.0 0.0
""",
        encoding="utf-8",
    )

    record = next(
        iter_fort7_charge_frames(fort7, include_atom_types=False)
    )

    assert "charge_atom_type_nums" not in record
    np.testing.assert_array_equal(record["charge_atom_ids"], [10002])
    np.testing.assert_allclose(record["charges"], [1.177])


def test_quick_xmolout_reader_rejects_malformed_geometry_name(tmp_path) -> None:
    xmolout = tmp_path / "xmolout"
    xmolout.write_text(
        """1
Lattice="102.95859867431082 0 -4428142.44 104.96 60.60 78.26 90.00 90.00 90.00
Al 0.95263 2.39140 14.93730
""",
        encoding="utf-8",
    )

    with pytest.raises(ParseError, match="Malformed xmolout frame header.*Lattice="):
        list(iter_xmolout_atom_identities(xmolout))


def test_quick_fort7_reader_rejects_malformed_geometry_name(tmp_path) -> None:
    fort7 = tmp_path / "fort.7"
    fort7.write_text(
        """28880 Lattice="102.95859867431082 Iteration: 0 #Bonds: 10
""",
        encoding="utf-8",
    )

    with pytest.raises(ParseError) as error:
        list(iter_fort7_charge_frames(fort7))

    message = str(error.value)
    assert "Malformed fort.7 frame header" in message
    assert "frame 0" in message
    assert "Lattice=" in message
    assert "short geometry name" in message


def test_full_fort7_handler_rejects_malformed_geometry_name(tmp_path) -> None:
    fort7 = tmp_path / "fort.7"
    fort7.write_text(
        """28880 Lattice="102.95859867431082 Iteration: 0 #Bonds: 10
""",
        encoding="utf-8",
    )

    with pytest.raises(ParseError) as error:
        Fort7Handler(fort7).dataframe()

    message = str(error.value)
    assert "Malformed fort.7 frame header" in message
    assert "frame 0, line 1" in message
    assert "Lattice=" in message
