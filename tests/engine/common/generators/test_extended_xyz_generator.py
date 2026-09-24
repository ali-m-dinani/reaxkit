from __future__ import annotations

import numpy as np

from reaxkit.engine.common.generators.extended_xyz_generator import (
    ExtendedXYZFrame,
    write_extended_xyz_trajectory,
)


def test_extended_xyz_generator_streams_typed_atom_properties(tmp_path) -> None:
    def frames():
        yield ExtendedXYZFrame(
            species=["Al", "N"],
            positions=np.asarray([[0.0, 0.0, 1.2], [1.1, 0.0, 1.8]]),
            properties={
                "atom_number": np.asarray([1, 2]),
                "charge": np.asarray([0.035, -0.021]),
                "delta_charge": np.asarray([0.0, 0.0]),
            },
            frame=0,
            iteration=10,
            lattice=np.diag([5.0, 6.0, 7.0]),
            pbc=(True, True, True),
        )

    destination = write_extended_xyz_trajectory(
        frames(),
        tmp_path / "trajectory.extxyz",
        precision=8,
    )
    lines = destination.read_text(encoding="utf-8").splitlines()

    assert lines[0] == "2"
    assert (
            "Properties=species:S:1:pos:R:3:atom_number:I:1:charge:R:1:delta_charge:R:1"
            in lines[1]
    )
    assert 'Lattice="5 0 0 0 6 0 0 0 7"' in lines[1]
    assert 'pbc="T T T"' in lines[1]
    assert "frame=0 iter=10" in lines[1]
    assert lines[2].split() == ["Al", "0", "0", "1.2", "1", "0.035", "0"]
    assert lines[3].split() == ["N", "1.1", "0", "1.8", "2", "-0.021", "0"]
