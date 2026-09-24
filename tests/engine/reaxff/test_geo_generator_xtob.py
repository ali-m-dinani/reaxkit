"""Checks for XYZ to GEO conversion."""

import pytest

from reaxkit.engine.reaxff.generators.geo_generator import xtob


@pytest.mark.parametrize(
    "comment",
    [
        'Lattice="1 0 0 0 1 0 0 0 1" Properties=species:S:1:pos:R:3',
        "A custom XYZ comment",
    ],
)
def test_xtob_uses_input_filename_for_descrp(tmp_path, comment):
    xyz = tmp_path / "slab.xyz"
    xyz.write_text(f"2\n{comment}\nAl 0 0 0\nN 0 0 1\n", encoding="utf-8")

    geo = tmp_path / "geo"
    xtob(xyz, geo, box_lengths=(1, 1, 2))

    assert geo.read_text(encoding="utf-8").splitlines()[1] == "DESCRP  slab"
