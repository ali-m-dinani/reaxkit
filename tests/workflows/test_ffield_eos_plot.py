from __future__ import annotations

import argparse
from types import SimpleNamespace

import pandas as pd
import pytest

from reaxkit.workflows.file_tools.ffield_workflow import (
    _eos_identifier_coordinate,
    _eos_material_name,
    _plot_payload,
    _validate_eos_save_target,
)


def test_eos_identifier_coordinate_supports_decimal_and_index_suffixes() -> None:
    assert _eos_identifier_coordinate("H2BCH3", "H2BCH3_0_652") == 0.652
    assert _eos_identifier_coordinate("H2BNH2.opt", "H2BNH2_12") == 12.0
    assert _eos_identifier_coordinate("bulk", "bulk_1.05") == 1.05
    assert _eos_identifier_coordinate("bulk", "bulk_diss") is None


def test_eos_material_name_groups_scan_families_by_material() -> None:
    assert _eos_material_name("bulk_0_mp_1008557") == "mp_1008557"
    assert _eos_material_name("c11_0_mp_1008557") == "mp_1008557"
    assert _eos_material_name("c66_0_AlBN_tetragonal") == "AlBN_tetragonal"
    assert _eos_material_name("cBN_opt") == "cBN_opt"
    assert _eos_material_name("unstructured_identifier") == "unstructured_identifier"


def test_eos_single_plot_builds_one_populated_payload_per_identifier() -> None:
    result = SimpleNamespace(
        table=pd.DataFrame(
            {
                "base_iden": ["bulk", "bulk", "molecule.opt", "molecule.opt"],
                "other_iden": ["bulk_0.9", "bulk_1.0", "molecule_1", "molecule_2"],
                "V_other_iden": [9.0, 10.0, float("nan"), float("nan")],
                "E_other_iden": [-4.0, -5.0, 2.0, 1.0],
                "ffield_value": [-3.8, -4.9, 2.2, 1.1],
                "qm_value": [-4.0, -5.0, 2.0, 1.0],
            }
        )
    )

    payloads = _plot_payload(
        "get_ffield_opt_eos",
        result,
        argparse.Namespace(plot="single", grid=None),
    )

    assert isinstance(payloads, list)
    assert len(payloads) == 2
    assert [series["label"] for series in payloads[0]["series"]] == [
        "ReaxFF",
        "QM/Literature",
    ]
    assert payloads[0]["series"][0]["x"] == [9.0, 10.0]
    assert payloads[0]["series"][0]["y"] == [-3.8, -4.9]
    assert payloads[0]["series"][1]["x"] == [9.0, 10.0]
    assert payloads[0]["series"][1]["y"] == [-4.0, -5.0]
    assert payloads[0]["series"][0]["color"] == "tab:blue"
    assert payloads[0]["series"][1]["color"] == "#C0504D"
    assert payloads[0]["xlabel"] == "Volume (Å³)"
    assert payloads[0]["figsize"] == (6.0, 5.0)
    assert payloads[1]["series"][0]["x"] == [1.0, 2.0]
    assert payloads[1]["series"][0]["y"] == [2.2, 1.1]
    assert payloads[1]["series"][1]["x"] == [1.0, 2.0]
    assert payloads[1]["series"][1]["y"] == [2.0, 1.0]
    assert payloads[1]["xlabel"] == "Scan coordinate"
    assert payloads[1]["filename"] == "eos_molecule.opt.png"
    assert payloads[1]["subdirectory"] == "molecule.opt"


def test_eos_elastic_families_plot_strain_percent_instead_of_volume() -> None:
    result = SimpleNamespace(
        table=pd.DataFrame(
            {
                "base_iden": ["c12_0_mp_2604"] * 3 + ["c66_0_mp_2604"] * 3,
                "other_iden": [
                    "c12_c1_mp_2604",
                    "c12_0_mp_2604",
                    "c12_e1_mp_2604",
                    "c66_c1_mp_2604",
                    "c66_0_mp_2604",
                    "c66_e1_mp_2604",
                ],
                "V_other_iden": [350.1, 350.0, 350.1, 356.0, 355.9, 356.0],
                "E_other_iden": [1.0, 0.0, 1.0, 2.0, 0.0, 2.0],
                "strain_percent": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
                "ffield_value": [1.1, 0.0, 1.1, 2.1, 0.0, 2.1],
                "qm_value": [1.0, 0.0, 1.0, 2.0, 0.0, 2.0],
            }
        )
    )

    payloads = _plot_payload(
        "get_ffield_opt_eos",
        result,
        argparse.Namespace(plot="single", grid=None),
    )

    assert payloads[0]["series"][0]["x"] == [-1.0, 0.0, 1.0]
    assert payloads[0]["xlabel"] == "Orthorhombic strain δ (%)"
    assert payloads[1]["series"][0]["x"] == [-1.0, 0.0, 1.0]
    assert payloads[1]["xlabel"] == "Shear angle change (%)"


def test_eos_single_plot_requires_save_directory() -> None:
    with pytest.raises(ValueError, match="--save must be a directory"):
        _validate_eos_save_target(
            argparse.Namespace(plot="single", save="eos.png")
        )

    _validate_eos_save_target(
        argparse.Namespace(plot="single", save="eos_plots")
    )
