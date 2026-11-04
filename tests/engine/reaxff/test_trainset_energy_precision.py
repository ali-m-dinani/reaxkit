import numpy as np
import pytest

from reaxkit.engine.reaxff.generators import trainset_elastic_energy
from reaxkit.engine.reaxff.generators.trainset_elastic_energy import (
    BulkEnergySpec, CellSpec, ElasticEnergySpec, TrainsetEnergyResult,
    _generate_bulk_data, _generate_elastic_data, _write_trainset_energy,
)
from reaxkit.engine.reaxff.generators.trainset_heatfo import (
    HeatFoReferenceSpec, _build_energy_line,
)


@pytest.mark.parametrize("energy", [0.0, -0.0, 1e-10, -1e-10, 1.2345678901234567])
def test_heatfo_targets_preserve_nonzero_energy(energy):
    line = _build_energy_line(
        identifier="sample", composition_counts={"Zn": 1}, element_order=["Zn"],
        total_atoms=1, per_atom_heat_kcal_mol=energy, weight=1.0,
        references_by_element={"Zn": HeatFoReferenceSpec("reference", 1)},
    )
    assert float(line.split()[-1]) == (0.0001 if energy == 0 else energy)
    if energy == 0:
        assert line.split()[-1] == "0.0001"


@pytest.mark.parametrize("energy", [0.0, -0.0, 1e-10, -1e-10, 1.2345678901234567])
def test_bulk_targets_preserve_nonzero_energy(monkeypatch, energy):
    monkeypatch.setattr(trainset_elastic_energy, "vinet_energy_trainset", lambda **kwargs: energy)
    cell = CellSpec(3, 4, 5, 90, 90, 90)
    rows, lines = _generate_bulk_data(BulkEnergySpec(100, 4, 1, cell))
    for (_, target), line in zip(rows, lines):
        assert target == (0.0001 if energy == 0 else energy)
        assert float(line.split()[-1]) == target


@pytest.mark.parametrize("tensor_mode", [False, True])
def test_elastic_targets_preserve_small_nonzero_energy(tensor_mode):
    cell = CellSpec(3, 4, 5, 90, 90, 90)
    constants = dict(c11=180, c22=190, c33=220, c12=40, c13=50, c23=60,
                     c44=70, c55=80, c66=90)
    spec = ElasticEnergySpec(
        constants, 1e-5, cell, strain_step=1e-7,
        tensor_gpa=(np.eye(6) * 100).tolist() if tensor_mode else None,
    )
    for rows, lines in _generate_elastic_data(spec).values():
        for (strain, energy), line in zip(rows, lines):
            assert float(line.split()[-1]) == energy
            if strain == 0:
                assert line.split()[-1] == "0.0001"
            else:
                assert 0 < energy < 0.00005


def test_elastic_table_writer_preserves_energy_precision(tmp_path):
    energies = [0.0001, 1e-10, -1e-10, 1.2345678901234567]
    rows = list(enumerate(energies))
    result = TrainsetEnergyResult("ENERGY\nENDENERGY\n", rows, {"c11": rows})
    paths = _write_trainset_energy(result, out_dir=tmp_path)
    for path in paths.values():
        if path.suffix == ".dat":
            targets = [float(line.split()[-1]) for line in path.read_text().splitlines()[1:]]
            assert targets == energies
