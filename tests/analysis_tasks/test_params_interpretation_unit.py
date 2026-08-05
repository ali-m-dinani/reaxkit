from __future__ import annotations

import numpy as np
import pandas as pd

from reaxkit.analysis.params.params import FFieldOptimizationParameterRequest, FFieldOptimizationParameterTask
from reaxkit.domain.data_models import ForceFieldOptimizationParameterBundleData, ForceFieldOptimizationParameterData, ForceFieldParametersData


def test_interpreted_params_support_general_atom_and_bond_sections() -> None:
    params = ForceFieldOptimizationParameterData(
        ff_section=np.array([1, 2, 3]), ff_section_line=np.array([1, 1, 1]),
        ff_parameter=np.array([1, 1, 1]), search_interval=np.array([0.1, 0.2, 0.3]),
        min_value=np.array([np.nan, 0.5, 1.0]), max_value=np.array([np.nan, 2.0, 200.0]),
        inline_comment=np.array(["global", "atom", "bond"], dtype=object),
    )
    force_field = ForceFieldParametersData(
        general_parameters=pd.DataFrame([{"name": "overcoord_1", "value": 50.0, "raw_comment": ""}], index=[1]),
        atom_parameters=pd.DataFrame([{"symbol": "C", "cov.r": 1.4}], index=[1]),
        bond_parameters=pd.DataFrame([{"i": 1, "j": 1, "Edis1": 140.0}], index=[1]),
        off_diagonal_parameters=pd.DataFrame(), angle_parameters=pd.DataFrame(),
        torsion_parameters=pd.DataFrame(), hydrogen_bond_parameters=pd.DataFrame(),
    )
    result = FFieldOptimizationParameterTask().run(
        ForceFieldOptimizationParameterBundleData(optimization_parameters=params, force_field_parameters=force_field),
        FFieldOptimizationParameterRequest(interpret=True, drop_duplicate=False),
    )

    assert result.table["component"].tolist() == ["overcoord_1", "cov.r", "Edis1"]
    assert result.table["ffield_value"].tolist() == [50.0, 1.4, 140.0]
    assert pd.isna(result.table.iloc[0]["min_value"])
