import json

import numpy as np
import pytest

from reaxkit.optimization.conversion import collect_native
from reaxkit.optimization.adapters.ezff.cobalt import OBJECTIVE, SCOPE
from .test_converters import snapshot, write


@pytest.fixture
def cobalt(tmp_path):
    native = tmp_path / "native"
    manifest = dict(engine="LAMMPS ReaxFF", objective=OBJECTIVE, dataset_scope=SCOPE, ezff_version="1.0.11")
    write(native, "cobalt-run.json", json.dumps(manifest))
    write(native, "variable_bounds", "parameter 0 3\n")
    write(native, "results/1/errors", "1\n")
    write(native, "results/1/variables", "2\n")
    write(native, "results/1/forcefields/FF_1", "synthetic recommended field\n")
    np.savez(native / "results/1/all_evaluated_001.npz", variables=[[1], [2]], errors=[[4], [1]])
    for index, predicted in enumerate((8, 2)):
        report = dict(manifest, candidate=index, loss=predicted / 2, geometry_relaxation=False,
                      boundary="nonperiodic", energies={"cluster": predicted, "atom": 0}, items=[
                          dict(line=3, first="cluster", second="atom", first_divisor=1,
                               second_divisor=1, tolerance=2, reference=0, prediction=predicted,
                               unit="kcal/mol", weighted_squared_error=(predicted / 2) ** 2)])
        write(native, f"evaluations/candidate-{index:04d}/report.json", json.dumps(report))
        write(native, f"evaluations/candidate-{index:04d}/parameters.json", json.dumps({"parameter": index + 1}))
    return native, tmp_path / "normalized"


def test_candidate_predictions_and_real_campaign_metadata(cobalt):
    native, output = cobalt
    metadata = dict(engine="User-supplied existing external engine/objective runner", objective="user_defined_vector",
                    dataset_scope="native_input_semantics", optimizer_version=None, comparison_status="verified",
                    comparison_fingerprint="old", run_id="run", replica=1)
    report = collect_native("ezff", native, output, metadata=metadata)
    assert report["progress_count"] == 3
    assert report["prediction_count"] == 2
    assert report["artifact_count"] == 2
    predictions = snapshot(output, "predictions.jsonl")
    assert [record["candidate_id"] for record in predictions] == ["evaluated:1", "evaluated:2"]
    assert predictions[0]["training_entry_id"] == predictions[1]["training_entry_id"]
    assert predictions[0]["weight_semantics"] == "accuracy_divisor"
    assert predictions[0]["weighted_squared_error"] == 16
    assert predictions[0]["unit"] == "kcal/mol"
    records = snapshot(output, "progress.jsonl")
    assert records[0]["record_kind"] == "recommendation"
    assert records[0]["evaluated_candidate_id"] == "evaluated:2"
    for record in records + predictions:
        assert record["engine_identity"] == "LAMMPS ReaxFF"
        assert record["objective_definition"] == OBJECTIVE
        assert record["dataset_scope"] == SCOPE
        assert record["optimizer_version"] == "1.0.11"
        assert record["comparison_status"] == "unverified"
        assert record["comparison_fingerprint"] != "old"
        assert record["run_id"] == "run"
    assert "cobalt-run.json" in report["sources"]
    assert "evaluations/candidate-0000/parameters.json" in report["sources"]
    assert "variable_bounds" in report["sources"]
    assert collect_native("ezff", native, output, metadata=metadata)["snapshot"] == report["snapshot"]
    assert metadata["engine"].startswith("User-supplied")


@pytest.mark.parametrize("field,value", [
    ("candidate", 9), ("candidate", False), ("loss", 99), ("items", []),
    ("items", [None]), ("geometry_relaxation", True), ("boundary", "periodic"),
    ("energies", {}), ("engine", "other"), ("objective", "RMSE"),
])
def test_inconsistent_reports_are_withheld(cobalt, field, value):
    native, output = cobalt
    path = native / "evaluations/candidate-0000/report.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    report = collect_native("ezff", native, output)
    assert report["prediction_count"] == 1
    assert report["progress_count"] == 3
    assert snapshot(output, "predictions.jsonl")[0]["candidate_id"] == "evaluated:2"
    assert any("withheld" in warning for warning in report["warnings"])


@pytest.mark.parametrize("field,value", [
    ("reference", "nan"), ("prediction", 100), ("weighted_squared_error", 0),
    ("tolerance", 0), ("first_divisor", 0), ("unit", "eV"), ("line", 0),
    ("prediction", 1e308),
])
def test_invalid_prediction_values(cobalt, field, value):
    native, output = cobalt
    path = native / "evaluations/candidate-0000/report.json"
    data = json.loads(path.read_text())
    data["items"][0][field] = value
    path.write_text(json.dumps(data))
    assert collect_native("ezff", native, output)["prediction_count"] == 1


def test_partial_json_and_changed_parameters(cobalt):
    native, output = cobalt
    write(native, "evaluations/candidate-0000/report.json", "{")
    write(native, "evaluations/candidate-0001/parameters.json", '{"parameter":1}')
    report = collect_native("ezff", native, output)
    assert report["prediction_count"] == 0
    assert report["progress_count"] == 3
    assert any("parameters do not match" in warning for warning in report["warnings"])


def test_unknown_profile_does_not_override_metadata(cobalt):
    native, output = cobalt
    write(native, "cobalt-run.json", '{"engine":"unknown"}')
    report = collect_native("ezff", native, output, metadata={"engine": "declared"})
    assert report["prediction_count"] == 0
    assert snapshot(output, "progress.jsonl")[0]["engine_identity"] == "declared"


def test_missing_archive_or_bounds_is_not_guessed(cobalt):
    native, output = cobalt
    bounds = native / "variable_bounds"
    bounds.unlink()
    assert collect_native("ezff", native, output)["prediction_count"] == 0
    write(native, "variable_bounds", "parameter 0 3\n")
    (native / "results/1/all_evaluated_001.npz").unlink()
    report = collect_native("ezff", native, output)
    assert report["progress_count"] == 1
    assert report["prediction_count"] == 0


def test_recommendation_does_not_link_on_score_alone(cobalt):
    native, output = cobalt
    write(native, "results/1/variables", "1\n")
    report = collect_native("ezff", native, output)
    assert report["prediction_count"] == 2
    assert "evaluated_candidate_id" not in snapshot(output, "progress.jsonl")[0]
