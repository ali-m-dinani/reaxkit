import json
from pathlib import Path

import pytest

from reaxkit.optimization.conversion import collect_native
from reaxkit.optimization.storage import read_json


def snapshot(output, filename):
    report = read_json(output / "conversion.json")
    path = output / report["snapshot"] / filename
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines()]
    return read_json(path)


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_indeedopt_candidate_predictions(tmp_path):
    native, output = tmp_path / "native", tmp_path / "normalized"
    write(native, "short_results/results.csv", "sample,loss\n0,4\n1,1\n")
    for sample, predicted in enumerate((4, 2)):
        report = {"loss": (predicted / 2) ** 2,
                  "items": [{"prediction": predicted, "target": 0, "weight": 2}]}
        write(native, f"ffield-{sample}/jax_md_results/evaluation.json", json.dumps(report))
    result = collect_native("indeedopt", native, output)
    predictions = snapshot(output, "predictions.jsonl")
    assert result["prediction_count"] == 2
    assert [row["candidate_id"] for row in predictions] == ["sample:0", "sample:1"]
    assert [row["weighted_squared_error"] for row in predictions] == [4, 1]
    assert all(row["identity_scope"] == "source" and row["unit"] is None for row in predictions)
    assert all(row["engine_identity"] == "JAX-ReaxFF (jax-md)" for row in snapshot(output, "progress.jsonl"))
    assert result["coverage"] == "partial"
    assert collect_native("indeedopt", native, output)["snapshot"] == result["snapshot"]


@pytest.mark.parametrize("report", [
    "{", "[]", '{"loss":4,"items":[]}',
    '{"loss":4,"items":[{"prediction":4,"target":0,"weight":0}]}',
    '{"loss":4,"items":[{"prediction":"nan","target":0,"weight":2}]}',
    '{"loss":5,"items":[{"prediction":4,"target":0,"weight":2}]}',
    '{"loss":4,"items":[{"prediction":2,"target":0,"weight":2}]}',
])
def test_indeedopt_invalid_report_preserves_progress(tmp_path, report):
    native, output = tmp_path / "native", tmp_path / "normalized"
    write(native, "short_results/results.csv", "sample,loss\n0,4\n")
    write(native, "ffield-0/jax_md_results/evaluation.json", report)
    result = collect_native("indeedopt", native, output)
    assert result["progress_count"] == 1
    assert result["prediction_count"] == 0
    assert result["warnings"]


@pytest.mark.parametrize("optimizer,filename,content,options,expected", [
    ("ams_mcff", "MCFFOptimizer.log", "iteration error seconds\n1 9.5 2.0\n2 8.0 4.0\n",
     {"skip_rows": 1, "columns": {"step": 0, "score": 1, "elapsed_seconds": 2}}, 9.5),
    ("ams_cmaes", "cmaes_progress.txt", "1 16\n2 9\n", {"columns": {"step": 0, "score": 1}}, 16),
    ("standalone_soppi", "fort.83", "Error force field:  12.5000\nsynthetic field\n", {}, 12.5),
    ("jax_soppi", "history.jsonl", '{"step_index":0,"current_error":7.2,"best_error":7.2,"attempts":[]}\n', {}, 7.2),
    ("jax_reaxff", "stdout.log", "Trial-1 is starting...\nIteration: 0\nTrue total loss: 8.25\n", {}, 8.25),
    ("indeedopt", "short_results/results.csv", "sample,loss\n0,4.25\n", {}, 4.25),
    ("ogolem", "evaluations/candidate_example/fitness.json", '{"fitness":6.25}', {}, 6.25),
    ("pibo", "history.csv", "evaluation,loss,error\n1,2.25,\n", {}, 2.25),
    ("ezff", "results/1/errors", "3.25\n", {}, 3.25),
])
def test_each_supported_adapter_preserves_source_values(tmp_path, optimizer, filename, content, options, expected):
    native, output = tmp_path / "native", tmp_path / "normalized"
    write(native, filename, content)
    report = collect_native(optimizer, native, output, options)
    values = snapshot(output, "progress.jsonl")
    assert values[0]["score"] == expected
    assert values[0]["engine_calls"] is None
    assert report["progress_count"] == len(values)
    repeat = collect_native(optimizer, native, output, options)
    assert repeat["snapshot"] == report["snapshot"]
    assert len(snapshot(output, "progress.jsonl")) == len(values)


def test_partial_lines_rewrite_and_snapshot_retention(tmp_path):
    native, output = tmp_path / "native", tmp_path / "normalized"
    path = write(native, "history.jsonl", '{"step_index":0,"current_error":8,"best_error":8}\n{"step_index":')
    first = collect_native("jax_soppi", native, output)
    assert first["progress_count"] == 1
    assert any("partial" in warning for warning in first["warnings"])
    path.write_text('{"step_index":0,"current_error":3,"best_error":3}\n')
    second = collect_native("jax_soppi", native, output)
    assert second["snapshot"] != first["snapshot"]
    assert (output / first["snapshot"] / "progress.jsonl").exists()
    assert snapshot(output, "source_changes.json")[0]["change"] == "truncated"
    assert snapshot(output, "progress.jsonl")[0]["score"] == 3


def test_ams_never_guesses_columns_and_parity(tmp_path):
    native, output = tmp_path / "native", tmp_path / "normalized"
    write(native, "MCFFOptimizer.log", "1 999 3\n")
    result = collect_native("ams_mcff", native, output)
    assert result["progress_count"] == 0
    assert any("columns" in warning for warning in result["warnings"])


def test_jax_report_identity_weight_units_and_candidate(tmp_path):
    native, output = tmp_path / "native", tmp_path / "normalized"
    write(native, "report_1_16.txt", "Item Text Weight Target Prediction Weighted Error Cum. Sum.\nENERGY: + molecule/1   2.00  1.00  9.00  16.00  16.00\n")
    collect_native("jax_reaxff", native, output)
    row = snapshot(output, "predictions.jsonl")[0]
    assert row["reference"] == 1 and row["prediction"] == 9
    assert row["weight"] == 2 and row["weight_semantics"] == "accuracy_divisor"
    assert row["weighted_squared_error"] == ((9 - 1) / 2) ** 2
    assert row["unit"] == "kcal/mol"
    assert row["candidate_id"] == "report_1_16.txt"
    assert row["entry_label"].startswith("ENERGY: + molecule/1")


def test_vector_objectives_and_nonfinite_values_are_not_scalarized(tmp_path):
    native, output = tmp_path / "native", tmp_path / "normalized"
    write(native, "results/1/errors", "1.5 2.5\nnan 3.0\n")
    collect_native("ezff", native, output)
    rows = snapshot(output, "progress.jsonl")
    assert rows[0]["score"] is None
    assert rows[0]["objectives"] == {"objective_1": 1.5, "objective_2": 2.5}
    assert rows[1]["status"] == "nonfinite"
    assert rows[1]["objectives"]["objective_1"] is None


def test_pibo_penalty_and_posthoc_metrics_are_distinct(tmp_path):
    native, output = tmp_path / "native", tmp_path / "normalized"
    write(native, "history.csv", "evaluation,loss,error\n1,1000000,engine failed\n")
    write(native, "benchmark_per_run.csv", "Loss(RMSE),Loss(MAE)\n9,4\n")
    report = collect_native("pibo", native, output)
    rows = snapshot(output, "progress.jsonl")
    assert len(rows) == 1 and rows[0]["status"] == "penalty"
    assert rows[0]["objective"] == "cobalt_subset_normalized_RMS"
    assert any("offset" in warning for warning in report["warnings"])


def test_completed_ogolem_archives_only_and_npz_snapshots(tmp_path):
    import numpy as np
    native, output = tmp_path / "native", tmp_path / "normalized"
    write(native, "evaluations/.pending_example/fitness.json", '{"fitness":1}')
    report = collect_native("ogolem", native, output)
    assert report["progress_count"] == 0
    np.savez(native / "bruteforce_best.npz", parameters=np.array([1, 2]), error=12.5)
    collect_native("indeedopt", native, output)
    assert snapshot(output, "progress.jsonl")[0]["score"] == 12.5


def test_writer_lock_prevents_collection_races(tmp_path):
    from reaxkit.optimization.models import OptimizationError
    from reaxkit.optimization.storage import lock
    native, output = tmp_path / "native", tmp_path / "normalized"
    native.mkdir()
    with lock(output, ".conversion.lock"):
        with pytest.raises(OptimizationError, match="Writer lock"):
            collect_native("jax_soppi", native, output)


def test_restarted_segments_have_distinct_candidate_identities(tmp_path):
    native, output = tmp_path / "native", tmp_path / "normalized"
    write(native, "history.jsonl",
          '{"step_index":0,"current_error":8,"best_error":8}\n'
          '{"step_index":0,"current_error":7,"best_error":7}\n')
    collect_native("jax_soppi", native, output)
    rows = snapshot(output, "progress.jsonl")
    assert rows[0]["candidate_id"] != rows[1]["candidate_id"]
    assert [row["segment"] for row in rows] == [0, 1]


def test_appended_source_is_detected_without_duplicate_rows(tmp_path):
    native, output = tmp_path / "native", tmp_path / "normalized"
    path = write(native, "history.csv", "evaluation,loss,error\n1,2,\n")
    collect_native("pibo", native, output)
    with path.open("a") as stream:
        stream.write("2,1,\n")
    collect_native("pibo", native, output)
    assert len(snapshot(output, "progress.jsonl")) == 2
    assert snapshot(output, "source_changes.json")[0]["change"] == "appended"


def test_latest_ezff_archive_not_cumulative_duplicates(tmp_path):
    import numpy as np
    native, output = tmp_path / "native", tmp_path / "normalized"
    write(native, "results/1/errors", "1 2\n")
    write(native, "results/2/errors", "0.5 1\n")
    np.savez(native / "results/1/all_evaluated_261105_120000.npz",
             variables=np.array([[3]]), errors=np.array([[1, 2]]))
    np.savez(native / "results/2/all_evaluated_261105_120100.npz",
             variables=np.array([[3], [4]]), errors=np.array([[1, 2], [0.5, 1]]))
    collect_native("ezff", native, output)
    rows = snapshot(output, "progress.jsonl")
    evaluated = [row for row in rows if row["record_kind"] == "evaluation"]
    assert len(evaluated) == 2
    assert evaluated[1]["objectives"]["objective_2"] == 1
    assert evaluated[1]["score"] is None


def test_nested_force_residuals_preserve_components(tmp_path):
    native, output = tmp_path / "native", tmp_path / "normalized"
    record = {"step_index": 0, "current_error": 3, "best_error": 3,
              "attempts": [{"candidate_residuals": {"FORCE": {
                  "predicted": [[1, 2, 3]], "reference": [[0, 0, 0]],
              }}}]}
    write(native, "history.jsonl", json.dumps(record) + "\n")
    collect_native("jax_soppi", native, output)
    rows = snapshot(output, "predictions.jsonl")
    assert [row["prediction"] for row in rows] == [1, 2, 3]
    assert len({row["training_entry_id"] for row in rows}) == 3
    assert all(row["identity_scope"] == "source_property_index" for row in rows)
