import pytest

from reaxkit.optimization.conversion import collect_native
from .test_converters import snapshot, write


def test_mcff_annotations_initial_and_nonfinite(tmp_path):
    native = tmp_path / "native"
    output = tmp_path / "normalized"
    path = write(native, "MCFFOptimizer.log",
                 "Iter Repl Current error Changed Bounded Accepted Rejected AcceptRate Beta MaxStep Distance Elapsed\n"
                 "0 1 0.0* 0 0 1* 0 50 0.001692 1 0 0.36\n"
                 "1 1 3.0 3 0 2* 0 50 0.001692 1 1.09 0.10\n"
                 "2 1 nan 1 0 3* 0 50 0.001692 1 1.09 0.10\n"
                 "3 1 4")
    report = collect_native("ams_mcff", native, output)
    rows = snapshot(output, "progress.jsonl")
    assert len(rows) == 3
    assert [row["best_score"] for row in rows] == [0, 0, 0]
    assert rows[0]["score_kind"] == "initial"
    assert rows[1]["elapsed_seconds"] is None
    assert rows[1]["native_elapsed_seconds"] == 0.10
    assert rows[2]["status"] == "nonfinite"
    assert collect_native("ams_mcff", native, output)["snapshot"] == report["snapshot"]
    path.write_text(path.read_text() + " 1 0 4 0 50 1 1 1 0.1\n")
    assert collect_native("ams_mcff", native, output)["progress_count"] == 4


def block(count, score, best):
    return (f"# --------- timestamp\n N 12\nfunction evaluations {count}\n"
            f"elapsed (CPU) time [s] 3.28\nfunction value f(x)={score}\n"
            f"xbestever found after 1 evaluations, function value {best}\n"
            "xmean\n 123 456\n")


@pytest.mark.parametrize("baseline", ["1\n4\n5\n6\n7\n", "1\n99\n5\n6\n7\n", "1\n4"])
def test_cmaes_blocks_baseline_and_partial_report(tmp_path, baseline):
    native, output = tmp_path / "native", tmp_path / "normalized"
    write(native, "cmaes_progress.txt", block(2, 4, 4) + block(4, 6, 4)
          + "# --------- next\nfunction evaluations 6\n")
    write(native, "fort.13", baseline)
    report = collect_native("ams_cmaes", native, output)
    rows = snapshot(output, "progress.jsonl")
    assert len(rows) == 2
    assert [row["score"] for row in rows] == [4, 6]
    assert [row["native_step"] for row in rows] == [2, 4]
    assert rows[-1]["best_score"] == 4
    assert rows[-1]["step_kind"] == "evaluation"
    assert rows[-1]["cpu_seconds"] == 3.28
    assert rows[-1]["elapsed_seconds"] is None
    assert rows[-1].get("initial_score") == (1 if baseline.startswith("1\n4\n") else None)
    assert any("deferred" in warning for warning in report["warnings"])
    assert collect_native("ams_cmaes", native, output)["snapshot"] == report["snapshot"]
