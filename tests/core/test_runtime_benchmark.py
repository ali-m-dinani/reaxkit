from __future__ import annotations

from reaxkit.core.runtime.benchmark import run_case, run_matrix


def test_benchmark_is_deterministic_across_worker_counts(tmp_path):
    report = run_matrix(
        frames=12,
        payload_bytes=1024,
        worker_counts=[1, 2, 4],
        workspace=tmp_path,
        include_file_source=True,
    )

    assert report["validation"] == {
        "deterministic": True,
        "bounded": True,
        "case_count": 9,
    }
    assert len({case["result_digest"] for case in report["cases"]}) == 1


def test_peak_payload_is_independent_of_trajectory_length():
    short = run_case(
        frames=8,
        payload_bytes=2048,
        workers=2,
        max_in_flight=4,
    )
    long = run_case(
        frames=80,
        payload_bytes=2048,
        workers=2,
        max_in_flight=4,
    )

    assert short.peak_in_flight_bytes <= 4 * 2048
    assert long.peak_in_flight_bytes <= 4 * 2048
    assert long.peak_in_flight_bytes == short.peak_in_flight_bytes
