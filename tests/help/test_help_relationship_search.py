from __future__ import annotations

from reaxkit.help.help_index_loader import (
    build_help_relationship_report,
    load_engine_data_maps,
    search_help_commands,
)


def test_heatfo_query_surfaces_trainset_command() -> None:
    hits = search_help_commands("make trainset using heatfo", top_k=8, min_score=35.0)

    assert hits
    assert hits[0].command == "make-trainset-heatfo"


def test_coords_query_maps_to_trajectory_analyzer_path() -> None:
    report = build_help_relationship_report("coords", top_k=8, min_score=35.0, engine="reaxff", all_info=False)

    assert "DATACLASS -> ANALYZER -> WORKFLOW" in report
    assert "cli command: timeseries" in report
    assert "no query-matched analyzer tasks for mapped dataclasses" not in report


def test_relationship_report_includes_command_matches_section() -> None:
    report = build_help_relationship_report(
        "make trainset using heatfo",
        top_k=8,
        min_score=35.0,
        engine="reaxff",
        all_info=False,
    )

    assert "COMMAND MATCHES" in report
    assert "make-trainset-heatfo --generator" in report


def test_engine_map_is_loaded_from_help_search_index() -> None:
    maps = load_engine_data_maps()

    assert "reaxff" in maps
    assert maps["reaxff"]["loader_map_file"] == "reaxff_map.py"
