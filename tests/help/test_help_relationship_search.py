from __future__ import annotations

import pytest

from reaxkit.help.help_index_loader import (
    _score_metadata_entry,
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


@pytest.mark.parametrize("query", [
    "gen-elastic-trainingset", "generate elastic training set",
    "make elastic trainset", "generate elastci training set",
])
def test_elastic_workflow_ranks_first_without_unrelated_sections(query: str) -> None:
    report = build_help_relationship_report(query, top_k=1)
    assert "make-trainset-elastic" in report.split("WORKFLOW LEVEL", 1)[1]
    assert "get_rdf_property" not in report
    assert "UTILITY LEVEL" not in report
    assert "ANALYZER LEVEL" not in report


@pytest.mark.parametrize(("query", "expected"), [
    ("make heat of formation training set", "make-trainset-heatfo"),
    ("radial distribution function", "get_rdf"),
    ("mean squared displacement", "msd"),
    ("plot temperature", "get_temperature"),
    ("get_rdf_property", "get_rdf_property"),
    ("get-rdf-property", "get_rdf_property"),
])
def test_command_search_across_domains(query: str, expected: str) -> None:
    assert search_help_commands(query, top_k=1)[0].command == expected


@pytest.mark.parametrize("query", ["zzzxxyy", "", "please help", "elastic banana"])
def test_unrelated_queries_do_not_fill_requested_result_count(query: str) -> None:
    assert search_help_commands(query, top_k=100) == []
    assert build_help_relationship_report(query, top_k=100).startswith("No matches")


def test_shared_generic_metadata_cannot_outvote_missing_subject() -> None:
    entry = dict.fromkeys(
        ["aliases", "tags", "description", "notes", "help_search_examples"],
        "generate property training set " * 20,
    )
    assert _score_metadata_entry("get_rdf_property", entry, "gen elastic trainingset", set()) == 0


def test_exact_match_filters_command_matches_too() -> None:
    report = build_help_relationship_report("get_rdf_property", top_k=20, exact_match=True)
    commands = report.split("COMMAND MATCHES", 1)[1].split("\n\n", 1)[0]
    assert "get_rdf_property" in commands
    assert "- rdf_property " not in commands


def test_more_results_preserve_best_match_and_exclude_unrelated_commands() -> None:
    hits = search_help_commands("gen-elastic-trainingset", top_k=100)
    assert hits[0] == search_help_commands("gen-elastic-trainingset", top_k=1)[0]
    assert "make-trainset-elastic" in {hit.command for hit in hits}
    assert "get_rdf_property" not in {hit.command for hit in hits}
    assert "gen-plot" not in {hit.command for hit in hits}
    assert len(hits) < 100
