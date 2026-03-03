"""Tests for task alias resolution."""

from __future__ import annotations

import pytest

from reaxkit.core.command_registry import COMMAND_REGISTRY, register_generator_command
from reaxkit.core.task_resolution_using_alias import (
    build_task_alias_index,
    is_known_command,
    resolve_command_name,
    is_known_task,
    resolve_task_name,
)


def test_build_task_alias_index_includes_separator_variants():
    alias_index = build_task_alias_index(["mean_square_displacement"])

    assert alias_index["meansquaredisplacement"] == "mean_square_displacement"


def test_resolve_task_name_matches_explicit_alias():
    resolved = resolve_task_name(
        "mean-square-displacement",
        task_names=["msd", "rdf"],
        aliases={"msd": ["mean-square-displacement", "mean_square_displacement"]},
    )

    assert resolved == "msd"


def test_resolve_task_name_matches_case_insensitively():
    resolved = resolve_task_name("MSD", task_names=["msd", "rdf"])

    assert resolved == "msd"


def test_resolve_command_name_uses_registry_aliases():
    resolved = resolve_command_name("mean-square-displacement", task_names=["msd", "rdf"])

    assert resolved == "msd"


def test_resolve_task_name_matches_separator_variants_without_explicit_alias():
    resolved = resolve_task_name("mean-square-displacement", task_names=["mean_square_displacement"])

    assert resolved == "mean_square_displacement"


def test_is_known_task_false_for_unknown_value():
    assert is_known_task("unknown-task", task_names=["msd", "rdf"]) is False


def test_resolve_command_name_raises_helpful_error():
    with pytest.raises(KeyError) as exc:
        resolve_command_name("msx", task_names=["msd", "rdf"])

    assert "Unknown command alias 'msx'." in str(exc.value)
    assert "msd" in str(exc.value)


def test_registered_generator_command_participates_in_resolution():
    name = "trainset-yaml"
    try:
        register_generator_command(
            name,
            aliases=["trainset_yaml", "generate-trainset-yaml"],
        )
        assert resolve_command_name("generate-trainset-yaml") == name
        assert is_known_command("trainset_yaml") is True
    finally:
        COMMAND_REGISTRY.pop(name, None)
