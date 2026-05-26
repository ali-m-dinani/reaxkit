"""
Source adapter layer for trainset generation.

This keeps workflow code source-agnostic and routes data fetching/generation
to source-specific scraper implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol

from reaxkit.engine.reaxff.generators.trainset_heatfo import (
    HeatFoReferenceSpec,
    MaterialsProjectHeatFoSpec,
    _generate_heatfo_trainset_from_mp,
)
from reaxkit.engine.reaxff.generators.trainset_mp import (
    _generate_trainset_settings_yaml_from_mp_simple,
    _mp_search_material_ids_by_elements,
)


@dataclass(frozen=True)
class HeatFoTrainsetRequest:
    out_dir: str | Path
    elements: list[str]
    material_ids: Optional[list[str]]
    references_by_element: Dict[str, HeatFoReferenceSpec]
    exact_element_count: bool
    api_key: Optional[str]
    max_materials: Optional[int]
    crystallographic_setting_conversion: str
    weight: float
    trainset_filename: str
    concatenated_geo_filename: str
    verbose: bool


class TrainsetSourceAdapter(Protocol):
    source_name: str

    def generate_elastic_settings_yaml_from_material_id(
        self,
        *,
        mat_id: str,
        out_yaml: str | Path,
        structure_dir: Optional[str | Path],
        bulk_mode: str,
        crystallographic_setting_conversion: str,
        api_key: Optional[str],
        verbose: bool,
    ) -> Dict[str, str]:
        ...

    def search_material_ids_by_elements(
        self,
        *,
        api_key: str,
        elements: list[str],
        exact_element_count: bool,
        max_materials: Optional[int],
    ) -> list[str]:
        ...

    def generate_heatfo_trainset(self, request: HeatFoTrainsetRequest):
        ...


class _MaterialsProjectTrainsetSourceAdapter:
    source_name = "mp"

    def generate_elastic_settings_yaml_from_material_id(
        self,
        *,
        mat_id: str,
        out_yaml: str | Path,
        structure_dir: Optional[str | Path],
        bulk_mode: str,
        crystallographic_setting_conversion: str,
        api_key: Optional[str],
        verbose: bool,
    ) -> Dict[str, str]:
        return _generate_trainset_settings_yaml_from_mp_simple(
            mp_id=mat_id,
            out_yaml=out_yaml,
            structure_dir=structure_dir,
            bulk_mode=bulk_mode,
            crystallographic_setting_conversion=crystallographic_setting_conversion,
            api_key=api_key,
            verbose=verbose,
        )

    def search_material_ids_by_elements(
        self,
        *,
        api_key: str,
        elements: list[str],
        exact_element_count: bool,
        max_materials: Optional[int],
    ) -> list[str]:
        return _mp_search_material_ids_by_elements(
            api_key=api_key,
            elements=elements,
            exact_element_count=exact_element_count,
            max_materials=max_materials,
        )

    def generate_heatfo_trainset(self, request: HeatFoTrainsetRequest):
        return _generate_heatfo_trainset_from_mp(
            MaterialsProjectHeatFoSpec(
                out_dir=request.out_dir,
                elements=request.elements,
                material_ids=request.material_ids,
                references_by_element=request.references_by_element,
                exact_element_count=request.exact_element_count,
                api_key=request.api_key,
                max_materials=request.max_materials,
                crystallographic_setting_conversion=request.crystallographic_setting_conversion,
                weight=request.weight,
                trainset_filename=request.trainset_filename,
                concatenated_geo_filename=request.concatenated_geo_filename,
                verbose=request.verbose,
            )
        )


class _JarvisTrainsetSourceAdapter:
    source_name = "jarvis"

    @staticmethod
    def _not_implemented() -> None:
        raise NotImplementedError(
            "source='jarvis' is not implemented yet for trainset generation. "
            "Add a JARVIS scraper/provider and wire it into trainset_source_adapter."
        )

    def generate_elastic_settings_yaml_from_material_id(
        self,
        *,
        mat_id: str,
        out_yaml: str | Path,
        structure_dir: Optional[str | Path],
        bulk_mode: str,
        crystallographic_setting_conversion: str,
        api_key: Optional[str],
        verbose: bool,
    ) -> Dict[str, str]:
        self._not_implemented()

    def search_material_ids_by_elements(
        self,
        *,
        api_key: str,
        elements: list[str],
        exact_element_count: bool,
        max_materials: Optional[int],
    ) -> list[str]:
        self._not_implemented()

    def generate_heatfo_trainset(self, request: HeatFoTrainsetRequest):
        self._not_implemented()


def _normalize_trainset_source_name(source: str) -> str:
    value = str(source).strip().lower()
    if not value:
        raise ValueError("Source must be provided.")
    return value


def _get_trainset_source_adapter(source: str) -> TrainsetSourceAdapter:
    source_name = _normalize_trainset_source_name(source)
    if source_name == "mp":
        return _MaterialsProjectTrainsetSourceAdapter()
    if source_name == "jarvis":
        return _JarvisTrainsetSourceAdapter()
    raise NotImplementedError(
        f"Unsupported source={source!r}. "
        "Supported today: 'mp' (fully) and 'jarvis' (placeholder adapter only)."
    )
