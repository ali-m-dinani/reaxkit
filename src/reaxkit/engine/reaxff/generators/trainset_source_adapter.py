"""
Source adapter layer for trainset generation.

This keeps workflow code source-agnostic and routes data fetching/generation
to source-specific scraper implementations.

**Usage context**

- Template generation: Produce canonical text payloads for ReaxFF artifacts.
- File writing: Persist generated outputs to disk with stable formatting.
- Workflow integration: Support higher-level ReaxKit workflow commands.
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
    """Represent HeatFoTrainsetRequest.

    Public class used by ReaxFF generator components.

    Fields
    ------
    out_dir : str | Path
        Dataclass field.
    elements : list[str]
        Dataclass field.
    material_ids : Optional[list[str]]
        Dataclass field.
    references_by_element : Dict[str, HeatFoReferenceSpec]
        Dataclass field.
    exact_element_count : bool
        Dataclass field.
    api_key : Optional[str]
        Dataclass field.
    max_materials : Optional[int]
        Dataclass field.
    crystallographic_setting_conversion : str
        Dataclass field.
    weight : float
        Dataclass field.
    trainset_filename : str
        Dataclass field.
    concatenated_geo_filename : str
        Dataclass field.
    verbose : bool
        Dataclass field.
    """
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
    """Represent TrainsetSourceAdapter.

    Public class used by ReaxFF generator components.
    """
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
        """Generate elastic settings yaml from material id.

        Parameters
        ----------
        mat_id : str
            Keyword-only parameter.
        out_yaml : str | Path
            Keyword-only parameter.
        structure_dir : Optional[str | Path]
            Keyword-only parameter.
        bulk_mode : str
            Keyword-only parameter.
        crystallographic_setting_conversion : str
            Keyword-only parameter.
        api_key : Optional[str]
            Keyword-only parameter.
        verbose : bool
            Keyword-only parameter.

        Returns
        -------
        Dict[str, str]
            Return value.

        Examples
        --------
        ```python
        # Example
        generate_elastic_settings_yaml_from_material_id(...)
        ```
        """
        ...

    def search_material_ids_by_elements(
        self,
        *,
        api_key: str,
        elements: list[str],
        exact_element_count: bool,
        max_materials: Optional[int],
    ) -> list[str]:
        """Search material ids by elements.

        Parameters
        ----------
        api_key : str
            Keyword-only parameter.
        elements : list[str]
            Keyword-only parameter.
        exact_element_count : bool
            Keyword-only parameter.
        max_materials : Optional[int]
            Keyword-only parameter.

        Returns
        -------
        list[str]
            Return value.

        Examples
        --------
        ```python
        # Example
        search_material_ids_by_elements(...)
        ```
        """
        ...

    def generate_heatfo_trainset(self, request: HeatFoTrainsetRequest):
        """Generate heatfo trainset.

        Parameters
        ----------
        request : HeatFoTrainsetRequest
            Input parameter.

        Returns
        -------
        Any
            Return value.

        Examples
        --------
        ```python
        # Example
        generate_heatfo_trainset(...)
        ```
        """
        ...


class _MaterialsProjectTrainsetSourceAdapter:
    """Represent MaterialsProjectTrainsetSourceAdapter."""
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
        """Generate elastic settings yaml from material id.

        Parameters
        ----------
        mat_id : str
            Keyword-only parameter.
        out_yaml : str | Path
            Keyword-only parameter.
        structure_dir : Optional[str | Path]
            Keyword-only parameter.
        bulk_mode : str
            Keyword-only parameter.
        crystallographic_setting_conversion : str
            Keyword-only parameter.
        api_key : Optional[str]
            Keyword-only parameter.
        verbose : bool
            Keyword-only parameter.

        Returns
        -------
        Dict[str, str]
            Return value.

        Examples
        --------
        ```python
        # Example
        generate_elastic_settings_yaml_from_material_id(...)
        ```
        """
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
        """Search material ids by elements.

        Parameters
        ----------
        api_key : str
            Keyword-only parameter.
        elements : list[str]
            Keyword-only parameter.
        exact_element_count : bool
            Keyword-only parameter.
        max_materials : Optional[int]
            Keyword-only parameter.

        Returns
        -------
        list[str]
            Return value.

        Examples
        --------
        ```python
        # Example
        search_material_ids_by_elements(...)
        ```
        """
        return _mp_search_material_ids_by_elements(
            api_key=api_key,
            elements=elements,
            exact_element_count=exact_element_count,
            max_materials=max_materials,
        )

    def generate_heatfo_trainset(self, request: HeatFoTrainsetRequest):
        """Generate heatfo trainset.

        Parameters
        ----------
        request : HeatFoTrainsetRequest
            Input parameter.

        Returns
        -------
        Any
            Return value.

        Examples
        --------
        ```python
        # Example
        generate_heatfo_trainset(...)
        ```
        """
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
    """Represent JarvisTrainsetSourceAdapter."""
    source_name = "jarvis"

    @staticmethod
    def _not_implemented() -> None:
        """Not implemented."""
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
        """Generate elastic settings yaml from material id.

        Parameters
        ----------
        mat_id : str
            Keyword-only parameter.
        out_yaml : str | Path
            Keyword-only parameter.
        structure_dir : Optional[str | Path]
            Keyword-only parameter.
        bulk_mode : str
            Keyword-only parameter.
        crystallographic_setting_conversion : str
            Keyword-only parameter.
        api_key : Optional[str]
            Keyword-only parameter.
        verbose : bool
            Keyword-only parameter.

        Returns
        -------
        Dict[str, str]
            Return value.

        Examples
        --------
        ```python
        # Example
        generate_elastic_settings_yaml_from_material_id(...)
        ```
        """
        self._not_implemented()

    def search_material_ids_by_elements(
        self,
        *,
        api_key: str,
        elements: list[str],
        exact_element_count: bool,
        max_materials: Optional[int],
    ) -> list[str]:
        """Search material ids by elements.

        Parameters
        ----------
        api_key : str
            Keyword-only parameter.
        elements : list[str]
            Keyword-only parameter.
        exact_element_count : bool
            Keyword-only parameter.
        max_materials : Optional[int]
            Keyword-only parameter.

        Returns
        -------
        list[str]
            Return value.

        Examples
        --------
        ```python
        # Example
        search_material_ids_by_elements(...)
        ```
        """
        self._not_implemented()

    def generate_heatfo_trainset(self, request: HeatFoTrainsetRequest):
        """Generate heatfo trainset.

        Parameters
        ----------
        request : HeatFoTrainsetRequest
            Input parameter.

        Returns
        -------
        Any
            Return value.

        Examples
        --------
        ```python
        # Example
        generate_heatfo_trainset(...)
        ```
        """
        self._not_implemented()


def _normalize_trainset_source_name(source: str) -> str:
    """Normalize trainset source name."""
    value = str(source).strip().lower()
    if not value:
        raise ValueError("Source must be provided.")
    return value


def _get_trainset_source_adapter(source: str) -> TrainsetSourceAdapter:
    """Get trainset source adapter."""
    source_name = _normalize_trainset_source_name(source)
    if source_name == "mp":
        return _MaterialsProjectTrainsetSourceAdapter()
    if source_name == "jarvis":
        return _JarvisTrainsetSourceAdapter()
    raise NotImplementedError(
        f"Unsupported source={source!r}. "
        "Supported today: 'mp' (fully) and 'jarvis' (placeholder adapter only)."
    )
