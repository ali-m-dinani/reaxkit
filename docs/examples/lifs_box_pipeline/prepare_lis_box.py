"""Li/S composition and random-packing preparation (engine-agnostic middle step)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LiSBoxPreparationConfig:
    """Configuration for preparing a Li/S random packed box."""

    desired_density_g_per_a3: float = 0.007645
    sulfur_mono_fraction: float = 0.12
    lithium_per_sulfur: float = 2.0
    sulfur_mass_amu: float = 32.0
    lithium_mass_amu: float = 7.0
    s2_bond_length_a: float = 1.9
    tolerance_a: float = 4.0
    max_attempts_per_fragment: int = 1000
    seed: int | None = 42


def plan_species_counts(total_s: int, cfg: LiSBoxPreparationConfig) -> tuple[int, int, int]:
    """Compute Li, S2, and atomic-S counts from total sulfur and composition config."""
    li_required = int(np.round(total_s * float(cfg.lithium_per_sulfur)))

    x_s = float(cfg.sulfur_mono_fraction)
    x_s2 = 1.0 - x_s
    if x_s < 0.0 or x_s > 1.0:
        raise ValueError("sulfur_mono_fraction must be in [0, 1].")

    total_species_moles = total_s / (2.0 * x_s2 + x_s)
    total_s2 = int(np.round(x_s2 * total_species_moles))
    total_s_atoms = int(np.round(x_s * total_species_moles))
    return li_required, total_s2, total_s_atoms


def required_box_size_a(total_s: int, li_required: int, cfg: LiSBoxPreparationConfig) -> float:
    """Compute cubic box length from mass and target density."""
    total_mass = total_s * cfg.sulfur_mass_amu + li_required * cfg.lithium_mass_amu
    req_vol = total_mass / cfg.desired_density_g_per_a3
    return float(req_vol ** (1.0 / 3.0))


def build_fragments(total_s2: int, total_s_atoms: int, li_required: int, cfg: LiSBoxPreparationConfig):
    """Build fragment templates as (symbols, coordinates) tuples."""
    fragments: list[tuple[list[str], np.ndarray]] = []
    for _ in range(total_s2):
        fragments.append(
            (
                ["S", "S"],
                np.asarray([[0.0, 0.0, 0.0], [cfg.s2_bond_length_a, 0.0, 0.0]], dtype=float),
            )
        )
    for _ in range(total_s_atoms):
        fragments.append((["S"], np.asarray([[0.0, 0.0, 0.0]], dtype=float)))
    for _ in range(li_required):
        fragments.append((["Li"], np.asarray([[0.0, 0.0, 0.0]], dtype=float)))
    return fragments


def place_fragments(
    fragments: list[tuple[list[str], np.ndarray]],
    box_size_a: float,
    cfg: LiSBoxPreparationConfig,
) -> tuple[list[str], np.ndarray]:
    """Random non-overlapping placement with minimum distance tolerance."""
    rng = np.random.default_rng(cfg.seed)
    placed_positions = np.empty((0, 3), dtype=float)
    symbols: list[str] = []
    coords_rows: list[np.ndarray] = []

    for fragment_symbols, fragment_coords in fragments:
        fragment_placed = False
        center = fragment_coords.mean(axis=0)
        rel = fragment_coords - center

        for _ in range(int(cfg.max_attempts_per_fragment)):
            random_pos = rng.uniform(cfg.tolerance_a, box_size_a - cfg.tolerance_a, size=3)
            new_positions = rel + random_pos

            if placed_positions.size:
                ok = True
                for pos in new_positions:
                    dists = np.linalg.norm(placed_positions - pos, axis=1)
                    if float(dists.min()) < cfg.tolerance_a:
                        ok = False
                        break
                if not ok:
                    continue

            symbols.extend(fragment_symbols)
            coords_rows.extend(new_positions)
            placed_positions = np.vstack([placed_positions, new_positions])
            fragment_placed = True
            break

        if not fragment_placed:
            raise RuntimeError(
                "Could not place fragment after "
                f"{cfg.max_attempts_per_fragment} attempts; "
                "consider reducing tolerance, reducing density, or increasing attempts."
            )

    return symbols, np.asarray(coords_rows, dtype=float)


def prepare_lis_box_from_total_s(total_s: int, cfg: LiSBoxPreparationConfig = LiSBoxPreparationConfig()) -> dict[str, object]:
    """Middle-step preparation from sulfur total -> packed atom data."""
    li_required, total_s2, total_s_atoms = plan_species_counts(total_s, cfg)
    box_size_a = required_box_size_a(total_s, li_required, cfg)
    fragments = build_fragments(total_s2, total_s_atoms, li_required, cfg)
    atom_types, coords = place_fragments(fragments, box_size_a, cfg)

    return {
        "atom_types": atom_types,
        "coords": coords,
        "box_size_a": box_size_a,
        "total_s": int(total_s),
        "li_required": int(li_required),
        "species_counts": {
            "S2": int(total_s2),
            "S": int(total_s_atoms),
            "Li": int(li_required),
        },
    }

