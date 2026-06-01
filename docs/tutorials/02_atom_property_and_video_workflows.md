# Atom-Property Plots and Video Workflows

This tutorial replaces the old `xmolout_fort7` tutorial with the current
ReaxKit command structure.

You will use:
- `plot_atom_property` for frame-wise 3D scatter or 2D heatmap plots of
  per-atom properties, and
- `gen-video` to convert saved frame images into an animation.

---

## Why this replaced the old tutorial

Earlier versions used an `xmolout_fort7` workflow namespace.  
Current ReaxKit exposes this behavior as direct commands:

- analysis/presentation: `plot_atom_property`
- video assembly: `gen-video`

The core concept is unchanged: combine geometry (usually from `xmolout`) and
per-atom properties (usually from `fort.7`) and visualize them over frames.

---

## Example 1: 3D scatter of partial charge

```bash
reaxkit plot_atom_property --type scatter3d --property charge --frames 0:20:5 --save charge_scatter_frames
```

This command:
- loads trajectory coordinates,
- loads per-atom charge values,
- aligns frame/atom data,
- saves one 3D image per selected frame.

---

## Example 2: 2D heatmap of bond-order sum

```bash
reaxkit plot_atom_property --type heatmap2d --property sum_BOs --plane xz --bins 60 --frames 0:20:5 --save bo_heatmaps
```

This command:
- projects atom coordinates to a 2D plane,
- aggregates property values on a 2D grid,
- saves one heatmap per frame.

---

## Key flags

`--property`
- Typical values: `charge`, `q`, `partial_charge`, `sum_BOs`
- Aliases are normalized internally.

`--frames` and `--every`
- `--frames` selects explicit frames/ranges.
- `--every` subsamples selected frames.

`--atom-ids` / `--atom-types`
- Restrict plotting to a subset of atoms.

`--save`
- Directory for per-frame images.

`--export`
- Exports assembled per-atom table (`x,y,z,value`) as CSV.

---

## Build a video from saved frames

After generating images:

```bash
reaxkit gen-video --folder reaxkit_outputs/plot_atom_property/charge_scatter_frames --output charge_scatter.mp4 --fps 10
```

This creates an MP4 from ordered frame images.

---

## Recommended workflow

1. Start with sparse sampling (for example `--frames 0:200:20`).
2. Validate scale/projection (`--vmin`, `--vmax`, `--plane`).
3. Export one CSV (`--export`) for QA and reproducibility.
4. Increase frame density for final rendering.
5. Use `gen-video` for publication-ready animations.

---

[Back to Tutorials](index.md)
