# Tutorials

This section contains step-by-step tutorials that build from a minimal “it runs” example
to multi-file analysis and finally to a file-agnostic plotting workflow.

If you’re new, follow them **in order**.

---

## Tutorial sequence

1. **[01 — Understanding the Quick Start](01_understanding_quickstart.md)**  
   What happens internally when you run a simple ReaxKit command (workflows → handlers → analyzers),
   and how to think about **frame vs iter vs time**.

2. **[02 — xmolout + fort.7: Coordination and multi-file analysis](02_xmolout_fort7_coordination.md)**  
   The first multi-file workflow: aligning geometry (`xmolout`) with chemistry (`fort.7`) to enable
   coordination-style analysis and spatial property visualization.

3. **[03 — Generating electric-field schedules (eregime.in)](03_eregime_generator.md)**  
   Input generation: producing reproducible `eregime.in` protocols (sinusoidal, smooth pulses,
   and arbitrary functions) for electric-field–driven simulations.

4. **[04 — Meta plotting with plotter](04_plotter_meta_workflow.md)**  
   A file-agnostic CLI plotting workflow for any tabular data (single, directed, dual-axis, tornado,
   3D scatter, 2D heatmaps).

---

## Tips

- Start with **sparse frame sampling**, then refine.
- Use **frame** for selection, and **iter/time** for interpretation and plots.
- Keep raw outputs and generated inputs under version control when possible.

---

## Related docs

- **Examples:** see [examples](../examples/README.md) for runnable scripts and mini demos.
- **ReaxFF file reference:** see [ReaxFF references](../reaxff_reference/index.md) for input/output file semantics.
