# Resources

This page lists external references that are either:
- directly used in ReaxKit development, or
- useful background for theory, methods, and physical models used in ReaxKit.

Resource types include: `Paper`, `Review`, `Book`, `Spec`, `Manual`,
`Tutorial`, and `Dataset`.

---

## Resources Used in Development

These references informed implementation details, algorithms, or validation.

| Type  | File                    | Specific Function/command                           | Purpose                                                               | Resource |
|-------|-------------------------|-----------------------------------------------------|-----------------------------------------------------------------------|----------|
| Paper | `equation_of_states.py` | `vinet_energy_ev` and `vinet_energy_trainset`       | Rose-Vinet equation of state                                          | Vinet, P., Ferrante, J., Rose, J. H., and Smith, J. R. (1987). *Compressibility of solids*. *J. Geophys. Res.*, 92(B9), 9319-9325. doi:10.1029/JB092iB09p09319 |


---

## Background, Theory, and Reference Material

These are optional references for deeper context.

| Type    | File                   | Specific Function                                 | Purpose                                                                                                                                      | Resource                                                                                                                              |
|---------|------------------------|---------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Manual  | `trainset_workflow.py` | `gen_elastic_trainset` and `gen_heatfo_trainset`  | Obtained cell angle and length values using Material's project API when generating training sets based on elastic and heat-of-formation data | https://docs.materialsproject.org/methodology/materials-methodology/understanding-structures-and-properties-in-the-materials-project  |

---

**Note:** whenever possible, references should also be cited inline in the
relevant module docstrings and docs pages. This page is a central index, not a
replacement for local citations.
