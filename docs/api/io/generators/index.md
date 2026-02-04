# Generators

---

Generators are responsible for **creating ReaxFF input files programmatically**.

They are the inverse of handlers:

- Handlers → read existing files
- Generators → write new files

Generators enable:

- Reproducible simulation setup
- Scripted workflows
- Parameter sweeps and automation

## Generator Philosophy

Each generator:

- Produces **one specific ReaxFF file**
- Accepts structured Python inputs (dicts, arrays, settings objects)
- Writes **formatted, human-readable files**
- Avoids embedding simulation logic or execution

## Typical Usage Pattern

```python
from reaxkit.io.generators import SomeGenerator

gen = SomeGenerator(settings)
gen.write("output_file")
```

## What Generators Do (and Don’t)

✅ Write valid ReaxFF input files

✅ Enforce formatting and ordering rules

✅ Make implicit defaults explicit

❌ No parsing of simulation output

❌ No job submission

❌ No result analysis

---


