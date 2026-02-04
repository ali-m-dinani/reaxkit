# Handlers

Handlers are responsible for **reading and parsing ReaxFF files** into structured Python objects.

Each handler:

- Corresponds to **one specific file type**
- Converts raw text into a **clean `pandas.DataFrame`**
- Optionally exposes **frame-based access** for trajectory-like files
- Performs **no analysis or plotting**

## Common Handler Contract

All handlers follow the same high-level API:

```python
handler = SomeHandler("file_name")

df = handler.dataframe()     # main tabular data
meta = handler.metadata()    # parsed metadata
```

Some handlers additionally support:

* `n_frames()`
* `frame(i)`
* `iter_frames(step=...)`

Exact behavior is documented per handler.

---

## What Handlers Do (and Don’t)

✅ Parse raw files

✅ Normalize column names and aliases

✅ Drop duplicated iterations when appropriate

❌ No physics interpretation

❌ No plotting

❌ No CLI logic

---

