# `iopt` file — Force field optimization switch

The **`iopt`** file controls whether ReaxFF runs in **normal simulation mode** or **force field optimization mode**.

This file exists because the force field optimization routines are implemented as an **external shell** around the core ReaxFF code. As a result, the standard **`control`** file cannot be used to switch between normal and optimization runs.

---

## Purpose of the `iopt` file

- Acts as a **mode selector** for ReaxFF
- Read by the external optimization shell
- Copied by the `exe` script to **`fort.20`**
- Determines whether force field optimization logic is activated

---

## File format

The `iopt` file contains **exactly one integer**, using format `i3`:

```text
<value>
```

### Allowed values

| Value | Meaning |
|---|---|
| `0` | Normal ReaxFF run |
| `1` | Force field optimization run |

---

## Example

```text
1
```

This configuration instructs ReaxFF to execute a **force field optimization** run.

---

## Behavior when missing

Although the `iopt` file is **officially mandatory**, most workflows do not require the user to create it manually.

- If the `iopt` file is **missing**, the `exe` script:
  - Automatically creates **`fort.20`**
  - Assigns a default value of `0`
- This results in a **normal (non-optimization) run**

---

## Relation to the `exe` script

- The `exe` script is responsible for:
  - Copying `iopt` → `fort.20`
  - Ensuring a valid default exists if `iopt` is absent
- Users typically do **not** need to interact with `iopt` directly unless performing force field optimization

---

## Summary

- `iopt` is a **single-value control file**
- It switches ReaxFF between **normal** and **optimization** modes
- Defaults to `0` (normal run) if not explicitly provided
- Required for advanced force field fitting workflows
