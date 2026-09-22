# Release notes

## Unreleased

- Added a persistent, dependency-free source-frame cache for overlapping
  xmolout and fort.7 selections. Finite selections now reuse individual parsed
  frames across commands and processes and seek directly to uncached frames.
- Added `--input-cache` / `--no-input-cache`, a 10 GiB default
  `--frame-cache-max-gb` limit, cache hit/miss diagnostics, and frame-cache
  inspection and clearing through `manage-workspace --folder cache/frames`.
- Kept analysis-result caching independent, so changes such as dynamic versus
  formal charge still produce separate results while sharing compatible input
  frames.
