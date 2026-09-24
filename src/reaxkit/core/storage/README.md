# core/storage

## Purpose
Provides storage layout, cache management, and parsed artifact persistence for run-scoped reproducible execution.

## What Belongs Here
- Cache key/value persistence for analysis results.
- Source-oriented frame payload and byte-offset persistence.
- Run/index directory layout conventions.
- Parsed artifact serialization utilities.

## What Does Not Belong Here
- UI-level export formatting.
- Domain-specific analysis computation.

## Structure
- `cache_manager.py`
- `frame_store.py`
- `storage_layout.py`
- `parsed_store.py`

## Flow
Runtime writes/reads parsed and analyzed artifacts via this subpackage to support cache hits and reproducibility.

## Frame cache contract

`frame_store.py` defines the command-independent cache foundation for partially
overlapping frame requests. A cache identity has four levels:

1. logical source path, engine, and source kind;
2. validated source generation;
3. parser version, representation, capabilities, and parse options;
4. zero-based source-frame index.

Representations describe payload schemas. Capabilities describe the information
contained by a representation. Callers may reuse a richer view for a narrower
request only when `FrameViewIdentity.can_satisfy` confirms the required
capabilities; a narrow record must never satisfy a richer request implicitly.

Parsed-input caching is separate from analysis-result caching. The existing
`cache` and `no_cache` runtime fields continue to govern analysis results.
`--input-cache` is enabled by default; `--no-input-cache` forces source reads
without changing analysis-result caching. `--frame-cache-max-gb` sets the
workspace limit (10 GiB by default, 0 for unlimited).

The frame store uses standard-library SQLite and pickle payloads with SHA-256
checksums. It does not require HDF5 or Parquet. Schema incompatibility is
rejected explicitly; missing or corrupt frame records are treated as cache
misses.

Explicit finite xmolout and fort.7 selections use an incremental byte-offset
index. A later selection reads cached frames and seeks directly to missing
frames, even when the two selections differ. Full cached records may serve
coordinates-only or charge-only consumers. Unbounded streams intentionally
remain one-frame-at-a-time source reads and are not persisted automatically.

Stores live under `cache/frames`, and `cache/index/frames.json` makes them
discoverable. Source size, nanosecond modification time, and bounded head/tail
signatures select a source generation; parser versions and parse options select
a view. Changed inputs or parser contracts therefore create new stores rather
than mixing records. `manage-workspace --folder cache/frames --action list`
inspects them, and `--action delete` clears the frame stores and their index.

The core store is engine-neutral. ReaxFF xmolout and fort.7 have tested random
access implementations. LAMMPS dump and AMS inputs remain on their existing
paths until their format-specific offset and identity rules have equivalent
coverage.

Long readers hold a `FrameStore.session()`: one connection per calling thread
and a cross-process SQLite lease outside the evictable generation. Quota
eviction holds an exclusive lease through deletion, so active streams cannot
be removed by a peer writer. A process exit releases its lease automatically.
Explicit administrative cache deletion should be done with readers stopped.

Quota maintenance uses file sizes, not payload-table inventory queries. It
runs after approximately 64 MiB of estimated growth (less for small quotas),
30 seconds, budget pressure, or stream closure. This is a reconciled workspace
budget, not a strict instantaneous filesystem cap: SQLite pages/WAL, small
offset metadata, and concurrent writers can temporarily exceed it. When pinned
stores leave no room, optional payload writes pause while reading continues.
Necessary offset metadata can still grow; unpinned generations can be reclaimed
at closure. Setting the limit to zero explicitly permits unlimited storage.

Frame counts live in a small transactionally maintained summary. Existing
schema-v1 stores gain it additively using a one-time primary-key-index count;
valid payloads are preserved. Warm reads update that summary at most every
30 seconds instead of touching every BLOB row. Frame offsets are also recorded
during a cold forward scan, so a later overlapping selection can reuse them.

See [the h-BN benchmark guide](../../../../benchmarks/HBN_PERFORMANCE.md)
for metrics, reproducible bounded comparisons, and the outstanding production
wall-time acceptance test.

## Extension Points
- Extend storage schema/index versioning in `storage_layout.py`.
