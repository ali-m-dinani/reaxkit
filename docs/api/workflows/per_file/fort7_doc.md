# Fort7 Workflow

CLI namespace: `reaxkit fort7 <task> [flags]`

fort.7 analysis workflow for ReaxKit.

This workflow provides tools for inspecting and analyzing ReaxFF `fort.7` files,
which contain per-frame bond-order information and derived atomic connectivity.

It supports:
- Extracting atom-level or system-level features (e.g. charges, coordination,
  bond-order–derived quantities) as functions of frame, iteration, or time.
- Building explicit connectivity representations (edge lists) from bond orders,
  with configurable thresholds and directionality.
- Computing connection statistics over time, such as mean or maximum coordination.
- Generating bond-order time series for specific atom pairs.
- Detecting bond formation and breakage events using threshold and hysteresis
  criteria, with optional diagnostic visualizations.

The workflow is designed to bridge raw ReaxFF bond-order output with higher-level
connectivity, dynamics, and event-based analyses in a reproducible, CLI-driven way.

## Available tasks

### `bond-events`

#### Examples

- `reaxkit fort7 bond-events --export events.csv`
- `reaxkit fort7 bond-events --src 1 --dst 19 --threshold 0.38 --hysteresis 0.10 --smooth ema --window 7 --min-run 4 --export events_1_19.csv --save overlay.png`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.7 file. |
| `--frames FRAMES` | Frame selection. |
| `--src SRC` | Source atom. |
| `--dst DST` | Destination atom. |
| `--threshold THRESHOLD` | Schmitt trigger base threshold. |
| `--hysteresis HYSTERESIS` | Hysteresis width around threshold. |
| `--smooth {ma,ema,none}` | Smoothing method. |
| `--window WINDOW` | Window size for MA/EMA. |
| `--ema-alpha EMA_ALPHA` | Optional EMA alpha. |
| `--min-run MIN_RUN` | Minimum consecutive points. |
| `--xaxis {iter,frame}` | Internal event x-axis. |
| `--directed` | Do not merge A–B/B–A. |
| `--save SAVE` | Save debug overlay (requires --src --dst). |
| `--export EXPORT` | Export detected events CSV. |

### `bond-ts`

#### Examples

- `reaxkit fort7 bond-ts --frames 0:500 --export bo.csv`
- `reaxkit fort7 bond-ts --src 1 --dst 19 --plot`
- `reaxkit fort7 bond-ts --wide --bo-threshold 0.1 --export wide.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.7 file. |
| `--frames FRAMES` | Frame selection. |
| `--directed` | Do not merge A–B with B–A. |
| `--bo-threshold BO_THRESHOLD` | Zero out BO below this. |
| `--wide` | Return wide matrix (frames × bonds). |
| `--xaxis {iter,frame,time}` | X-axis for quick plot. |
| `--control CONTROL` | Control file for --xaxis time. |
| `--src SRC` | Source atom for quick plot. |
| `--dst DST` | Destination atom for quick plot. |
| `--plot` | Show plot interactively. |
| `--save SAVE` | Save bond time-series plot. |
| `--export EXPORT` | Export bond-order time series CSV. |

### `constats`

#### Examples

- `reaxkit fort7 constats --frames 0:1000 --how mean --export stats.csv`
- `reaxkit fort7 constats --how count --min-bo 0.4 --export counts.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.7 file. |
| `--frames FRAMES` | Frame selection. |
| `--min-bo MIN_BO` | BO threshold before stats. |
| `--directed` | Do not merge A–B with B–A. |
| `--how {mean,max,count}` | Statistic to compute. |
| `--save SAVE` | (Unused) No plot. |
| `--export EXPORT` | Export connection stats as CSV. |

### `edges`

#### Examples

- `reaxkit fort7 edges --frames 0:1000:10 --min-bo 0.4 --export edges.csv`
- `reaxkit fort7 edges --plot --min-bo 0.3`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.7 file. |
| `--frames FRAMES` | Frame selection. |
| `--min-bo MIN_BO` | Minimum BO. |
| `--directed` | Treat edges as directed. |
| `--aggregate {max,mean}` | Aggregation for undirected edges. |
| `--include-self` | Keep self-edges. |
| `--xaxis {iter,frame,time}` | X-axis for quick plot. |
| `--control CONTROL` | Control file for --xaxis time. |
| `--plot` | Show plot interactively. |
| `--save SAVE` | Save edge-count plot. |
| `--export EXPORT` | Export edge list CSV. |

### `get`

#### Examples

- `reaxkit fort7 get --yaxis charge --atom 'all' --plot`
- `reaxkit fort7 get --yaxis charge --atom 1 --plot`
- `reaxkit fort7 get --yaxis q_.* --regex --export charges.csv to get all columns starting with 'q_'`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.7 file. |
| `--yaxis YAXIS` | Feature name or regex (with --regex). |
| `--atom ATOM` | Atom selection: a number (e.g., 5), a list (e.g., 1,2,7), or 'all'. |
| `--frames FRAMES` | Frame selection: 'a:b[:c]' or 'i,j,k'. |
| `--xaxis {iter,frame,time}` | X-axis mode. |
| `--control CONTROL` | Control file (for --xaxis time). |
| `--regex` | Interpret --feature as regex, which calls ally-columns with the given --yaxis |
| `--plot` | Show plot interactively. |
| `--save SAVE` | Save feature plot image. |
| `--export EXPORT` | Export extracted feature(s) as CSV. |
