"""Screen-sized summaries of complete columnar artifacts, run in query workers.

The input remains immutable. SQL scans and aggregates spill to disk; only bounded
summaries become Python objects. Null-containing dense buckets are disconnected
conservatively, so decimation never draws a line across a missing observation.
"""
from __future__ import annotations

import json
import math

from reaxkit.webui.backend.artifact_tables import quote_column, table_descriptor
from reaxkit.webui.presentation.perf_config import load_ui_performance_config

RENDERER_VERSION = 1


def plot_budget(kind, width=1000):
    cfg = load_ui_performance_config()
    width = max(160, min(4096, int(width or 1000)))
    if kind == 'histogram':
        return min(cfg['histogram_max_bins'], max(16, width // 4))
    if kind == 'scatter3d':
        budget = cfg['scatter3d_max_points']
        if budget < 42:
            raise ValueError('3D geometry budget must be at least 42 points.')
        return budget
    budget = min(cfg['plot2d_max_points'], cfg['plot2d_initial_max_points'], width * 12)
    if budget < 12:
        raise ValueError('Line-plot budget must be at least 12 points to preserve extrema and gaps.')
    return budget


def plot_mapping(descriptor, request):
    cols = descriptor['columns']
    numeric = [c for c in cols if any(t in descriptor.get('types', {}).get(c, '')
               for t in ('int', 'float', 'double', 'decimal'))]
    kind = str(request.get('visualization_type') or 'plot2d').lower()
    if kind not in ('plot2d', 'histogram'):
        kind = 'scatter3d'
    x = request.get('x_col') or next((c for c in ('x', 'iter', 'frame_index') if c in cols), (numeric or cols or [''])[0])
    y = request.get('y_col') or next((c for c in numeric if c != x), x)
    if kind == 'histogram':
        y = x
    z = request.get('z_col') or ('z' if 'z' in cols else next((c for c in numeric if c not in (x, y)), y))
    frame = request.get('frame_col') or (next((c for c in ('frame_index', 'timestep', 'iter') if c in cols), '')
             if kind == 'scatter3d' and all(c in cols for c in ('x', 'y', 'z')) else '')
    return dict(kind=kind, x=x, y=y, z=z, group=request.get('group_col') or '',
                color=request.get('color_col') or '', frame=frame,
                atom=next((c for c in ('atom_id', 'id') if c in cols), ''))


def summarize_plot(tables, payload, request, *, width=1000, x_range=None, frame=None):
    import duckdb
    original = table_descriptor(payload)
    if original is None:
        raise ValueError('No tabular result to plot.')
    mapping = plot_mapping(original, request)
    # Legacy filters retain the same numeric/string semantics as table exports.
    descriptor = tables.filtered(original, request.get('row_filters'), include_row_ids=True)
    cfg = load_ui_performance_config()
    budget = plot_budget(mapping['kind'], width)
    spill = tables.write_dir / 'plot-spill'
    spill.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(config={'memory_limit': '256MB', 'threads': 1, 'temp_directory': str(spill)})
    try:
        def col(key):
            name = mapping[key]
            expr = 'source.' + quote_column(name, descriptor['columns'])
            return f"json_extract_string({expr}, '$')" if name in descriptor.get('encoded_columns', []) else expr
        def number(key):
            expr = f'TRY_CAST({col(key)} AS DOUBLE)'
            return f'CASE WHEN isfinite({expr}) THEN {expr} END'
        path = str(tables.path(descriptor))
        if descriptor.get('source_row_column'):
            source = 'read_parquet(?) AS source'
            row_id = 'source.' + quote_column(descriptor['source_row_column'], descriptor['columns'])
        elif 'file_row_number' not in descriptor['columns']:
            source = 'read_parquet(?, file_row_number=true) AS source'
            row_id = 'source.file_row_number'
        else:
            internal = '__rk_plot_row'
            while internal in descriptor['columns']:
                internal += '_'
            source = f'(SELECT *, row_number() OVER () - 1 AS "{internal}" FROM read_parquet(?)) AS source'
            row_id = f'source."{internal}"'
        where, parameters = [], [path]
        group_count = 0
        frame_bounds = None
        if mapping['kind'] == 'plot2d':
            group_expr = f'CAST({col("group")} AS VARCHAR)' if mapping['group'] else "'all'"
            max_groups = min(cfg['plot2d_max_curves_display'], max(1, budget // 12))
            if mapping['group']:
                group_count = con.execute(f'SELECT count(*) FROM (SELECT DISTINCT {group_expr} AS g FROM {source})', [path]).fetchone()[0]
                con.execute(f'CREATE TEMP TABLE groups AS SELECT DISTINCT {group_expr} AS g FROM {source} ORDER BY g NULLS FIRST LIMIT ?', [path, max_groups])
                where.append(f'EXISTS (SELECT 1 FROM groups WHERE groups.g IS NOT DISTINCT FROM {group_expr})')
            else:
                group_count = int(descriptor['row_count'] > 0)
                con.execute("CREATE TEMP TABLE groups AS SELECT 'all' AS g WHERE ?", [bool(group_count)])
            if x_range is not None:
                lo, hi = sorted(float(v) for v in x_range)
                if not all(math.isfinite(v) for v in (lo, hi)) or lo == hi:
                    raise ValueError('Invalid plot range')
                # Native numeric columns retain row-group statistics pruning.
                x_expr = col('x') if _native_numeric(descriptor, mapping['x']) else number('x')
                where.append(f'{x_expr} BETWEEN ? AND ?')
                integer_x = 'int' in descriptor.get('types', {}).get(mapping['x'], '') and _native_numeric(descriptor, mapping['x'])
                parameters.extend([math.ceil(lo), math.floor(hi)] if integer_x else [lo, hi])
        elif mapping['kind'] == 'scatter3d' and mapping['frame']:
            frame_bounds = con.execute(f'SELECT min({number("frame")}), max({number("frame")}) FROM {source}', [path]).fetchone()
            frame = frame_bounds[0] if frame is None else float(frame)
            if frame is not None and not math.isfinite(frame):
                raise ValueError('Invalid frame value')
            frame_expr = col('frame') if _native_numeric(descriptor, mapping['frame']) else number('frame')
            where.append(f'{frame_expr} = ?')
            integer_frame = 'int' in descriptor.get('types', {}).get(mapping['frame'], '') and _native_numeric(descriptor, mapping['frame'])
            parameters.append(int(frame) if integer_frame and frame is not None and float(frame).is_integer() else frame)
        projection = [f'{number("x")} AS x']
        if mapping['kind'] != 'histogram':
            projection += [f'{row_id} AS rid', f'{number("y")} AS y']
        if mapping['kind'] == 'plot2d':
            projection += [f'CAST({col("group")} AS VARCHAR) AS g' if mapping['group'] else "'all' AS g"]
        if mapping['kind'] == 'scatter3d':
            projection += [f'{number("z")} AS z', f'{number("color")} AS c' if mapping['color'] else 'NULL::DOUBLE AS c',
                           f'{number("frame")} AS f' if mapping['frame'] else '0::DOUBLE AS f',
                           f'CAST({col("atom")} AS VARCHAR) AS atom' if mapping['atom'] else 'NULL::VARCHAR AS atom']
        con.execute('CREATE TEMP TABLE points AS SELECT '
                    + ', '.join(projection) + f' FROM {source}'
                    + (' WHERE ' + ' AND '.join(where) if where else ''), parameters)
        result = dict(version=RENDERER_VERSION, revision=original['revision'], mapping=mapping,
                      total_rows=original['row_count'], matching_rows=descriptor['row_count'], budget=budget)
        if mapping['kind'] == 'histogram':
            result.update(_histogram(con, budget))
        elif mapping['kind'] == 'scatter3d':
            cell = _cell(payload.get('cell') or request.get('cell'))
            bond_limit = min(cfg['scatter3d_max_bonds'], max(0, (budget - 42) // 3)) if payload.get('bonds') else 0
            overlay_budget = (36 if cell else 0) + 3 * bond_limit
            result.update(_points3d(con, max(6, budget - overlay_budget), frame, cfg))
            if frame_bounds:
                result.update(frame_min=frame_bounds[0], frame_max=frame_bounds[1])
            result['cell'] = cell
            result['bonds'] = _bonds(tables, payload.get('bonds'), result['points'], bond_limit)
            result['geometry_points'] = result['displayed_points'] + (36 if cell else 0) + 3 * len(result['bonds'])
        else:
            result.update(_lines(con, request, budget, x_range, group_count))
        size = len(json.dumps(result, allow_nan=False, separators=(',', ':')).encode())
        if size > cfg['plot_max_bytes']:
            raise ValueError('Plot exceeds the payload byte budget. Reduce visible curves or use the table.')
        result['payload_bytes'] = size
        return result
    finally:
        con.close()


def _native_numeric(descriptor, name):
    return name not in descriptor.get('encoded_columns', []) and any(
        t in descriptor.get('types', {}).get(name, '') for t in ('int', 'float', 'double', 'decimal'))


def _lines(con, request, budget, x_range, group_count):
    invalid_x = con.execute('SELECT count(*) FROM points WHERE x IS NULL').fetchone()[0]
    if invalid_x:
        # Removing unknown x positions could create scientifically false continuity.
        raise ValueError(f'{invalid_x:,} rows have missing or nonnumeric x values. Choose a numeric x column or inspect the table.')
    mode = str(request.get('group_agg') or 'none').lower()
    aggregates = {'mean': 'avg(y)', 'median': 'median(y)', 'min': 'min(y)', 'max': 'max(y)',
                  'sum': 'sum(y)', 'count': 'count(y)', 'std': 'stddev_pop(y)'}
    if mode in aggregates:
        con.execute('CREATE TEMP TABLE aggregated AS SELECT x, g, min(rid) AS rid, '
                    + aggregates[mode] + ' AS y FROM points WHERE y IS NOT NULL GROUP BY x, g')
        con.execute('DROP TABLE points')
        con.execute('ALTER TABLE aggregated RENAME TO points')
    elif mode != 'none':
        raise ValueError('Unsupported group aggregation')
    if x_range is not None:
        lo, hi = sorted(float(v) for v in x_range)
        if not all(math.isfinite(v) for v in (lo, hi)) or lo == hi:
            raise ValueError('Invalid plot range')
        con.execute('DELETE FROM points WHERE x < ? OR x > ?', [lo, hi])
    lo, hi, count = con.execute('SELECT min(x), max(x), count(*) FROM points').fetchone()
    groups = con.execute('SELECT g FROM groups ORDER BY g NULLS FIRST').fetchall()
    per_group = max(12, budget // max(1, len(groups)))
    traces, displayed, reduced = [], 0, False
    for (group,) in groups:
        n = con.execute('SELECT count(*) FROM points WHERE g IS NOT DISTINCT FROM ?', [group]).fetchone()[0]
        xs, ys, ids = [], [], []
        if n <= per_group:
            rows = con.execute('SELECT x, y, rid FROM points WHERE g IS NOT DISTINCT FROM ? ORDER BY x, rid', [group]).fetchall()
            for x, y, rid in rows:
                xs.append(x); ys.append(y); ids.append(rid)
        elif n:
            reduced = True
            bins = max(1, per_group // 12)
            span = hi - lo or 1.0
            rows = con.execute('''WITH bucketed AS (
                SELECT *, least(? - 1, floor((x - ?) / ? * ?)) AS bucket
                FROM points WHERE g IS NOT DISTINCT FROM ?)
                SELECT first(struct_pack(x := x, y := y, rid := rid) ORDER BY x, rid),
                       last(struct_pack(x := x, y := y, rid := rid) ORDER BY x, rid),
                       arg_min(struct_pack(x := x, y := y, rid := rid), y),
                       arg_max(struct_pack(x := x, y := y, rid := rid), y),
                       bool_or(y IS NULL)
                FROM bucketed GROUP BY bucket ORDER BY bucket''', [bins, lo, span, bins, group]).fetchall()
            for first, last, minimum, maximum, gap in rows:
                candidates = {p['rid']: p for p in (first, last, minimum, maximum) if p is not None}
                for p in sorted(candidates.values(), key=lambda p: (p['x'], p['rid'])):
                    if gap:
                        xs.append(None); ys.append(None); ids.append(None)
                    xs.append(p['x']); ys.append(p['y']); ids.append(p['rid'])
                    if gap:
                        xs.append(None); ys.append(None); ids.append(None)
        displayed += len(xs)
        traces.append(dict(name='(missing)' if group is None else group, x=xs, y=ys,
                           row_ids=ids if mode == 'none' else [None] * len(ids), source_points=n))
    return dict(traces=traces, view_rows=count, displayed_points=displayed,
                group_count=group_count, shown_groups=len(groups), aggregation=mode,
                reduced=reduced, x_range=x_range,
                method='Per-range bucket endpoints and extrema; gaps conservatively disconnected')


def _histogram(con, bins):
    lo, hi, count = con.execute('SELECT min(x), max(x), count(x) FROM points').fetchone()
    missing = con.execute('SELECT count(*) - count(x) FROM points').fetchone()[0]
    if not count:
        return dict(bins=[], valid_rows=0, missing_rows=missing, displayed_points=0, reduced=False)
    if lo == hi:
        return dict(bins=[dict(left=lo, right=hi, count=count)], valid_rows=count,
                    missing_rows=missing, displayed_points=1, reduced=False)
    width = (hi - lo) / bins
    counts = dict(con.execute('SELECT least(? - 1, floor((x - ?) / ?))::INTEGER AS b, count(*) '
                             'FROM points WHERE x IS NOT NULL GROUP BY b', [bins, lo, width]).fetchall())
    return dict(bins=[dict(left=lo + i * width, right=hi if i == bins - 1 else lo + (i + 1) * width,
                           count=counts.get(i, 0)) for i in range(bins)], valid_rows=count,
                missing_rows=missing, displayed_points=bins, reduced=False)


def _points3d(con, budget, frame, cfg):
    first, last = con.execute('SELECT min(f), max(f) FROM points').fetchone()
    frame = first if frame is None else float(frame)
    con.execute('DELETE FROM points WHERE f IS DISTINCT FROM ?', [frame])
    total = con.execute('SELECT count(*) FROM points').fetchone()[0]
    con.execute('DELETE FROM points WHERE x IS NULL OR y IS NULL OR z IS NULL')
    count = con.execute('SELECT count(*) FROM points').fetchone()[0]
    # Deterministic full-frame sample plus all six spatial extrema. No row prefix.
    con.execute('CREATE TEMP TABLE selected AS SELECT rid FROM points ORDER BY hash(rid), rid LIMIT ?',
                [count if count <= budget else max(1, budget - 6)])
    for axis in ('x', 'y', 'z'):
        for order in ('ASC', 'DESC'):
            con.execute(f'INSERT INTO selected SELECT rid FROM points ORDER BY {axis} {order}, rid LIMIT 1')
    rows = con.execute('SELECT x, y, z, c, atom, rid FROM points WHERE rid IN (SELECT rid FROM selected) ORDER BY rid').fetchall()
    points = {key: [row[i] for row in rows] for i, key in enumerate(('x', 'y', 'z', 'color', 'atom_ids', 'row_ids'))}
    return dict(points=points, frame=frame, frame_min=first, frame_max=last,
                frame_rows=total, valid_rows=count, missing_rows=total-count,
                displayed_points=len(rows), reduced=len(rows) < count,
                method='Deterministic full-frame sample with spatial extrema' if len(rows) < count else 'All finite points in frame')


def _cell(value):
    """Optional explicit periodic cell; never infer a simulation box from atoms."""
    if not isinstance(value, dict):
        return None
    try:
        origin = [float(v) for v in value.get('origin', [0, 0, 0])]
        vectors = [[float(v) for v in row] for row in value['vectors']]
        if len(origin) == 3 and len(vectors) == 3 and all(len(r) == 3 for r in vectors) and all(
                math.isfinite(v) for v in origin + sum(vectors, [])):
            return dict(origin=origin, vectors=vectors)
    except (KeyError, TypeError, ValueError):
        pass
    return None


def _bonds(tables, value, points, limit):
    # Only explicit bonds whose endpoints are both visible; never guess bonds.
    if not limit:
        return []
    if isinstance(value, dict) and value.get('kind') == 'array' and len(value.get('shape', [])) == 2 and value['shape'][1] == 2:
        values = (row['value'] for row in tables.rows(value))
        value = zip(values, values)
    elif not isinstance(value, list):
        return []
    visible = {str(a) for a in points['atom_ids'] if a is not None}
    result = []
    for pair in value:
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and all(str(v) in visible for v in pair):
            result.append([str(v) for v in pair])
            if len(result) >= limit:
                break
    return result
