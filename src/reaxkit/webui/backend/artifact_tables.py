"""Immutable Parquet tables and bounded, parameterized result queries.

Descriptors are internal storage objects, never user-supplied file paths. Consumers
must authorize the artifact against its pipeline before calling this module.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

TABLE_MARKER = '__reaxkit_table_v1__'
MAX_PAGE_SIZE = 500
MAX_PAGE_BYTES = 2 * 1024 * 1024


def is_table(value):
    return isinstance(value, dict) and value.get(TABLE_MARKER) is True


def table_descriptor(payload):
    for key in ('table', 'rows', 'records', 'data'):
        if is_table(payload.get(key)) and payload[key].get('kind') != 'array':
            return payload[key]
    return next((v for v in payload.values() if is_table(v) and v.get('kind') != 'array'), None)


def quote_column(name, columns):
    if name not in columns:
        raise ValueError(f'Unknown column: {name}')
    if isinstance(columns, dict):
        return columns[name]
    return '"' + name.replace('"', '""') + '"'


def compile_filter(tree, columns, depth=0):
    """Compile a Dash filter AST without accepting SQL or Python expressions."""
    if not tree:
        return 'TRUE', []
    if depth > 20 or not isinstance(tree, dict):
        raise ValueError('Invalid or excessively nested filter')
    kind, op = tree.get('type'), tree.get('subType')
    if kind == 'open-block':
        return compile_filter(tree.get('block'), columns, depth + 1)
    if kind == 'logical-operator' and op in ('&&', '||'):
        left, lp = compile_filter(tree.get('left'), columns, depth + 1)
        right, rp = compile_filter(tree.get('right'), columns, depth + 1)
        return f'({left} {"AND" if op == "&&" else "OR"} {right})', lp + rp
    def operand(value):
        if not isinstance(value, dict) or value.get('type') != 'expression':
            raise ValueError('Invalid filter operand')
        if value.get('subType') == 'field':
            return quote_column(str(value.get('value')), columns), []
        if value.get('subType') == 'value' and isinstance(value.get('value'), (str, int, float, bool, type(None))):
            return '?', [value.get('value')]
        raise ValueError('Unsupported filter value')
    if kind == 'relational-operator':
        left, lp = operand(tree.get('left'))
        right, rp = operand(tree.get('right'))
        if left.startswith('json_extract_string'):
            rp = [str(v) if v is not None else None for v in rp]
        aliases = {'eq': '=', 'ne': '!=', 'ge': '>=', 'gt': '>', 'le': '<=', 'lt': '<'}
        op = aliases.get(op, op)
        if rp == [None] and op in ('=', '!='):
            return f'({left} IS {"NOT " if op == "!=" else ""}NULL)', lp
        if op in ('=', '!=', '>=', '>', '<=', '<'):
            return f'({left} {op} {right})', lp + rp
        if op == 'contains':
            return f'contains(lower(CAST({left} AS VARCHAR)), lower(CAST({right} AS VARCHAR)))', lp + rp
        if op == 'datestartswith':
            return f'starts_with(CAST({left} AS VARCHAR), CAST({right} AS VARCHAR))', lp + rp
    if kind == 'unary-operator' and op in ('is nil', 'is blank'):
        left, params = operand(tree.get('left'))
        if op == 'is blank':
            return f"({left} IS NULL OR CAST({left} AS VARCHAR) = '')", params
        return f'({left} IS NULL)', params
    raise ValueError(f'Unsupported filter operator: {op}')


def json_cell(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


class ArtifactTables:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.write_dir = self.root

    def path(self, descriptor):
        name = descriptor['file']
        path = (self.root / name).resolve()
        if self.root not in path.parents or path.suffix != '.parquet':
            raise ValueError('Invalid artifact table reference')
        return path

    def write(self, value):
        if is_table(value):
            return value
        encoded_columns = []
        try:
            if isinstance(value, pa.Table):
                table = value
            elif hasattr(value, 'columns'):
                table = pa.Table.from_pandas(value, preserve_index=False)
            else:
                table = pa.Table.from_pylist(value)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            import pandas as pd
            frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
            arrays = {}
            for column in frame.columns:
                values = frame[column].tolist()
                try:
                    arrays[str(column)] = pa.array(values, from_pandas=True)
                except (pa.ArrowInvalid, pa.ArrowTypeError):
                    encoded_columns.append(str(column))
                    arrays[str(column)] = pa.array([None if v is None else json.dumps(v, default=str) for v in values])
            table = pa.table(arrays)
        if len(set(table.column_names)) != len(table.column_names):
            raise ValueError('Result table columns must be unique')
        name = uuid4().hex + '.parquet'
        self.write_dir.mkdir(parents=True, exist_ok=True)
        target = self.write_dir / name
        temporary = target.with_suffix('.tmp')
        pq.write_table(table, temporary, row_group_size=65_536)
        temporary.replace(target)
        sample_size = min(50, table.num_rows)
        while sample_size and table.slice(0, sample_size).nbytes > 64 * 1024:
            sample_size //= 2
        preview = table.slice(0, sample_size).to_pylist()
        if len(json.dumps(preview, default=str).encode()) > 64 * 1024:
            preview = []
        return {TABLE_MARKER: True, 'file': target.relative_to(self.root).as_posix(), 'revision': name[:-8],
                'row_count': table.num_rows, 'columns': table.column_names,
                'byte_count': table.nbytes,
                'types': {f.name: str(f.type) for f in table.schema},
                'encoded_columns': encoded_columns,
                'preview': preview}

    def persist_payload(self, payload):
        import pandas as pd
        import numpy as np
        result = {}
        for key, value in payload.items():
            if isinstance(value, (pd.DataFrame, pa.Table)) or (isinstance(value, list) and value and isinstance(value[0], dict)):
                result[key] = self.write(value)
            elif isinstance(value, (pd.Series, np.ndarray)) or (isinstance(value, list) and len(value) > 1024):
                array = np.asarray(value)
                descriptor = self.write(pd.DataFrame({'value': array.reshape(-1)}))
                result[key] = {**descriptor, 'kind': 'array', 'shape': list(array.shape)}
            else:
                result[key] = value
        return result

    def iter_batches(self, descriptor):
        yield from pq.ParquetFile(self.path(descriptor)).iter_batches(batch_size=8192)

    def rows(self, descriptor):
        for batch in self.iter_batches(descriptor):
            for row in batch.to_pylist():
                for column in descriptor.get('encoded_columns', []):
                    if row[column] is not None:
                        row[column] = json.loads(row[column])
                yield row

    def preview(self, descriptor, limit=12_000):
        """Bounded preview for legacy renderers; never use for scientific utilities."""
        result = []
        remaining_bytes = 4 * 1024 * 1024
        if descriptor.get('byte_count', 0) // max(1, descriptor['row_count']) > remaining_bytes:
            return result
        for batch in self.iter_batches(descriptor):
            take = min(len(batch), limit - len(result))
            while take and batch.slice(0, take).nbytes > remaining_bytes:
                take //= 2
            if not take:
                break
            sample = batch.slice(0, take)
            remaining_bytes -= sample.nbytes
            rows = sample.to_pylist()
            for row in rows:
                for column in descriptor.get('encoded_columns', []):
                    if row[column] is not None:
                        row[column] = json.loads(row[column])
            result.extend(rows)
            if len(result) >= limit or take < len(batch):
                break
        return result

    def filtered(self, descriptor, filters, *, include_row_ids=False):
        """Stream legacy presentation filters into a worker-local Parquet table."""
        if not filters:
            return descriptor
        from reaxkit.webui.backend.row_filters import _apply_row_filters
        source = pq.ParquetFile(self.path(descriptor))
        schema = source.schema_arrow
        row_column = '__rk_source_row'
        while row_column in schema.names:
            row_column += '_'
        if include_row_ids:
            schema = schema.append(pa.field(row_column, pa.int64()))
        self.write_dir.mkdir(parents=True, exist_ok=True)
        name = uuid4().hex
        path = self.write_dir / (name + '.parquet')
        count, preview = 0, []
        offset = 0
        with pq.ParquetWriter(path, schema) as writer:
            for batch in source.iter_batches(batch_size=8192):
                rows = batch.to_pylist()
                for index, row in enumerate(rows):
                    if include_row_ids:
                        row[row_column] = offset + index
                    for col in descriptor.get('encoded_columns', []):
                        if row[col] is not None:
                            row[col] = json.loads(row[col])
                rows = _apply_row_filters(rows, filters)
                offset += len(batch)
                if rows:
                    for row in rows:
                        for col in descriptor.get('encoded_columns', []):
                            if row[col] is not None:
                                row[col] = json.dumps(row[col], default=str)
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    count += len(rows)
                    preview.extend(rows[:max(0, 50 - len(preview))])
        return {**descriptor, **({'source_row_column': row_column, 'columns': schema.names} if include_row_ids else {}),
                'file': path.relative_to(self.root).as_posix(), 'revision': name,
                'row_count': count, 'preview': preview}

    def query(self, descriptor, *, page=0, page_size=100, sort_by=None, filter_tree=None, columns=None):
        import duckdb
        size = max(1, min(MAX_PAGE_SIZE, int(page_size)))
        page = max(0, int(page))
        all_columns = descriptor['columns']
        selected = all_columns if columns is None else list(columns)
        if not selected:
            raise ValueError('Select at least one column')
        projection = ', '.join(quote_column(c, all_columns) for c in selected)
        expressions = {c: quote_column(c, all_columns) for c in all_columns}
        for c in descriptor.get('encoded_columns', []):
            expressions[c] = f"json_extract_string({expressions[c]}, '$')"
        where, params = compile_filter(filter_tree, expressions)
        order = []
        for spec in sort_by or []:
            direction = spec.get('direction')
            if direction not in ('asc', 'desc'):
                raise ValueError('Invalid sort direction')
            order.append(quote_column(spec['column_id'], expressions) + ' ' + direction.upper() + ' NULLS LAST')
        # Parquet physical row number is stable within an immutable revision.
        internal = '__rk_physical_row__'
        while internal in all_columns:
            internal += '_'
        path = str(self.path(descriptor))
        if not filter_tree and not sort_by:
            count = descriptor['row_count']
            parquet = pq.ParquetFile(path)
            start, remaining, physical, raw = page * size, size, 0, []
            for group in range(parquet.num_row_groups):
                group_size = parquet.metadata.row_group(group).num_rows
                if physical + group_size <= start:
                    physical += group_size
                    continue
                for batch in parquet.iter_batches(row_groups=[group], columns=selected, batch_size=8192):
                    offset = max(0, start - physical)
                    if offset < len(batch):
                        records = batch.slice(offset, remaining).to_pylist()
                        raw.extend(tuple(r[c] for c in selected) + (physical + offset + i,) for i, r in enumerate(records))
                        remaining -= len(records)
                    physical += len(batch)
                    if not remaining:
                        break
                if not remaining:
                    break
        else:
            with duckdb.connect(config={'threads': 1, 'memory_limit': '256MB',
                                    'temp_directory': str(self.root / 'query-spill')}) as connection:
            # file_row_number conflicts with a user column of the same name; a
            # window row number avoids imposing reserved names on scientific data.
                source = f'(SELECT *, row_number() OVER () - 1 AS "{internal}" FROM read_parquet(?))'
                count = descriptor['row_count'] if not filter_tree else connection.execute(
                    f'SELECT count(*) FROM {source} WHERE {where}', [path] + params).fetchone()[0]
                order.append(f'"{internal}" ASC')
                cursor = connection.execute(f'SELECT {projection}, "{internal}" FROM {source} WHERE {where} '
                    f'ORDER BY {", ".join(order)} LIMIT ? OFFSET ?', [path] + params + [size, page * size])
                raw = cursor.fetchall()
        rows, row_ids = [], []
        for record in raw:
            rows.append({key: json_cell(json.loads(value) if key in descriptor.get('encoded_columns', []) and value is not None else value)
                         for key, value in zip(selected, record[:-1])})
            row_ids.append(f'{descriptor["revision"]}:{record[-1]}')
        response = {'rows': rows, 'row_ids': row_ids, 'row_count': count,
                    'page_count': math.ceil(count / size), 'columns': selected,
                    'revision': descriptor['revision'], 'page': page, 'page_size': size}
        if len(json.dumps(response, default=str).encode()) > MAX_PAGE_BYTES:
            raise ValueError('Page exceeds 2 MiB. Select fewer columns or reduce page size.')
        return response

    def export_csv(self, descriptor, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.' + uuid4().hex + '.tmp')
        try:
            with temporary.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=descriptor['columns'])
                writer.writeheader()
                writer.writerows(self.rows(descriptor))
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return str(target)
