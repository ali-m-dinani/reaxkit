"""Reproducible GUI data-path benchmark (no scientific calculation included).

Run from an installed source checkout: python benchmarks/webui_pipeline.py --output report.json
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psutil

from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.backend.adapters.result_normalizer import normalize_result
from reaxkit.webui.backend.schemas import PipelineNode, ResultArtifact
from reaxkit.webui.ui.analysis import callback_helpers as helpers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=100_000)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frame = pd.DataFrame({'step': np.arange(args.rows), 'value': np.sin(np.arange(args.rows) / 100),
                          'atom_id': np.arange(args.rows) % 1000})
    timings = {}
    def measure(name, function):
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = function()
            samples.append((time.perf_counter() - start) * 1000)
        timings[name] = {'median_ms': statistics.median(samples), 'samples_ms': samples}
        return result
    service = WebUIApiService()
    pipeline = service.create_pipeline()
    pid = pipeline['id']
    service.store.upsert_node(pid, PipelineNode(id='benchmark', kind='analysis', name='benchmark'))
    result = SimpleNamespace(table=frame)
    if hasattr(service.store, 'tables'):
        payload = measure('normalize_and_persist', lambda: normalize_result(result, tables=service.store.tables))
    else:
        payload = measure('normalize_and_persist', lambda: normalize_result(result))
    artifact = ResultArtifact(id='benchmark-artifact', node_id='benchmark', payload=payload)
    service.store.store_artifact(pid, artifact)
    snapshot = measure('metadata_snapshot', lambda: service.get_pipeline(pid))
    if hasattr(service, 'query_table'):
        page = measure('table_first_page', lambda: service.query_table(pid, artifact.id, page_size=100))
        page_bytes = len(json.dumps(page, default=str).encode())
        measure('table_sorted_page', lambda: service.query_table(pid, artifact.id, page_size=100,
                    sort_by=[{'column_id': 'value', 'direction': 'desc'}]))
    else:
        table = measure('table_first_page', lambda: helpers._build_result_table(helpers._artifact_rows(artifact.__dict__), max_rows=100))
        page_bytes = len(json.dumps(table.to_plotly_json(), default=str).encode())
    report = {'environment': {'os': platform.platform(), 'python': platform.python_version(),
              'cpu': platform.processor(), 'logical_cpus': psutil.cpu_count(),
              'ram_bytes': psutil.virtual_memory().total,
              'versions': {p: importlib.metadata.version(p) for p in ['dash', 'plotly', 'pandas', 'pyarrow']}},
              'fixture': {'rows': args.rows, 'columns': list(frame.columns), 'seed': 'deterministic-sine',
                          'repeats': args.repeats, 'cache': 'first sample cold; remaining warm'},
              'timings': timings, 'metadata_bytes': len(json.dumps(snapshot).encode()),
              'page_bytes': page_bytes, 'process_rss_bytes': psutil.Process().memory_info().rss,
              'scope': 'Python GUI data path only; excludes browser, disk discovery, and scientific compute'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if hasattr(service, 'close'):
        service.close()


if __name__ == '__main__':
    main()
