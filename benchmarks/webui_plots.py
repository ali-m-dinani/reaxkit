"""Reproducible phase-4 CPU/payload audit. Does not measure browser/GPU FPS.

Run from the repository with its configured Python interpreter:
    benchmarks/webui_plots.py --output benchmarks/webui-plots-phase4.json
"""
import argparse
import gc
import importlib.metadata
import json
import platform
from pathlib import Path
import statistics
import tempfile
import threading
import time

import numpy as np
import psutil
import pyarrow as pa

from reaxkit.webui.backend.artifact_tables import ArtifactTables
from reaxkit.webui.backend.plot_queries import summarize_plot
from reaxkit.webui.presentation.plot_summary import summary_figure


def measure(fn, repeats):
    timings, peaks, value = [], [], None
    process = psutil.Process()
    for _ in range(repeats):
        gc.collect()
        rss = [process.memory_info().rss]
        stop = threading.Event()
        def sample():
            while not stop.wait(0.02):
                rss.append(process.memory_info().rss)
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        started = time.perf_counter()
        try:
            value = fn()
        finally:
            timings.append((time.perf_counter() - started) * 1000)
            rss.append(process.memory_info().rss)
            stop.set(); thread.join()
        peaks.append(max(rss))
    return value, dict(repetitions=repeats, milliseconds=timings, median_ms=statistics.median(timings),
                       sampled_process_peak_rss_bytes=max(peaks))


def measure_job_cache(root, payload, request, repeats):
    from reaxkit.webui.backend.pipeline_store import PipelineStore
    from reaxkit.webui.backend.jobs import JobService, TERMINAL
    from reaxkit.webui.backend.schemas import PipelineNode, ResultArtifact
    store = PipelineStore(table_root=root)
    jobs = JobService(store)
    pid = store.create_pipeline().id
    store.upsert_node(pid, PipelineNode(id='source', kind='analysis', name='Source', result_ref='result'))
    store.store_artifact(pid, ResultArtifact(id='result', node_id='source', payload=payload))
    arguments = {'artifact_id': 'result', 'request': request, 'width': 1000, 'view_id': 'benchmark'}
    def query(cold=False):
        if cold:
            jobs.query_cache.clear()
        result = jobs.submit(pid, 'plot', arguments)
        deadline = time.monotonic() + 120
        while result['state'] not in TERMINAL and time.monotonic() < deadline:
            time.sleep(0.02)
            result = jobs.get(pid, result['id'])
        if result['state'] != 'succeeded':
            raise RuntimeError(result.get('error') or result['state'])
        return len(json.dumps(result['result']).encode())
    try:
        _, cold = measure(lambda: query(True), repeats)
        byte_count, warm = measure(query, repeats)
        return dict(cold_spawn_and_render=cold, cached_submission_and_json=warm, response_bytes=byte_count)
    finally:
        jobs.close(); store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='benchmarks/webui-plots-phase4.json')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    results = {'machine': platform.platform(), 'processor': platform.processor(),
               'ram_bytes': psutil.virtual_memory().total,
               'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'pyarrow', 'duckdb', 'plotly', 'dash')},
               'fixture': 'seed=42; numeric step/sine with late spike and missing interval; normal xyz with stable IDs',
               'scope': 'Summary/serialization timings are in-process with first and warm OS-cache runs retained. job_path separately includes Windows spawn or cached submission plus JSON. No handler, HTTP, browser or GPU timing.',
               'results': []}
    for kind, counts in [('plot2d', [100_000, 1_000_000, 10_000_000]),
                         ('scatter3d', [10_000, 100_000, 1_000_000])]:
        for n in counts:
            with tempfile.TemporaryDirectory() as root:
                tables = ArtifactTables(root)
                if kind == 'plot2d':
                    x = np.arange(n)
                    y = np.sin(x / 100)
                    y[n-123] = 1000
                    y[n//2:n//2+100] = np.nan
                    table = pa.table({'x': x, 'y': y})
                else:
                    points = np.random.default_rng(42).normal(size=(n, 3))
                    table = pa.table({**{a: points[:, i] for i, a in enumerate('xyz')}, 'atom_id': np.arange(n)})
                payload = {'table': tables.write(table)}
                del table
                request = {'visualization_type': kind, 'x_col': 'x', 'y_col': 'y'}
                summary, timing = measure(lambda: summarize_plot(tables, payload, request), args.repeats)
                encoded, rendering = measure(lambda: summary_figure(summary, request, view_key='benchmark').to_json(), args.repeats)
                entry = dict(kind=kind, rows=n, summary=timing, figure_serialization=rendering,
                             displayed_points=summary['displayed_points'], json_bytes=len(encoded.encode()),
                             parquet_bytes=tables.path(payload['table']).stat().st_size)
                if kind == 'plot2d':
                    histogram, hist_time = measure(lambda: summarize_plot(tables, payload,
                        {'visualization_type': 'histogram', 'x_col': 'y'}), args.repeats)
                    entry['histogram'] = {**hist_time, 'bins': len(histogram['bins']), 'exact_count': histogram['valid_rows']}
                    zoom, zoom_time = measure(lambda: summarize_plot(tables, payload, request,
                        x_range=[n-200, n-100]), args.repeats)
                    entry['zoom'] = {**zoom_time, 'displayed_points': zoom['displayed_points'], 'reduced': zoom['reduced']}
                if n == 1_000_000:
                    entry['job_path'] = measure_job_cache(root, payload, request, args.repeats)
                results['results'].append(entry)
                print(f"{kind} {n:,}: summary {timing['median_ms']:.1f} ms, figure {rendering['median_ms']:.1f} ms, {entry['json_bytes']:,} bytes", flush=True)
                Path(args.output).write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
