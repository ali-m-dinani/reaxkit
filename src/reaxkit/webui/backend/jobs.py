"""Bounded local jobs. Workers never mutate the live pipeline store.

Only small immutable snapshots/descriptors cross the spawn boundary. Results are
published atomically by the owner after checking input revisions. Each job has an
isolated process so cancelling native/non-cooperative code is possible on Windows.
"""
from __future__ import annotations

import atexit
from collections import deque
from copy import deepcopy
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from threading import Condition, Thread
import time
import traceback
from uuid import uuid4

TERMINAL = {'succeeded', 'failed', 'cancelled', 'superseded', 'expired'}
QUERY_OPERATIONS = {'query', 'plot'}


def fingerprint(snapshot, node_id=None):
    # Only the selected node and its transitive input dependencies matter.
    selected = set(snapshot['nodes']) if node_id is None else set()
    pending = [node_id] if node_id else []
    while pending:
        key = pending.pop()
        if key in selected:
            continue
        selected.add(key)
        node = snapshot['nodes'].get(key, {})
        pending.extend(v for v in [node.get('parent_id'), (node.get('request') or {}).get('right_source_node_id')]
                       + list((node.get('metadata') or {}).get('source_node_ids', [])) if v)
    nodes = {key: {k: v for k, v in snapshot['nodes'].get(key, {}).items() if k not in ('status', 'updated_at')}
             for key in selected}
    for node in nodes.values():
        if node.get('kind') == 'dataset':
            node['metadata'] = deepcopy(node.get('metadata', {}))
            dataset = node['metadata'].get('dataset', {})
            for derived in ('frames', 'atoms'):
                dataset.pop(derived, None)
    return sha256(json.dumps(nodes, sort_keys=True, default=str).encode()).hexdigest()


def _worker(root, job_id, snapshot, operation, arguments):
    """Spawn entrypoint; errors are data, not an unobserved process exception."""
    from reaxkit.webui.backend.pipeline_store import PipelineStore
    from reaxkit.webui.backend.node_runtime import PipelineRuntime
    from threadpoolctl import threadpool_limits
    work = Path(root) / job_id
    work.mkdir(parents=True, exist_ok=True)
    last_report = [0.0, '']
    started = time.perf_counter()
    def reporter(stage, current=0, total=0, message=None):
        now = time.monotonic()
        if stage == last_report[1] and now - last_report[0] < 0.25 and (not total or current < total):
            return
        last_report[:] = [now, stage]
        counts = f' ({current:,}/{total:,})' if total else ''
        progress = {'stage': str(message or stage)[:180] + counts, 'current': current, 'total': total}
        try:
            temp = work / 'progress.tmp'
            temp.write_text(json.dumps(progress), encoding='utf-8')
            temp.replace(work / 'progress.json')
        except OSError:
            pass
    try:
        store = PipelineStore(table_root=root)
        store.tables.write_dir = work
        store.reporter = reporter
        store.load_snapshot(snapshot)
        runtime = PipelineRuntime(store)
        pid = snapshot['id']
        with threadpool_limits(limits=1):
            reporter(operation, message={'apply': 'Running analysis or utility', 'query': 'Querying complete result',
                'plot': 'Summarizing complete result', 'probe': 'Reading dataset metadata',
                'export_table': 'Exporting complete result'}.get(operation, operation))
            if operation == 'apply':
                result = runtime.apply_node(pid, arguments['node_id'])
            elif operation == 'load':
                result = runtime.load_dataset(pid, **arguments)
            elif operation == 'probe':
                node = store.get_node(pid, arguments['node_id'])
                meta = node.metadata['dataset']
                frames, atoms = runtime._probe_dataset_dimensions(run_dir=node.metadata['run_dir'],
                    engine_name=meta.get('engine_override') or meta.get('engine_detected'), sources=meta.get('sources', {}))
                meta.update(frames=frames, atoms=atoms)
                node.touch()
                result = {'frames': frames, 'atoms': atoms}
            elif operation == 'save_snapshot':
                from reaxkit.webui.backend.serializer import save_snapshot
                result = {'path': save_snapshot(snapshot, arguments['path'], tables=store.tables)}
            elif operation == 'import_snapshot':
                from reaxkit.webui.backend.serializer import load_snapshot
                imported = load_snapshot(arguments['path'], tables=store.tables)
                imported['id'] = pid
                store.load_snapshot(imported)
                result = {'path': arguments['path']}
            elif operation == 'export_bundle':
                from reaxkit.webui.backend.serializer import export_bundle
                selected = snapshot['nodes'].get(arguments.get('selected_node_id'), {})
                artifact = snapshot['artifacts'].get(selected.get('result_ref'))
                result = export_bundle(snapshot=snapshot, output_dir=arguments['path'],
                    selected_node_id=selected.get('id'), selected_artifact=artifact, tables=store.tables)
            elif operation == 'plot':
                from reaxkit.webui.backend.plot_queries import summarize_plot
                from reaxkit.webui.presentation.plot_summary import summary_figure, summary_notice
                from reaxkit.webui.presentation.perf_config import load_ui_performance_config
                artifact = store.get_artifact(pid, arguments['artifact_id'])
                summary = summarize_plot(store.tables, artifact.payload, arguments['request'],
                    width=arguments.get('width', 1000), x_range=arguments.get('x_range'), frame=arguments.get('frame'))
                reporter('plot', message='Preparing bounded plot geometry')
                encoded = summary_figure(summary, arguments['request'], view_key='').to_json()
                if len(encoded.encode()) > load_ui_performance_config()['plot_max_bytes']:
                    raise ValueError('Figure exceeds the payload budget. Reduce visible curves or use the table.')
                result = {'figure': json.loads(encoded), 'status': summary_notice(summary),
                          'summary': {k: v for k, v in summary.items() if k not in ('points', 'traces', 'bins', 'cell', 'bonds')}}
            elif operation == 'query':
                artifact = store.get_artifact(pid, arguments['artifact_id'])
                from reaxkit.webui.backend.artifact_tables import table_descriptor
                descriptor = store.tables.filtered(table_descriptor(artifact.payload), arguments.get('row_filters'))
                result = store.tables.query(descriptor, **arguments['query'])
            elif operation == 'export_table':
                from reaxkit.webui.backend.artifact_tables import table_descriptor
                descriptor = table_descriptor(store.get_artifact(pid, arguments['artifact_id']).payload)
                descriptor = store.tables.filtered(descriptor, arguments.get('row_filters'))
                target = Path(arguments['path'])
                target.parent.mkdir(parents=True, exist_ok=True)
                staged = target.with_name(target.name + '.' + job_id + '.tmp')
                if str(arguments['path']).lower().endswith('.xlsx'):
                    from openpyxl import Workbook
                    workbook = Workbook(write_only=True)
                    sheet = workbook.create_sheet('Results')
                    sheet.append(descriptor['columns'])
                    for row in store.tables.rows(descriptor):
                        sheet.append([row.get(c) for c in descriptor['columns']])
                    workbook.save(staged)
                    result = {'path': str(target)}
                else:
                    import csv
                    with staged.open('w', encoding='utf-8', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=descriptor['columns'])
                        writer.writeheader()
                        writer.writerows(store.tables.rows(descriptor))
                    result = {'path': str(target)}
            else:
                raise ValueError(f'Unknown job operation: {operation}')
        after = store.snapshot(pid)
        changed = {key: node for key, node in after['nodes'].items() if snapshot['nodes'].get(key) != node}
        response = {'result': result, 'nodes': changed,
                    'artifacts': {k: a for k, a in after['artifacts'].items() if k not in snapshot['artifacts']},
                    'worker_duration_ms': (time.perf_counter() - started) * 1000}
        if operation == 'import_snapshot':
            response['snapshot'] = after
        target = work / 'response.json'
        temp = work / 'response.tmp'
        temp.write_text(json.dumps(response, default=str), encoding='utf-8')
        temp.replace(target)
    except BaseException:
        (work / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')


class JobService:
    def __init__(self, store, *, max_pending=8):
        self.store = store
        self.max_pending = max_pending
        self.jobs = {}
        self.queue = deque()
        self.query_queue = deque()
        self.condition = Condition()
        self.closed = False
        self.thread = None
        self.query_thread = None
        from reaxkit.webui.backend.byte_cache import CacheBudget
        self.query_cache = CacheBudget(16 * 1024 * 1024).namespace('query')
        atexit.register(self.close)

    def submit(self, pipeline_id, operation, arguments):
        with self.condition, self.store._lock:
            if self.closed:
                raise RuntimeError('Job service is closed')
            if operation not in {'apply', 'probe', 'load', 'query', 'plot', 'export_table', 'save_snapshot', 'import_snapshot', 'export_bundle'}:
                raise ValueError('Unknown job operation')
            if operation in ('query', 'plot', 'export_table'):
                self.store.get_artifact(pipeline_id, arguments['artifact_id'])
            if operation in ('apply', 'probe'):
                self.store.get_node(pipeline_id, arguments['node_id'])
            if operation == 'query' and not isinstance(arguments.get('query'), dict):
                raise ValueError('Query must be an object')
            if operation == 'plot' and not isinstance(arguments.get('request'), dict):
                raise ValueError('Plot request must be an object')
            completed = [key for key, value in self.jobs.items() if value['state'] in TERMINAL]
            for key in completed[:-99]:
                del self.jobs[key]
            snapshot = self.store.snapshot(pipeline_id)
            revision = fingerprint(snapshot, arguments.get('node_id'))
            identity = (pipeline_id, operation, json.dumps(arguments, sort_keys=True), revision)
            if operation == 'plot':
                from reaxkit.webui.backend.artifact_tables import table_descriptor
                from reaxkit.webui.backend.plot_queries import RENDERER_VERSION
                descriptor = table_descriptor(self.store.get_artifact(pipeline_id, arguments['artifact_id']).payload)
                if descriptor is None:
                    raise ValueError('No tabular result to plot')
                identity = (pipeline_id, operation, descriptor['revision'], RENDERER_VERSION,
                            json.dumps({k: v for k, v in arguments.items() if k != 'view_id'}, sort_keys=True))
                # One latest viewport request per browser view. Cancel queued and
                # running predecessors before accepting its replacement.
                for old in self.jobs.values():
                    if (old['pipeline_id'] == pipeline_id and old['operation'] == 'plot'
                            and old['arguments'].get('view_id') == arguments.get('view_id')
                            and old['identity'] != identity and old['state'] not in TERMINAL):
                        old.update(state='cancelling', stage='Replacing obsolete viewport request')
            for job in self.jobs.values():
                if (job['identity'] == identity and job['state'] not in TERMINAL | {'cancelling'}
                        and (operation != 'plot' or job['arguments'].get('view_id') == arguments.get('view_id'))):
                    return self._public(job)
            if sum(j['state'] not in TERMINAL for j in self.jobs.values()) >= self.max_pending:
                raise ValueError('Job queue is full. Cancel or wait for an active job.')
            job_id = 'job_' + uuid4().hex
            job = {'id': job_id, 'pipeline_id': pipeline_id, 'operation': operation,
                   'arguments': deepcopy(arguments), 'snapshot': snapshot, 'revision': revision,
                   'identity': identity, 'state': 'queued', 'stage': 'Waiting for worker',
                   'created': time.time(), 'result': None, 'error': None}
            self.jobs[job_id] = job
            cached = self.query_cache.get(identity) if operation in QUERY_OPERATIONS else None
            if cached is not None:
                self._finish(job, 'succeeded', result=cached)
                job.pop('snapshot', None)
                return self._public(job)
            if operation == 'query' and not arguments.get('row_filters') and not arguments['query'].get('sort_by') and not arguments['query'].get('filter_tree'):
                from reaxkit.webui.backend.artifact_tables import table_descriptor
                descriptor = table_descriptor(self.store.get_artifact(pipeline_id, arguments['artifact_id']).payload)
                try:
                    result = self.store.tables.query(descriptor, **arguments['query'])
                    self.query_cache[identity] = result
                    self._finish(job, 'succeeded', result=result)
                except Exception as exc:
                    self._finish(job, 'failed', error=str(exc))
                job.pop('snapshot', None)
                return self._public(job)
            queue = self.query_queue if operation in QUERY_OPERATIONS else self.queue
            queue.append(job_id)
            if operation == 'apply':
                self.store.update_node(pipeline_id, arguments['node_id'], status='running', propagate_dirty=False)
            # Retain bounded terminal history without retaining worker snapshots.
            completed = [k for k, v in self.jobs.items() if v['state'] in TERMINAL]
            for key in completed[:-100]:
                del self.jobs[key]
            attribute = 'query_thread' if operation in QUERY_OPERATIONS else 'thread'
            if getattr(self, attribute) is None:
                thread = Thread(target=self._run, args=(queue,), name='reaxkit-' + attribute, daemon=True)
                setattr(self, attribute, thread)
                thread.start()
            self.condition.notify_all()
            return self._public(job)

    def _public(self, job, include_result=True):
        public = {key: deepcopy(job.get(key)) for key in
                ('id', 'pipeline_id', 'operation', 'state', 'stage', 'created', 'finished', 'error', 'peak_rss_bytes', 'worker_duration_ms')}
        public['node_id'] = job['arguments'].get('node_id')
        public['result'] = deepcopy(job.get('result')) if include_result else None
        if job['operation'] in QUERY_OPERATIONS and job['state'] == 'succeeded':
            result = self.query_cache.get(job['identity'])
            if result is None:
                public.update(state='expired', stage='Query expired; request the page again')
            public['result'] = deepcopy(result) if include_result else None
        return public

    def get(self, pipeline_id, job_id):
        with self.condition:
            job = self.jobs[job_id]
            if job['pipeline_id'] != pipeline_id:
                raise KeyError('Job does not belong to this pipeline')
            return self._public(job)

    def list(self, pipeline_id):
        with self.condition:
            return [self._public(j, include_result=False) for j in self.jobs.values() if j['pipeline_id'] == pipeline_id]

    def cancel(self, pipeline_id, job_id):
        with self.condition:
            job = self.jobs[job_id]
            if job['pipeline_id'] != pipeline_id:
                raise KeyError('Job does not belong to this pipeline')
            if job['state'] not in TERMINAL:
                job['state'] = 'cancelling'
                job['stage'] = 'Stopping worker'
            self.condition.notify_all()
            return self._public(job)

    def cancel_view(self, pipeline_id, view_id):
        with self.condition:
            for job in self.jobs.values():
                if (job['pipeline_id'] == pipeline_id and job['operation'] == 'plot'
                        and job['arguments'].get('view_id') == view_id and job['state'] not in TERMINAL):
                    job.update(state='cancelling', stage='Closing previous view')
            self.condition.notify_all()

    def cancel_deleted(self, pipeline_id, node_ids, artifact_ids):
        """Stop work for a removed subtree; revision checks still guard publication."""
        nodes, artifacts = set(node_ids), set(artifact_ids)
        with self.condition:
            for job in self.jobs.values():
                args = job['arguments']
                if (job['pipeline_id'] == pipeline_id and job['state'] not in TERMINAL and
                        (args.get('node_id') in nodes or args.get('artifact_id') in artifacts)):
                    job.update(state='cancelling', stage='Source deleted; stopping worker')
            self.condition.notify_all()

    def _run(self, queue):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or queue)
                if self.closed and not queue:
                    return
                job = self.jobs[queue.popleft()]
                if job['state'] == 'cancelling' or self.closed:
                    self._finish(job, 'cancelled')
                    job.pop('snapshot', None)
                    continue
                job['state'], job['stage'] = 'running', 'Computing in isolated worker'
            process = mp.get_context('spawn').Process(target=_worker, args=(str(self.store.tables.root),
                job['id'], job['snapshot'], job['operation'], job['arguments']))
            work = self.store.tables.root / job['id']
            try:
                process.start()
                import psutil
                memory_limit = int(os.environ.get('REAXKIT_UI_JOB_MEMORY_MB', '4096')) * 1024 * 1024
                while process.is_alive():
                    try:
                        progress = json.loads((work / 'progress.json').read_text(encoding='utf-8'))
                        if job['state'] == 'running':
                            job['stage'] = progress['stage']
                    except (OSError, ValueError):
                        pass
                    try:
                        rss = psutil.Process(process.pid).memory_info().rss
                        job['peak_rss_bytes'] = max(job.get('peak_rss_bytes', 0), rss)
                        if rss > memory_limit:
                            job['resource_error'] = 'Job exceeded its memory budget. Reduce the input range or increase REAXKIT_UI_JOB_MEMORY_MB.'
                    except psutil.NoSuchProcess:
                        pass
                    with self.condition:
                        stop = self.closed or job['state'] == 'cancelling' or job.get('resource_error')
                    if stop:
                        import psutil
                        try:
                            children = psutil.Process(process.pid).children(recursive=True)
                            for child in children:
                                try:
                                    child.kill()
                                except psutil.NoSuchProcess:
                                    pass
                        except psutil.NoSuchProcess:
                            pass
                        process.terminate()
                    process.join(timeout=0.1)
                with self.condition, self.store._lock:
                    if job.get('resource_error'):
                        self._finish(job, 'failed', error=job['resource_error'])
                    elif self.closed or job['state'] == 'cancelling':
                        self._finish(job, 'cancelled')
                    elif (work / 'error.txt').exists():
                        self._finish(job, 'failed', error=(work / 'error.txt').read_text(encoding='utf-8'))
                    elif process.exitcode != 0 or not (work / 'response.json').exists():
                        self._finish(job, 'failed', error=f'Worker exited with code {process.exitcode}')
                    else:
                        response = json.loads((work / 'response.json').read_text(encoding='utf-8'))
                        job['worker_duration_ms'] = response.get('worker_duration_ms')
                        current = self.store.metadata_snapshot(job['pipeline_id'])
                        if job['operation'] in ('apply', 'load', 'probe', 'import_snapshot') and fingerprint(current, job['arguments'].get('node_id')) != job['revision']:
                            self._finish(job, 'superseded')
                        else:
                            self._commit(job, response)
                            if job['operation'] in QUERY_OPERATIONS:
                                self.query_cache[job['identity']] = response['result']
                            self._finish(job, 'succeeded', result=response['result'])
            except BaseException:
                with self.condition:
                    self._finish(job, 'failed', error=traceback.format_exc())
            finally:
                if process.pid is not None:
                    if process.is_alive():
                        process.terminate()
                    process.join()
                    process.close()
                job.pop('snapshot', None)
                if job['operation'] == 'export_table':
                    target = Path(job['arguments']['path'])
                    try:
                        target.with_name(target.name + '.' + job['id'] + '.tmp').unlink(missing_ok=True)
                    except OSError:
                        pass
                if job['state'] != 'succeeded' or job['operation'] not in ('apply', 'load', 'import_snapshot'):
                    shutil.rmtree(work, ignore_errors=True)
                try:
                    self.collect()
                except OSError:
                    # A temporarily locked file must not stop the job dispatcher.
                    pass

    def collect(self):
        with self.condition, self.store._lock:
            active = [j for j in self.jobs.values() if j['state'] not in TERMINAL]
            self.store.expire_idle(protected={j['pipeline_id'] for j in active})
            protected = [j['snapshot'] for j in active if 'snapshot' in j]
            keep = self.store.prune_artifacts(protected)
            active_dirs = {j['id'] for j in active}
            for path in self.store.tables.root.rglob('*.parquet'):
                relative = path.relative_to(self.store.tables.root)
                if relative.as_posix() not in keep and relative.parts[0] not in active_dirs:
                    path.unlink(missing_ok=True)

    def _commit(self, job, response):
        from reaxkit.webui.backend.schemas import PipelineNode, ResultArtifact
        pid = job['pipeline_id']
        if job['operation'] == 'export_table':
            target = Path(job['arguments']['path'])
            target.with_name(target.name + '.' + job['id'] + '.tmp').replace(target)
            return
        if job['operation'] == 'import_snapshot':
            self.store.load_snapshot(response['snapshot'])
            return
        for artifact in response['artifacts'].values():
            self.store.store_artifact(pid, ResultArtifact(**artifact))
        for node in response['nodes'].values():
            self.store.upsert_node(pid, PipelineNode(**node))
        result = response.get('result') or {}
        node = result.get('node') or {}
        if job['operation'] == 'apply' and node.get('kind') == 'analysis':
            pipeline = self.store.get_pipeline(pid)
            children = pipeline.children.get(node['id'], [])
            if not any(pipeline.nodes[c].kind == 'visualization' for c in children):
                from reaxkit.presentation.specs import ensure_presentation_spec, spec_to_dash_request
                from reaxkit.webui.backend.node_runtime import PipelineRuntime
                for rec in (result.get('artifact') or {}).get('recommended_views', []):
                    spec = ensure_presentation_spec(rec)
                    request = spec_to_dash_request(spec or rec)
                    vtype = request.get('visualization_type', 'plot2d')
                    PipelineRuntime(self.store).add_node(pid, parent_id=node['id'], kind='visualization',
                        name=(spec.label if spec else rec.get('label')) or vtype, request=request,
                        metadata={'visualization_type': vtype, 'auto_recommended': True, 'presentation_spec': rec})

    def _finish(self, job, state, *, result=None, error=None):
        if job['operation'] == 'apply' and state in ('cancelled', 'failed', 'superseded'):
            with self.store._lock:
                node_id = job['arguments']['node_id']
                try:
                    node = self.store.get_node(job['pipeline_id'], node_id)
                    if node.status == 'running':
                        old_status = job.get('snapshot', {}).get('nodes', {}).get(node_id, {}).get('status', 'idle')
                        self.store.update_node(job['pipeline_id'], node_id,
                            status='error' if state == 'failed' else ('dirty' if state == 'superseded' else old_status),
                            propagate_dirty=False)
                except KeyError:
                    pass
        if job['operation'] in QUERY_OPERATIONS and state == 'succeeded':
            self.query_cache[job['identity']] = result
            result = None
        job.update(state=state, stage=state.title(), result=result, error=error, finished=time.time())

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=10)
        if self.query_thread:
            self.query_thread.join(timeout=10)
        atexit.unregister(self.close)
