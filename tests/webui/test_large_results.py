from dataclasses import dataclass
import json
import time

import pandas as pd
import pytest

from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.backend.artifact_tables import ArtifactTables, compile_filter, table_descriptor
from reaxkit.webui.backend.adapters.result_normalizer import normalize_result
from reaxkit.webui.backend.byte_cache import CacheBudget
from reaxkit.webui.backend.jobs import TERMINAL
from reaxkit.webui.backend.schemas import PipelineNode, ResultArtifact
from reaxkit.webui.backend.serializer import save_snapshot, load_snapshot, export_bundle


def delayed_worker(root, job_id, snapshot, operation, arguments):
    from reaxkit.webui.backend.jobs import _worker
    if operation == 'apply':
        time.sleep(3)
    _worker(root, job_id, snapshot, operation, arguments)


def analysis_worker(root, job_id, snapshot, operation, arguments):
    from types import SimpleNamespace
    from reaxkit.webui.backend import node_runtime
    from reaxkit.webui.backend.jobs import _worker
    def run_analysis(name, request, runtime_args):
        assert callable(runtime_args['reporter'])
        runtime_args['reporter']('analysis', 1, 1, 'Synthetic analysis complete')
        return SimpleNamespace(table=pd.DataFrame({'step': range(20_000), 'value': range(20_000)})), type('Task', (), {})
    node_runtime.run_analysis_task = run_analysis
    _worker(root, job_id, snapshot, operation, arguments)


@pytest.fixture
def service():
    service = WebUIApiService()
    yield service
    service.close()


def fixture_pipeline(service, rows=100):
    pid = service.create_pipeline()['id']
    service.store.upsert_node(pid, PipelineNode(id='source', kind='analysis', name='source', result_ref='data'))
    artifact = ResultArtifact(id='data', node_id='source', payload={'table': pd.DataFrame({
        'step': range(rows), 'value': [i % 7 for i in range(rows)], 'text': ['alpha', 'beta'] * (rows // 2)})})
    service.store.store_artifact(pid, artifact)
    return pid, artifact


def wait_job(service, pid, job, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = service.jobs.get(pid, job['id'])
        if value['state'] in TERMINAL:
            return value
        time.sleep(0.05)
    pytest.fail('Job did not finish')


def test_metadata_does_not_traverse_payload(service):
    pid, artifact = fixture_pipeline(service)
    class Poison(dict):
        def __deepcopy__(self, memo):
            raise AssertionError('metadata traversed result payload')
        def items(self):
            raise AssertionError('metadata iterated result payload')
    artifact.payload = Poison()
    snapshot = service.get_pipeline(pid)
    assert 'payload' not in snapshot['artifacts']['data']
    assert len(json.dumps(snapshot)) < 2000


def test_dataclass_dataframe_is_persisted_without_records(service):
    @dataclass
    class Result:
        table: pd.DataFrame
    frame = pd.DataFrame({'x': range(2000)})
    payload = normalize_result(Result(frame), tables=service.store.tables)
    descriptor = table_descriptor(payload)
    assert descriptor['row_count'] == 2000
    assert len(descriptor['preview']) <= 50
    assert list(service.store.tables.rows(descriptor))[-1] == {'x': 1999}


def test_whole_result_filter_sort_page_and_stable_ids(service):
    pid, artifact = fixture_pipeline(service, 20_000)
    tree = {'type': 'relational-operator', 'subType': '>=',
            'left': {'type': 'expression', 'subType': 'field', 'value': 'step'},
            'right': {'type': 'expression', 'subType': 'value', 'value': 19000}}
    result = service.query_table(pid, artifact.id, page=2, page_size=17, filter_tree=tree,
        sort_by=[{'column_id': 'value', 'direction': 'desc'}], columns=['step', 'value'])
    expected = pd.DataFrame({'step': range(20_000), 'value': [i % 7 for i in range(20_000)]})
    expected = expected[expected.step >= 19000].sort_values(['value', 'step'], ascending=[False, True])
    assert result['rows'] == expected.iloc[34:51].to_dict('records')
    assert result['row_count'] == 1000
    assert len(set(result['row_ids'])) == 17
    assert len(json.dumps(result)) < 5000


def test_filter_rejects_sql_and_unknown_columns():
    with pytest.raises(ValueError):
        compile_filter({'type': 'relational-operator', 'subType': '=',
            'left': {'type': 'expression', 'subType': 'field', 'value': 'x; DROP TABLE anything'},
            'right': {'type': 'expression', 'subType': 'value', 'value': 1}}, ['x'])


def test_snapshot_and_export_are_portable_and_complete(service, tmp_path):
    pid, artifact = fixture_pipeline(service, 20_000)
    path = tmp_path / 'pipeline.json'
    save_snapshot(service.store.snapshot(pid), str(path), tables=service.store.tables)
    tables = ArtifactTables(tmp_path / 'import')
    snapshot = load_snapshot(str(path), tables=tables)
    descriptor = table_descriptor(snapshot['artifacts']['data']['payload'])
    assert sum(len(b) for b in tables.iter_batches(descriptor)) == 20_000
    export_bundle(snapshot=service.store.snapshot(pid), output_dir=str(tmp_path / 'bundle'),
        selected_artifact=artifact.__dict__, tables=service.store.tables)
    exported = pd.read_csv(tmp_path / 'bundle' / 'selected_result.csv')
    assert len(exported) == 20_000
    assert exported.iloc[-1].step == 19999


def test_legacy_snapshot_import(service):
    snapshot = {'id': 'old', 'nodes': {}, 'artifacts': {'a': {'id': 'a', 'node_id': 'x',
                'payload': {'table': [{'x': 1}, {'x': None}]}}}}
    service.store.load_snapshot(snapshot)
    assert service.query_table('old', 'a')['rows'] == [{'x': 1}, {'x': None}]


def test_byte_budget_shared_between_caches():
    budget = CacheBudget(800)
    first, second = budget.namespace('rows'), budget.namespace('figures')
    first['a'] = 'x' * 500
    second['b'] = 'y' * 500
    assert 'a' not in first and 'b' in second
    second['huge'] = 'z' * 900
    assert 'huge' not in second and budget.bytes <= 800


def test_worker_query_and_pipeline_isolation(service):
    pid, artifact = fixture_pipeline(service, 1000)
    job = service.jobs.submit(pid, 'query', {'artifact_id': artifact.id, 'query': {'page': 3, 'page_size': 10}})
    with pytest.raises(KeyError):
        service.jobs.get('other-pipeline', job['id'])
    result = wait_job(service, pid, job)
    assert result['state'] == 'succeeded', result.get('error')
    assert result['result']['rows'][0]['step'] == 30


def test_worker_visualization_commit_and_stale_rejection(service):
    pid, _ = fixture_pipeline(service, 1000)
    service.store.upsert_node(pid, PipelineNode(id='view', kind='visualization', name='view', parent_id='source'))
    job = service.submit_node(pid, 'view')
    service.update_node(pid, 'view', {'request': {'visualization_type': 'histogram'}})
    result = wait_job(service, pid, job)
    assert result['state'] == 'superseded', result.get('error')
    assert service.store.get_node(pid, 'view').result_ref is None
    result = wait_job(service, pid, service.submit_node(pid, 'view'))
    assert result['state'] == 'succeeded', result.get('error')
    assert service.store.get_node(pid, 'view').result_ref


def test_cancel_deduplicate_and_keep_previous_result(service):
    pid, _ = fixture_pipeline(service)
    service.store.upsert_node(pid, PipelineNode(id='view', kind='visualization', name='view', parent_id='source', result_ref='data'))
    job = service.submit_node(pid, 'view')
    assert service.submit_node(pid, 'view')['id'] == job['id']
    start = time.perf_counter()
    service.jobs.cancel(pid, job['id'])
    assert time.perf_counter() - start < 0.25
    result = wait_job(service, pid, job)
    assert result['state'] == 'cancelled'
    assert service.store.get_node(pid, 'view').result_ref == 'data'


def test_worker_failure_returns_error(service):
    pid, _ = fixture_pipeline(service)
    service.store.upsert_node(pid, PipelineNode(id='bad', kind='utility', name='bad', parent_id='missing'))
    result = wait_job(service, pid, service.submit_node(pid, 'bad'))
    assert result['state'] == 'failed'
    assert 'missing' in result['error']


def test_mixed_cells_roundtrip_and_filter(service):
    pid = service.create_pipeline()['id']
    service.store.store_artifact(pid, ResultArtifact(id='mixed', node_id='source',
        payload={'table': [{'mixed': 1}, {'mixed': 'alpha'}, {'mixed': None}]}))
    artifact = service.store.get_artifact(pid, 'mixed')
    descriptor = table_descriptor(artifact.payload)
    assert list(service.store.tables.rows(descriptor)) == [{'mixed': 1}, {'mixed': 'alpha'}, {'mixed': None}]
    assert service.query_table(pid, 'mixed')['rows'][0]['mixed'] == 1
    tree = {'type': 'relational-operator', 'subType': 'contains',
            'left': {'type': 'expression', 'subType': 'field', 'value': 'mixed'},
            'right': {'type': 'expression', 'subType': 'value', 'value': 'alph'}}
    assert service.query_table(pid, 'mixed', filter_tree=tree)['rows'] == [{'mixed': 'alpha'}]


def test_worker_utility_uses_all_rows_and_releases_old_artifact(service):
    pid, artifact = fixture_pipeline(service, 20_000)
    service.store.upsert_node(pid, PipelineNode(id='transform', kind='utility', name='column_transform',
        parent_id='source', request={'source': 'step', 'new_column': 'scaled', 'scale': 2, 'offset': 1}))
    result = wait_job(service, pid, service.submit_node(pid, 'transform'))
    assert result['state'] == 'succeeded', result.get('error')
    aid = service.store.get_node(pid, 'transform').result_ref
    page = service.query_table(pid, aid, page=199, page_size=100)
    assert page['row_count'] == 20_000
    assert page['rows'][-1]['scaled'] == 39999
    assert service.store.get_node(pid, 'source').result_ref == aid
    service.jobs.collect()
    assert 'data' not in service.store.get_pipeline(pid).artifacts


def test_worker_filters_and_export_do_not_use_preview(service, tmp_path):
    pid, artifact = fixture_pipeline(service, 20_000)
    filters = [{'column': 'step', 'op': '>=', 'value': '19000'}]
    query = service.jobs.submit(pid, 'query', {'artifact_id': artifact.id, 'row_filters': filters,
        'query': {'page_size': 10}})
    result = wait_job(service, pid, query)
    assert result['state'] == 'succeeded', result.get('error')
    assert result['result']['row_count'] == 1000
    assert result['result']['rows'][0]['step'] == 19000
    path = tmp_path / 'full.csv'
    job = service.jobs.submit(pid, 'export_table', {'artifact_id': artifact.id, 'row_filters': filters, 'path': str(path)})
    result = wait_job(service, pid, job)
    assert result['state'] == 'succeeded', result.get('error')
    frame = pd.read_csv(path)
    assert len(frame) == 1000 and frame.iloc[-1].step == 19999


def test_async_snapshot_import(service, tmp_path):
    pid, _ = fixture_pipeline(service)
    path = tmp_path / 'snapshot.json'
    result = wait_job(service, pid, service.jobs.submit(pid, 'save_snapshot', {'path': str(path)}))
    assert result['state'] == 'succeeded', result.get('error')
    new_pid = service.create_pipeline()['id']
    result = wait_job(service, new_pid, service.jobs.submit(new_pid, 'import_snapshot', {'path': str(path)}))
    assert result['state'] == 'succeeded', result.get('error')
    assert service.query_table(new_pid, 'data')['row_count'] == 100


def test_dash_table_callback_roundtrip_is_bounded():
    from reaxkit.webui.dash_app import create_dash_app
    app = create_dash_app()
    service = app.reaxkit_service
    try:
        pid, _ = fixture_pipeline(service, 100_000)
        service.store.upsert_node(pid, PipelineNode(id='table', kind='visualization', name='Table',
            parent_id='source', request={'visualization_type': 'table'}))
        with app.server.test_client() as client:
            assert client.get('/').status_code == 200
            assert client.get('/_dash-layout').status_code == 200
            response = client.post('/_dash-update-component', json={
                'output': 'canvas-content.children', 'outputs': {'id': 'canvas-content', 'property': 'children'},
                'inputs': [{'id': 'session-store', 'property': 'data', 'value': {'pipeline_id': pid, 'selected_node_id': 'table'}},
                           {'id': 'result-store', 'property': 'data', 'value': {}},
                           {'id': 'pipeline-store', 'property': 'data', 'value': service.get_pipeline(pid)}],
                'state': [{'id': 'plot-context', 'property': 'data', 'value': None}],
                'changedPropIds': ['session-store.data']})
            assert response.status_code == 200, response.get_data(as_text=True)
            body = response.get_data(as_text=True)
            assert '100,000 rows' in body and 'page_action' in body
            assert len(body) < 30_000
            assert 'Server-Timing' in response.headers
    finally:
        service.close()


def test_repeat_jobs_do_not_retain_artifact_files(service):
    pid, _ = fixture_pipeline(service)
    service.store.upsert_node(pid, PipelineNode(id='view', kind='visualization', name='view', parent_id='source'))
    for _ in range(20):
        result = wait_job(service, pid, service.submit_node(pid, 'view'))
        assert result['state'] == 'succeeded', result.get('error')
    service.jobs.collect()
    assert len(service.store.get_pipeline(pid).artifacts) == 2
    assert len(list(service.store.tables.root.rglob('*.parquet'))) == 1


def test_probe_never_loads_trajectory(monkeypatch):
    from reaxkit.webui.backend.node_runtime import PipelineRuntime
    from reaxkit.core.platform import engine_resolver
    class Adapter:
        def quick_n_frames(self, _args):
            return None
        def load(self, *args, **kwargs):
            pytest.fail('Metadata probing attempted a full trajectory load')
    monkeypatch.setattr(engine_resolver, 'resolve_engine', lambda *a, **kw: Adapter())
    assert PipelineRuntime._probe_dataset_dimensions(run_dir='.', engine_name='reaxff', sources={}) == (None, None)


def test_unrelated_node_changes_do_not_discard_job(service):
    pid, _ = fixture_pipeline(service)
    service.store.upsert_node(pid, PipelineNode(id='view', kind='visualization', name='view', parent_id='source'))
    job = service.submit_node(pid, 'view')
    service.store.upsert_node(pid, PipelineNode(id='unrelated', kind='visualization', name='other', parent_id='source'))
    assert wait_job(service, pid, job)['state'] == 'succeeded'


def test_queue_bound_and_shutdown_cleanup(service):
    pid, _ = fixture_pipeline(service)
    service.jobs.max_pending = 1
    service.store.upsert_node(pid, PipelineNode(id='view', kind='visualization', name='view', parent_id='source'))
    service.submit_node(pid, 'view')
    with pytest.raises(ValueError, match='queue is full'):
        service.jobs.submit(pid, 'query', {'artifact_id': 'data', 'query': {'sort_by': [{'column_id': 'value', 'direction': 'asc'}]}})
    service.jobs.close()
    assert not service.jobs.thread.is_alive()


def test_snapshot_path_traversal_is_rejected(service, tmp_path):
    pid, _ = fixture_pipeline(service)
    snapshot = service.store.snapshot(pid)
    descriptor = table_descriptor(snapshot['artifacts']['data']['payload'])
    descriptor['file'] = '../private.parquet'
    path = tmp_path / 'unsafe.json'
    path.write_text(json.dumps(snapshot), encoding='utf-8')
    with pytest.raises(ValueError, match='invalid table reference'):
        load_snapshot(str(path), tables=service.store.tables)


def test_idle_sessions_expire_but_active_sessions_survive(service):
    from time import monotonic
    pid, _ = fixture_pipeline(service)
    other = service.create_pipeline()['id']
    service.store._last_access[pid] = monotonic() - 7200
    service.store._last_access[other] = monotonic() - 7200
    service.store.expire_idle(protected={pid})
    assert pid in service.store._pipelines and other not in service.store._pipelines


def test_query_lane_stays_available_during_compute(service, monkeypatch):
    from reaxkit.webui.backend import jobs
    pid, _ = fixture_pipeline(service, 1000)
    service.store.upsert_node(pid, PipelineNode(id='view', kind='visualization', name='view', parent_id='source'))
    monkeypatch.setattr(jobs, '_worker', delayed_worker)
    compute = service.submit_node(pid, 'view')
    query = service.jobs.submit(pid, 'query', {'artifact_id': 'data', 'query': {
        'sort_by': [{'column_id': 'step', 'direction': 'desc'}], 'page_size': 10}})
    result = wait_job(service, pid, query)
    assert result['state'] == 'succeeded', result.get('error')
    assert result['result']['rows'][0]['step'] == 999
    assert service.jobs.get(pid, compute['id'])['state'] not in TERMINAL
    service.jobs.cancel(pid, compute['id'])


def test_failed_export_keeps_destination(service, tmp_path):
    pid, _ = fixture_pipeline(service)
    target = tmp_path / 'result.csv'
    target.write_text('existing file', encoding='utf-8')
    job = service.jobs.submit(pid, 'export_table', {'artifact_id': 'data', 'path': str(target)})
    service.jobs.cancel(pid, job['id'])
    assert wait_job(service, pid, job)['state'] == 'cancelled'
    assert target.read_text(encoding='utf-8') == 'existing file'
    assert not list(tmp_path.glob('*.tmp'))


def test_reloading_same_dataset_invalidates_running_job(service):
    pid, _ = fixture_pipeline(service)
    service.store.upsert_node(pid, PipelineNode(id='view', kind='visualization', name='view', parent_id='source'))
    job = service.submit_node(pid, 'view')
    service.store.update_node(pid, 'source', request={}, metadata={})
    assert wait_job(service, pid, job)['state'] == 'superseded'


def test_large_non_table_arrays_are_descriptors(service):
    import numpy as np
    from types import SimpleNamespace
    payload = normalize_result(SimpleNamespace(vector=np.arange(100_000)), tables=service.store.tables)
    assert payload['vector']['kind'] == 'array'
    assert payload['vector']['shape'] == [100_000]
    assert len(json.dumps(payload)) < 10_000


def test_query_and_export_preserve_null_and_empty_string(service):
    pid = service.create_pipeline()['id']
    service.store.store_artifact(pid, ResultArtifact(id='nulls', node_id='x', payload={'table': [
        {'text': None}, {'text': ''}, {'text': 'a'}]}))
    tree = {'type': 'unary-operator', 'subType': 'is blank',
            'left': {'type': 'expression', 'subType': 'field', 'value': 'text'}}
    assert service.query_table(pid, 'nulls', filter_tree=tree)['row_count'] == 2


def test_analysis_publication_creates_views_without_browser(service, monkeypatch):
    from reaxkit.webui.backend import jobs
    monkeypatch.setattr(jobs, '_worker', analysis_worker)
    pid = service.create_pipeline()['id']
    service.store.upsert_node(pid, PipelineNode(id='dataset', kind='dataset', name='Dataset',
        metadata={'dataset': {'engine_detected': 'reaxff'}, 'run_dir': '.'}))
    service.store.upsert_node(pid, PipelineNode(id='analysis', kind='analysis', name='test', parent_id='dataset'))
    result = wait_job(service, pid, service.submit_node(pid, 'analysis'))
    assert result['state'] == 'succeeded', result.get('error')
    snapshot = service.get_pipeline(pid)
    assert any(snapshot['nodes'][n]['request'].get('visualization_type') == 'table' for n in snapshot['children']['analysis'])
    artifact_id = snapshot['nodes']['analysis']['result_ref']
    assert service.query_table(pid, artifact_id)['row_count'] == 20_000
    assert result['worker_duration_ms'] > 0


def test_preview_is_bounded_by_bytes_as_well_as_rows(service):
    descriptor = service.store.tables.write(pd.DataFrame({'wide': ['x' * 50_000] * 200}))
    rows = service.store.tables.preview(descriptor)
    assert 0 < len(rows) < 200
    assert sum(len(row['wide']) for row in rows) <= 4 * 1024 * 1024
