"""Real worker registration, presentation, and hierarchy deletion regressions."""
from types import SimpleNamespace

import dash
from dash import no_update

from reaxkit.webui.backend.schemas import PipelineNode, ResultArtifact
from reaxkit.webui.dash_app import create_dash_app
from test_large_results import wait_job, delayed_worker
from test_workspace import Callbacks, walk


def callback(app, name):
    return next(item['callback'].__wrapped__ for item in app.callback_map.values()
                if 'callback' in item and item['callback'].__wrapped__.__name__ == name)


def test_cold_worker_partial_energy_to_table_and_plot(tmp_path, monkeypatch):
    # No analysis task imports or stub executor in this process or its child.
    (tmp_path / 'fort.73').write_text('Iter. Ebond Eatom\n0 -12.5 2.0\n5 -11.5 3.0\n', encoding='utf-8')
    app = create_dash_app()
    service = app.reaxkit_service
    try:
        pid = service.create_pipeline()['id']
        dataset = service.runtime.load_dataset(pid, run_dir=str(tmp_path), engine='reaxff',
                                               project_root=str(tmp_path / 'workspace'), probe=False)
        analysis = service.add_node(pid, {'parent_id': dataset['id'], 'kind': 'analysis',
            'name': 'Partial Energy Series', 'metadata': {'task_name': 'partial_energy_series'},
            'request': {'components': ['Ebond', 'Eatom']}})
        session = {'pipeline_id': pid, 'selected_node_id': analysis['id']}
        job = wait_job(service, pid, service.submit_node(pid, analysis['id']))
        assert job['state'] == 'succeeded', job.get('error')
        with monkeypatch.context() as patch:
            patch.setattr(dash, 'ctx', SimpleNamespace(triggered_id='job-poll'))
            _, _, snapshot, refs, _ = callback(app, 'poll_jobs')(1, 0, session, [])
        views = [n for n in snapshot['nodes'].values() if n['kind'] == 'visualization']
        assert {n['request']['visualization_type'] for n in views} == {'table', 'plot2d'}
        artifact_id = refs[analysis['id']]
        rows = service.query_table(pid, artifact_id)['rows']
        assert {(r['iter'], r['component'], r['value']) for r in rows} == {
            (0, 'Ebond', -12.5), (0, 'Eatom', 2), (5, 'Ebond', -11.5), (5, 'Eatom', 3)}
        for view in views:
            selected = {**session, 'selected_node_id': view['id']}
            component = callback(app, 'render_result_views')(selected, refs, snapshot, None)
            controls = {getattr(c, 'id', ''): c for c in walk(component) if isinstance(getattr(c, 'id', ''), str)}
            if view['request']['visualization_type'] == 'table':
                assert len(controls['result-table'].data) == 4
                query, _, _ = callback(app, 'submit_page')(0, 100, [], None, ['iter', 'component', 'value'],
                    controls['table-context'].data, None, '')
                assert wait_job(service, pid, query)['state'] == 'succeeded'
                displayed, _, _, notice, _ = callback(app, 'finish_page')(1, query)
                assert len(displayed) == 4 and '4 matching rows' in notice
            else:
                context = controls['plot-context'].data
                assert context['mapping']['group'] == 'component'
                request = {'key': context['key'], 'request_id': 'regression', 'width': 640}
                query, _ = callback(app, 'advance_plot')(request, 0, None, context)
                assert wait_job(service, pid, query)['state'] == 'succeeded'
                _, response = callback(app, 'advance_plot')(request, 1, query, context)
                assert 'error' not in response, response
                assert {t['name'] for t in response['figure']['data']} == {'Ebond', 'Eatom'}
    finally:
        service.close()


def test_hierarchy_delete_analysis_removes_descendants_and_preserves_other_branch(monkeypatch):
    from reaxkit.webui.ui.analysis.callback_sections.pipeline_callbacks import register_pipeline_callbacks
    app = create_dash_app()
    service = app.reaxkit_service
    try:
        pid = service.create_pipeline()['id']
        for node in [PipelineNode(id='dataset', kind='dataset', name='Dataset'),
                     PipelineNode(id='a', kind='analysis', name='A', parent_id='dataset', result_ref='result'),
                     PipelineNode(id='view', kind='visualization', name='View', parent_id='a'),
                     PipelineNode(id='utility', kind='utility', name='Utility', parent_id='a'),
                     PipelineNode(id='b', kind='analysis', name='B', parent_id='dataset')]:
            service.store.upsert_node(pid, node)
        service.store.store_artifact(pid, ResultArtifact(id='result', node_id='a', payload={'table': [{'x': 1}]}))
        callbacks = Callbacks()
        register_pipeline_callbacks(callbacks, service)
        before = service.get_pipeline(pid)
        selected = {'pipeline_id': pid, 'selected_node_id': 'a'}
        assert not callbacks.functions['analysis_delete_available'](selected, before)
        # Delay the real worker so deletion exercises queued/running cleanup.
        monkeypatch.setattr('reaxkit.webui.backend.jobs._worker', delayed_worker)
        job = service.submit_node(pid, 'a')
        session, snapshot, refs, message = callbacks.functions['delete_hierarchy_analysis'](1, selected, before)
        assert session['selected_node_id'] == 'virtual:analysis'
        assert set(snapshot['nodes']) == {'dataset', 'b'}
        assert 'result' not in snapshot['artifacts'] and not refs
        assert message == 'Node deleted'
        assert wait_job(service, pid, job)['state'] in {'cancelled', 'superseded'}
        assert set(service.get_pipeline(pid)['nodes']) == {'dataset', 'b'}
        assert callbacks.functions['analysis_delete_available']({**selected, 'selected_node_id': 'dataset'}, snapshot)
        assert callbacks.functions['delete_hierarchy_analysis'](2, {**selected, 'selected_node_id': 'dataset'}, snapshot) == (no_update,) * 4
    finally:
        service.close()
