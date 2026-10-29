import json
import time

import numpy as np
import pandas as pd
import pytest

from reaxkit.webui.backend.artifact_tables import ArtifactTables, table_descriptor
from reaxkit.webui.backend.plot_queries import summarize_plot
from reaxkit.webui.presentation.plot_summary import summary_figure
from test_large_results import service, fixture_pipeline, wait_job


@pytest.fixture
def tables(tmp_path):
    return ArtifactTables(tmp_path)


def data(tables, **columns):
    return {'table': tables.write(pd.DataFrame(columns))}


def test_full_range_extrema_endpoints_and_zoom(tables):
    x = np.arange(100_000)
    y = np.zeros(len(x))
    y[90_123], y[91_001] = 999, -777
    payload = data(tables, x=x, y=y)
    request = {'x_col': 'x', 'y_col': 'y'}
    result = summarize_plot(tables, payload, request, width=320)
    trace = result['traces'][0]
    assert (trace['x'][0], trace['x'][-1]) == (0, 99_999)
    assert max(trace['y']) == 999 and min(trace['y']) == -777
    assert result['displayed_points'] <= result['budget']
    assert result['reduced'] and result['view_rows'] == len(x)
    zoom = summarize_plot(tables, payload, request, x_range=[90_100, 90_140], width=320)
    assert zoom['traces'][0]['x'] == list(range(90_100, 90_141))
    assert zoom['traces'][0]['row_ids'] == list(range(90_100, 90_141))
    assert not zoom['reduced']
    fractional = summarize_plot(tables, payload, request, x_range=[90_100.1, 90_105.9])
    assert fractional['traces'][0]['x'] == list(range(90_101, 90_106))


def test_dense_gaps_never_connect_across_missing_observations(tables):
    n = 20_000
    values = np.sin(np.arange(n))
    values[1000:1300] = np.nan
    values[13_001], values[13_002] = np.nan, np.inf
    values[12_999] = 900
    summary = summarize_plot(tables, data(tables, x=np.arange(n), y=values), {}, width=160)
    trace = summary['traces'][0]
    assert max(v for v in trace['y'] if v is not None) == 900
    assert summary['displayed_points'] <= summary['budget']
    for i in range(1, len(trace['x'])):
        a, b = trace['x'][i-1:i+1]
        if a is not None and b is not None and trace['y'][i-1] is not None and trace['y'][i] is not None:
            assert not np.any(~np.isfinite(values[int(a):int(b)+1]))
    figure = summary_figure(summary, {}, view_key='test')
    assert all(not trace.connectgaps for trace in figure.data)
    assert len(figure.to_json()) < 4 * 1024 * 1024


def test_all_groups_share_budget_and_remain_stable_on_zoom(tables):
    n = 120_000
    payload = data(tables, x=np.arange(n), y=np.arange(n) % 123, group=np.arange(n) % 75)
    request = {'x_col': 'x', 'y_col': 'y', 'group_col': 'group'}
    result = summarize_plot(tables, payload, request, width=160)
    zoom = summarize_plot(tables, payload, request, width=160, x_range=[110_000, 119_000])
    assert result['group_count'] == 75 and result['shown_groups'] == 10
    assert result['displayed_points'] <= result['budget']
    assert [t['name'] for t in result['traces']] == [t['name'] for t in zoom['traces']]


@pytest.mark.parametrize('mode', ['mean', 'median', 'min', 'max', 'sum', 'count', 'std'])
def test_aggregation_precedes_decimation(tables, mode):
    frame = pd.DataFrame({'x': np.arange(20_000) % 7, 'y': np.arange(20_000, dtype=float)})
    summary = summarize_plot(tables, {'table': tables.write(frame)}, {'x_col': 'x', 'y_col': 'y', 'group_agg': mode})
    groups = frame.groupby('x').y
    expected = groups.std(ddof=0) if mode == 'std' else getattr(groups, mode)()
    assert summary['traces'][0]['y'] == pytest.approx(expected.tolist())
    assert all(r is None for r in summary['traces'][0]['row_ids'])


def test_filters_scan_full_data_and_preserve_original_row_ids(tables):
    payload = data(tables, x=np.arange(50_000), y=np.arange(50_000))
    result = summarize_plot(tables, payload, {'row_filters': [{'column': 'x', 'op': '>=', 'value': '49000'}]})
    trace = result['traces'][0]
    assert result['matching_rows'] == 1000
    assert trace['row_ids'][0] == 49000
    original = tables.query(table_descriptor(payload), page=trace['row_ids'][0], page_size=1)
    assert original['rows'][0]['x'] == trace['x'][0]


def test_histogram_counts_match_numpy_entire_result(tables):
    values = np.concatenate([np.linspace(-10, 10, 99_999), [1e4, np.nan, np.inf]])
    summary = summarize_plot(tables, data(tables, x=values), {'visualization_type': 'histogram'})
    bins = summary['bins']
    counts, _ = np.histogram(values[np.isfinite(values)], bins=len(bins))
    assert [b['count'] for b in bins] == counts.tolist()
    assert sum(b['count'] for b in bins) == 100_000
    assert summary['missing_rows'] == 2
    assert bins[-1]['right'] == 1e4
    fig = summary_figure(summary, {}, view_key='hist')
    assert fig.data[0].type == 'bar'


@pytest.mark.parametrize('values', [[1, 1, 1], [None, None], []])
def test_degenerate_histograms(tables, values):
    result = summarize_plot(tables, data(tables, x=values), {'visualization_type': 'histogram'})
    assert sum(b['count'] for b in result['bins']) == sum(v is not None for v in values)


def test_3d_one_frame_stable_sample_and_spatial_extrema(tables):
    n = 100_000
    rng = np.random.default_rng(42)
    points = rng.normal(size=(n, 3))
    payload = data(tables, x=points[:, 0], y=points[:, 1], z=points[:, 2],
                   atom_id=np.arange(n), frame_index=np.arange(n) // 50_000)
    request = {'visualization_type': 'scatter3d'}
    a = summarize_plot(tables, payload, request, frame=1)
    b = summarize_plot(tables, payload, request, frame=1)
    assert a['points'] == b['points']
    assert a['frame_rows'] == 50_000 and a['reduced']
    assert min(a['points']['row_ids']) >= 50_000
    assert max(a['points']['row_ids']) > 98_000
    assert a['geometry_points'] <= a['budget']
    for i, axis in enumerate('xyz'):
        assert min(a['points'][axis]) == min(points[50_000:, i])
        assert max(a['points'][axis]) == max(points[50_000:, i])
    first = summarize_plot(tables, payload, request)
    assert first['frame'] == 0
    assert first['frame_min'] == 0 and first['frame_max'] == 1
    fig = summary_figure(a, request, view_key='frame')
    assert fig.data[0].type == 'scatter3d'
    assert fig.layout.scene.uirevision == fig.layout.uirevision
    assert len(set(fig.data[0].ids)) == len(fig.data[0].ids)
    assert len(fig.to_json()) < 4 * 1024 * 1024


def test_3d_small_frame_cell_and_bonds(tables):
    payload = data(tables, x=[0, 1, 2], y=[0, 1, 0], z=[0, 0, 1], atom_id=[42, 50, 70])
    payload.update(cell={'origin': [0, 0, 0], 'vectors': [[3, 0, 0], [1, 3, 0], [0, 0, 3]]},
                   bonds=[[42, 50], [50, 70], [70, 99]])
    request = {'visualization_type': 'scatter3d'}
    summary = summarize_plot(tables, payload, request)
    assert summary['displayed_points'] == 3 and not summary['reduced']
    assert summary['bonds'] == [['42', '50'], ['50', '70']]
    figure = summary_figure(summary, request, view_key='molecule')
    assert len(figure.data) == 3
    assert len(figure.data[1].x) == 36 and len(figure.data[2].x) == 6


def test_invalid_columns_do_not_execute_sql(tables):
    payload = data(tables, x=[1, 2], y=[1, 2])
    with pytest.raises(ValueError, match='Unknown column'):
        summarize_plot(tables, payload, {'x_col': 'x; DROP TABLE points;--'})


def test_reserved_column_collision_keeps_physical_row_identity(tables):
    payload = data(tables, x=np.arange(80_000), y=np.arange(80_000), file_row_number=np.zeros(80_000))
    result = summarize_plot(tables, payload, {'x_col': 'x', 'y_col': 'y'}, x_range=[79_900, 79_999])
    assert result['traces'][0]['row_ids'] == list(range(79_900, 80_000))


def test_plot_payload_budget_and_empty_range(tables, monkeypatch):
    from reaxkit.webui.backend import plot_queries
    config = plot_queries.load_ui_performance_config()
    config['plot_max_bytes'] = 2048
    monkeypatch.setattr(plot_queries, 'load_ui_performance_config', lambda: config)
    payload = data(tables, x=np.arange(1000), y=np.arange(1000))
    with pytest.raises(ValueError, match='payload byte budget'):
        summarize_plot(tables, payload, {})
    result = summarize_plot(tables, payload, {}, x_range=[2000, 3000])
    assert result['displayed_points'] == 0
    assert result['traces'][0]['x'] == []


def test_presentation_spec_mapping_and_labels_are_preserved(tables):
    from reaxkit.webui.ui.analysis.callback_sections.plot_callbacks import plot_context
    payload = data(tables, x=[1, 2], y=[5, 7])
    context = plot_context('pipeline', {'id': 'result', 'payload': payload}, {
        'id': 'view', 'request': {}, 'metadata': {'presentation_spec': {
            'renderer': 'single_plot', 'mapping': {'x_col': 'x', 'y_col': 'y'},
            'options': {'title': 'Energy', 'ylabel': 'Energy (eV)'}}}})
    summary = summarize_plot(tables, payload, context['request'])
    figure = summary_figure(summary, context['request'], view_key=context['key'])
    assert figure.layout.title.text == 'Energy'
    assert figure.layout.yaxis.title.text == 'Energy (eV)'


def test_plot_export_uses_displayed_summary_and_latest_camera():
    from reaxkit.webui.ui.analysis.callback_sections.canvas_callbacks import _displayed_figure
    figure = {'data': [{'type': 'scatter3d', 'x': [1], 'y': [2], 'z': [3]}],
              'layout': {'meta': {'summary': True}, 'scene': {'camera': {'eye': {'x': 1, 'y': 1, 'z': 1}}}}}
    displayed = _displayed_figure([figure], [{'slot': 'canvas'}], [{'scene.camera': {'eye': {'x': 3, 'y': 2, 'z': 1}},
                                  'xaxis.range[0]': 100, 'xaxis.range[1]': 200}])
    assert displayed.layout.scene.camera.eye.x == 3
    assert displayed.layout.xaxis.range == (100, 200)
    assert displayed.layout.meta['summary']
    assert list(displayed.data[0].x) == [1]
    assert _displayed_figure([{'data': []}], [{'slot': 'canvas'}], []) is None


def test_plot_jobs_cached_by_revision_range_and_cancel_obsolete(service):
    pid, artifact = fixture_pipeline(service, 50_000)
    args = dict(artifact_id=artifact.id, request={'x_col': 'step', 'y_col': 'value'}, width=320, view_id='test')
    job = service.jobs.submit(pid, 'plot', args)
    result = wait_job(service, pid, job)
    assert result['state'] == 'succeeded', result['error']
    cached = service.jobs.submit(pid, 'plot', {**args, 'view_id': 'another-browser'})
    assert cached['state'] == 'succeeded' and cached['result'] == result['result']
    old = service.jobs.submit(pid, 'plot', {**args, 'x_range': [100, 200]})
    new = service.jobs.submit(pid, 'plot', {**args, 'x_range': [40_000, 40_100]})
    assert wait_job(service, pid, old)['state'] == 'cancelled'
    latest = wait_job(service, pid, new)
    assert latest['state'] == 'succeeded', latest['error']
    assert latest['result']['summary']['x_range'] == [40_000, 40_100]
    service.jobs.query_cache.clear()
    assert service.jobs.get(pid, new['id'])['state'] == 'expired'


def test_dash_plot_mount_and_background_response():
    from reaxkit.webui.dash_app import create_dash_app
    from reaxkit.webui.backend.schemas import PipelineNode
    from reaxkit.webui.ui.analysis.callback_sections.plot_callbacks import plot_context
    app = create_dash_app()
    service = app.reaxkit_service
    try:
        pid, artifact = fixture_pipeline(service, 100_000)
        node = PipelineNode(id='plot', kind='visualization', name='Plot', parent_id='source',
                            request={'visualization_type': 'plot2d', 'x_col': 'step', 'y_col': 'value'})
        service.store.upsert_node(pid, node)
        with app.server.test_client() as client:
            assert client.get('/assets/plot_queries.js').status_code == 200
            response = client.post('/_dash-update-component', json={
                'output': 'canvas-content.children', 'outputs': {'id': 'canvas-content', 'property': 'children'},
                'inputs': [{'id': 'session-store', 'property': 'data', 'value': {'pipeline_id': pid, 'selected_node_id': 'plot'}},
                           {'id': 'result-store', 'property': 'data', 'value': {}},
                           {'id': 'pipeline-store', 'property': 'data', 'value': service.get_pipeline(pid)}],
                'state': [{'id': 'plot-context', 'property': 'data', 'value': None}],
                'changedPropIds': ['session-store.data']})
            assert response.status_code == 200, response.get_data(as_text=True)
            assert len(response.data) < 6000  # No rows or old prefix preview in mount response.
            context = response.json['response']['canvas-content']['children']['props']['children'][0]['props']['data']
            request = dict(key=context['key'], width=320, frame=None, x_range=None, request_id='abc')
            callback = next(v['callback'].__wrapped__ for k, v in app.callback_map.items() if 'plot-query-job.data' in k)
            current, _ = callback(request, 0, None, context)
            job = wait_job(service, pid, current)
            assert job['state'] == 'succeeded', job['error']
            _, result = callback(request, 1, current, context)
            assert result['request_id'] == 'abc' and 'figure' in result, result
            assert result['figure']['layout']['meta']['summary']
            assert '100,000 matching source rows' in result['status']
    finally:
        service.close()


def test_plot_cancellation_isolated_between_browser_views(service):
    pid, artifact = fixture_pipeline(service, 2000)
    arguments = dict(artifact_id=artifact.id, request={'x_col': 'step', 'y_col': 'value'}, width=320)
    first = service.jobs.submit(pid, 'plot', {**arguments, 'view_id': 'first'})
    second = service.jobs.submit(pid, 'plot', {**arguments, 'view_id': 'second'})
    assert first['id'] != second['id']
    service.jobs.cancel_view(pid, 'first')
    assert wait_job(service, pid, first)['state'] == 'cancelled'
    result = wait_job(service, pid, second)
    assert result['state'] == 'succeeded', result['error']
