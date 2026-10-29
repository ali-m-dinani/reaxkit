import copy
import json

import pytest
from dash import html, no_update

from reaxkit.webui.ui.shared.forms import filter_tree_rows, properties_key
from reaxkit.webui.ui.analysis.tasks.validation import validate_request
from reaxkit.webui.ui.analysis.tasks.autogen import render_auto_task_form, register_auto_task_callbacks
from reaxkit.webui.ui.logs.callbacks import _read_tail
from reaxkit.webui.ui.shell.workspace_callbacks import activity_rows


class Callbacks:
    def __init__(self):
        self.functions = {}

    def callback(self, *args, **kwargs):
        def register(fn):
            self.functions[fn.__name__] = fn
            return fn
        return register


def walk(component):
    if isinstance(component, (list, tuple)):
        for child in component:
            yield from walk(child)
    elif hasattr(component, 'to_plotly_json'):
        yield component
        yield from walk(getattr(component, 'children', None))


SCHEMA = {'fields': [
    {'name': 'every', 'kind': 'int', 'default': 1, 'semantic': {'min': 1, 'units': 'frames'}},
    {'name': 'frames', 'kind': 'list[int]', 'default': None},
    {'name': 'dims', 'kind': 'str', 'default': 'xyz', 'semantic': {'choices': ['xyz', 'xy']}},
    {'name': 'backend', 'kind': 'str', 'default': 'auto', 'semantic': {'choices': ['auto']}}]}


@pytest.mark.parametrize('name,raw', [('every', 0), ('every', 1.2), ('every', 'nan'),
                                    ('frames', '1,nope,3'), ('frames', '0:1000000000000'),
                                    ('frames', '1:100:0'), ('dims', 'unknown')])
def test_draft_validation_rejects_invalid_or_unbounded_input(name, raw):
    request, errors = validate_request(SCHEMA, {name: 'previous'}, [{'name': name}], [raw])
    assert errors and request[name] == 'previous'


def test_draft_validation_keeps_range_semantics_and_unedited_request_keys():
    request, errors = validate_request(SCHEMA, {'custom': 42}, [{'name': 'frames'}, {'name': 'every'}], ['1-3,8:12:2', 3])
    assert not errors
    assert request == {'custom': 42, 'frames': [1, 2, 3, 8, 10], 'every': 3}


def test_properties_ignore_job_updates_but_follow_selection_and_schema():
    snapshot = {'nodes': {'a': {'id': 'a', 'kind': 'analysis', 'name': 'msd', 'request': {'every': 1}}}}
    session = {'pipeline_id': 'p', 'selected_node_id': 'a'}
    key = properties_key(snapshot, session, {}, {}, {})
    updated = copy.deepcopy(snapshot)
    updated['nodes']['a'].update(status='running', result_ref='new', request={'every': 2})
    assert properties_key(updated, session, {'a': 'new'}, {}, {}) == key
    assert properties_key(updated, {**session, 'selected_node_id': 'b'}, {}, {}, {}) != key
    updated['nodes']['a']['metadata'] = {'task_name': 'rdf'}
    assert properties_key(updated, session, {}, {}, {}) != key


def test_search_keeps_ancestors_and_keyboard_entry():
    rows = [html.Button(label, tabIndex=-1, **{'data-label': label, 'data-depth': depth})
            for label, depth in [('Dataset', 0), ('MSD', 1), ('Presentation', 2), ('RDF', 1)]]
    visible = filter_tree_rows(rows, 'presentation')
    assert [r.children for r in visible] == ['Dataset', 'MSD', 'Presentation']
    assert visible[0].tabIndex == 0
    assert 'No matching' in filter_tree_rows(rows, 'missing')[0].children


def test_auto_form_groups_units_advanced_and_keeps_run_mounted():
    form = render_auto_task_form([], {'id': 'a', 'status': 'running'}, task_name='rdf', schema=SCHEMA)
    controls = list(walk(form))
    assert any(isinstance(c, html.Details) and not getattr(c, 'open', False) for c in controls)
    assert any(isinstance(c, html.Label) and c.children == 'every (frames)' for c in controls)
    run = next(c for c in controls if getattr(c, 'id', None) == 'btn-apply-node')
    assert run.disabled
    fields = [c for c in controls if isinstance(getattr(c, 'id', None), dict)]
    assert all(c.persistence == 'a' for c in fields)


def test_apply_and_discard_do_not_live_mutate_analysis():
    app = Callbacks()
    node = {'id': 'a', 'kind': 'analysis', 'name': 'msd', 'request': {'every': 1}}
    class Service:
        changes = []
        def get_catalog(self): return {'analysis_schemas': {'msd': SCHEMA}}
        def update_node(self, pid, nid, patch):
            self.changes.append(patch)
            node.update(patch)
        def get_pipeline(self, pid): return {'nodes': {'a': node}}
    service = Service()
    register_auto_task_callbacks(app, service, selected_node=lambda *_: node)
    ids, session = [{'name': 'every'}], {'pipeline_id': 'p'}
    message, apply_disabled, run_disabled = app.functions['validate_draft']([4], {}, ids, session)
    assert 'Unsaved' in message and not apply_disabled and not run_disabled
    assert service.changes == []
    assert app.functions['discard_draft'](1, ids, session, {}) == [1]
    app.functions['apply_draft'](1, [4], ids, session, {})
    assert node['request']['every'] == 4
    assert app.functions['discard_draft'](1, ids, session, {}) == [4]
    app.functions['apply_draft'](1, [0], ids, session, {})
    assert len(service.changes) == 1


def test_run_validates_and_saves_draft_before_submitting():
    from reaxkit.webui.ui.analysis.callback_sections.execution_callbacks import register_execution_callbacks
    app = Callbacks()
    node = {'id': 'a', 'kind': 'analysis', 'name': 'msd', 'request': {'every': 1}}
    snapshot = {'nodes': {'a': node}}
    class Service:
        submitted = 0
        def get_catalog(self): return {'analysis_schemas': {'msd': SCHEMA}}
        def update_node(self, pid, nid, patch): node.update(patch)
        def get_pipeline(self, pid): return snapshot
        def submit_node(self, pid, nid):
            assert node['request']['every'] == 4
            self.submitted += 1
            return {'id': 'job'}
    service = Service()
    register_execution_callbacks(app, service)
    run = app.functions['on_apply_node']
    session = {'pipeline_id': 'p', 'selected_node_id': 'a'}
    result = run(1, session, snapshot, {}, {}, [4], [{'name': 'every'}])
    assert service.submitted == 1 and result[0] == snapshot
    result = run(2, session, snapshot, {}, {}, [0], [{'name': 'every'}])
    assert service.submitted == 1 and 'Check parameters' in result[-1]


def test_utility_editor_refreshes_join_schema_and_own_completion():
    node = {'id': 'u', 'kind': 'utility', 'request': {'right_source_node_id': 'a'}, 'status': 'running'}
    snapshot, session = {'nodes': {'u': node}}, {'pipeline_id': 'p', 'selected_node_id': 'u'}
    first = properties_key(snapshot, session, {}, {}, {})
    node['request']['right_source_node_id'] = 'b'
    assert properties_key(snapshot, session, {}, {}, {}) != first
    second = properties_key(snapshot, session, {}, {}, {})
    node['status'] = 'done'
    assert properties_key(snapshot, session, {}, {}, {}) != second


def test_log_tail_and_activity_payloads_are_bounded(tmp_path):
    path = tmp_path / 'huge.log'
    path.write_text(('old line\n' * 100_000) + 'last line\n', encoding='utf-8')
    text = _read_tail(path, max_lines=8, max_bytes=128)
    assert text.splitlines()[0] == 'last line'
    assert len(text.splitlines()) == 8 and len(text.encode()) <= 128
    jobs = [{'id': str(i), 'state': 'failed', 'created': i, 'finished': i + 1,
             'operation': 'apply', 'stage': 'failed', 'error': 'E' * 100_000} for i in range(100)]
    rows = activity_rows(jobs, {})
    assert len(rows) == 12
    assert len(next(c for c in walk(rows[0]) if isinstance(c, html.Pre)).children) == 8000


def test_dash_workspace_layout_assets_and_callbacks():
    from reaxkit.webui.dash_app import create_dash_app
    app = create_dash_app()
    try:
        controls = list(walk(app.layout))
        ids = [json.dumps(c.id, sort_keys=True) for c in controls if hasattr(c, 'id')]
        assert len(ids) == len(set(ids))
        separators = [c for c in controls if getattr(c, 'role', '') == 'separator']
        assert len(separators) == 3
        assert all(c.tabIndex == 0 and getattr(c, 'aria-controls') for c in separators)
        with app.server.test_client() as client:
            for asset in ['workspace.js', 'workspace.css', 'workspace-icons.css']:
                assert client.get('/assets/' + asset).status_code == 200
            assert client.get('/_dash-layout').status_code == 200
            dependencies = client.get('/_dash-dependencies').json
            assert not any('splitter-' in str(d['inputs']) for d in dependencies)
    finally:
        app.reaxkit_service.close()
