"""Bounded activity/results views; panel geometry never passes through Python."""
from hashlib import sha256
import json
from dash import ALL, Input, Output, State, ctx, html, no_update
from reaxkit.webui.backend.artifact_tables import table_descriptor
from reaxkit.webui.backend.jobs import TERMINAL
from reaxkit.webui.ui.shared.workspace import empty_state, icon


def activity_rows(jobs, nodes):
    ordered = sorted(jobs, key=lambda j: (j['state'] in TERMINAL, -j['created']))[:12]
    labels = {'apply': 'Analysis / utility', 'plot': 'Prepare visualization', 'query': 'Table query',
              'probe': 'Dataset metadata', 'export_table': 'Export table', 'save_snapshot': 'Save workspace',
              'import_snapshot': 'Import workspace', 'export_bundle': 'Export bundle'}
    rows = []
    for job in ordered:
        label = (nodes.get(job.get('node_id')) or {}).get('name') or labels.get(job['operation'], job['operation'])
        children = [html.Strong(label), html.Span(job['stage'], className='rk-job-stage'),
                    html.Span(job['state'], className='rk-job-state ' + job['state'])]
        if job['state'] not in TERMINAL:
            children.append(html.Button('Stop', id={'type': 'activity-cancel', 'job_id': job['id']}, n_clicks=0,
                                        title=f'Stop {label}'))
        else:
            elapsed = max(0, (job.get('finished') or job['created']) - job['created'])
            children.append(html.Small(f'{elapsed:.1f} s'))
        if job.get('error'):
            children.append(html.Details([html.Summary('Error details'), html.Pre(str(job['error'])[-8000:])]))
        rows.append(html.Div(children, className='rk-job-row', key=job['id']))
    return rows or empty_state('Ready when you are', 'Run an analysis to see its progress here.', eyebrow='ACTIVITY')


def register_workspace_callbacks(app, service):
    @app.callback(Output('activity-jobs', 'children'), Output('activity-count', 'children'),
                  Output('activity-view-key', 'data'), Input('activity-tick', 'n_intervals'),
                  Input('session-store', 'data'), State('workspace-state', 'data'), State('activity-view-key', 'data'))
    def render_activity(_tick, session, workspace, previous):
        if not session or not session.get('pipeline_id'):
            return no_update, no_update, no_update
        if workspace and (not workspace.get('drawerOpen') or workspace.get('maximized') or workspace.get('drawerTab') != 'jobs'):
            return no_update, no_update, no_update
        jobs = service.jobs.list(session['pipeline_id'])
        key = sha256(json.dumps(jobs, sort_keys=True, default=str).encode()).hexdigest()
        if key == previous:
            return no_update, no_update, no_update
        snapshot = service.get_pipeline(session['pipeline_id'])
        active = sum(j['state'] not in TERMINAL for j in jobs)
        return activity_rows(jobs, snapshot['nodes']), f'{active} active / {len(jobs)} recent', key

    @app.callback(Output('status-banner', 'children', allow_duplicate=True),
                  Input({'type': 'activity-cancel', 'job_id': ALL}, 'n_clicks'), State('session-store', 'data'),
                  prevent_initial_call=True)
    def cancel_activity(_clicks, session):
        if not ctx.triggered or not ctx.triggered[0].get('value') or not session:
            return no_update
        try:
            service.jobs.cancel(session['pipeline_id'], ctx.triggered_id['job_id'])
            return 'Stopping selected job…'
        except KeyError:
            return 'Job is no longer available.'

    @app.callback(Output('activity-results', 'children'), Input('pipeline-store', 'data'),
                  Input('session-store', 'data'))
    def render_results(snapshot, session):
        if not snapshot or not session:
            return empty_state('Your results will appear here', 'Completed analyses remain available while you work.', eyebrow='RESULTS')
        nodes = snapshot.get('nodes', {})
        sources = [n for n in nodes.values() if n.get('result_ref') and n.get('kind') in ('analysis', 'utility')][-30:]
        rows = []
        for node in reversed(sources):
            try:
                artifact = service.store.get_artifact(session['pipeline_id'], node['result_ref'])
            except KeyError:
                continue
            table = table_descriptor(artifact.payload)
            description = f"{table['row_count']:,} rows · {len(table['columns'])} columns" if table else 'Analysis result'
            if node.get('status') in ('running', 'dirty'):
                description += ' · previous result'
            views = [n for n in nodes.values() if n.get('parent_id') == node['id'] and n.get('kind') == 'visualization']
            target = views[0]['id'] if views else node['id']
            provenance = artifact.metadata.get('provenance') or {}
            detail = html.Details([html.Summary('Provenance'), html.Pre(json.dumps(provenance, indent=2, default=str)[:6000])]) if provenance else None
            rows.append(html.Div([icon('plot'), html.Div([html.Strong(node.get('name') or 'Result'), html.Small(description), detail]),
                html.Button('Open', id={'type': 'result-open', 'node_id': target}, n_clicks=0)], className='rk-result-card', key=node['id']))
        return rows or empty_state('Your results will appear here', 'Completed analyses remain available while you work.', eyebrow='RESULTS')

    @app.callback(Output('session-store', 'data', allow_duplicate=True),
                  Input({'type': 'result-open', 'node_id': ALL}, 'n_clicks'), State('session-store', 'data'),
                  prevent_initial_call=True)
    def open_result(_clicks, session):
        if not session or not ctx.triggered or not ctx.triggered[0].get('value'):
            return no_update
        return {**session, 'selected_node_id': ctx.triggered_id['node_id']}

    @app.callback(Output('canvas-title', 'children'), Output('parameters-status', 'children'),
                  Input('session-store', 'data'), Input('pipeline-store', 'data'))
    def describe_selection(session, snapshot):
        node = (snapshot or {}).get('nodes', {}).get((session or {}).get('selected_node_id'))
        if not node:
            return 'Workspace', 'Choose a dataset, analysis, or presentation.'
        status = str(node.get('status') or 'idle')
        messages = {'idle': 'Ready to configure', 'done': 'Last execution completed', 'dirty': 'Parameters changed · previous result retained',
                    'running': 'Running in background · you can keep editing', 'error': 'Last execution failed · inspect Activity for details'}
        return node.get('name') or 'Workspace', messages.get(status, status.title())
