"""Latest-request-wins, asynchronous plots with browser-side state preservation."""
from hashlib import sha256
import json
from uuid import uuid4

from dash import ClientsideFunction, Input, Output, State, dcc, html, no_update

from reaxkit.webui.backend.artifact_tables import table_descriptor
from reaxkit.webui.backend.jobs import TERMINAL
from reaxkit.webui.backend.plot_queries import plot_mapping

GRAPH = {'type': 'plot-graph', 'slot': 'canvas'}


def plot_context(pipeline_id, artifact, node):
    from reaxkit.presentation.specs import ensure_presentation_spec, spec_to_dash_request
    request = dict(node.get('request') or {})
    spec = ensure_presentation_spec((node.get('metadata') or {}).get('presentation_spec'))
    if spec is not None:
        for key, value in spec_to_dash_request(spec).items():
            if not request.get(key):
                request[key] = value
        if request.get('visualization_type') == 'histogram' and not request.get('x_col'):
            request['x_col'] = spec.mapping.get('value_col', '')
        request['_presentation_options'] = spec.options
    context = dict(pipeline_id=pipeline_id, artifact_id=artifact['id'], node_id=node['id'], request=request)
    descriptor = table_descriptor(artifact.get('payload', {}))
    if descriptor is None:
        return None
    context['revision'] = descriptor['revision']
    context['mapping'] = plot_mapping(descriptor, request)
    context['key'] = sha256(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest()
    return context


def result_plot(context):
    context = {**context, 'view_id': uuid4().hex}
    is_frame = bool(context['mapping']['frame'])
    return html.Div([
        dcc.Store(id='plot-context', data=context), dcc.Store(id='plot-request'),
        dcc.Store(id='plot-query-job'), dcc.Store(id='plot-response'),
        dcc.Interval(id='plot-query-poll', interval=150, disabled=True),
        html.Div([html.Label('Frame value', htmlFor='plot-frame'),
                  dcc.Input(id='plot-frame', type='number', debounce=True, placeholder='First available'),
                  html.Span(' Enter a stored frame value; only this frame is prepared.')],
                 style={'display': 'flex' if is_frame else 'none', 'gap': '8px'}),
        html.Button('Refresh view', id='plot-refresh', n_clicks=0, style={'alignSelf': 'flex-end'}),
        html.Div('Preparing full-result view...', id='plot-query-status', role='status', className='rk-log-name'),
        dcc.Graph(id=GRAPH, figure={'data': [], 'layout': {'template': 'plotly_white'}},
                  style={'flex': '1 1 0', 'height': '0', 'minHeight': '0', 'width': '100%'},
                  config={'displaylogo': False, 'responsive': True}, responsive=True),
        html.Div(id='plot-exact-value', className='rk-log-name'),
    ], style={'display': 'flex', 'flexDirection': 'column', 'flex': '1 1 auto', 'minHeight': '0', 'width': '100%'})


def register_plot_callbacks(app, service):
    app.clientside_callback(ClientsideFunction(namespace='reaxkitPlots', function_name='request'),
        Output('plot-request', 'data'), Input('plot-context', 'data'), Input(GRAPH, 'relayoutData'),
        Input('plot-frame', 'value'), Input('plot-refresh', 'n_clicks'), State('plot-request', 'data'))

    @app.callback(Output('plot-query-job', 'data'), Output('plot-response', 'data'), Input('plot-request', 'data'),
                  Input('plot-query-poll', 'n_intervals'), State('plot-query-job', 'data'),
                  State('plot-context', 'data'), prevent_initial_call=True)
    def advance_plot(request, _tick, current, context):
        if not request or not context or request['key'] != context['key']:
            return no_update, no_update
        pid = context['pipeline_id']
        if not current or current['request_id'] != request['request_id']:
            try:
                job = service.jobs.submit(pid, 'plot', dict(artifact_id=context['artifact_id'],
                    request=context['request'], width=request['width'], x_range=request.get('x_range'),
                    frame=request.get('frame'), view_id=context['view_id']))
                current = dict(id=job['id'], pipeline_id=pid, request_id=request['request_id'], key=context['key'])
            except Exception as exc:
                return None, dict(request_id=request['request_id'], key=context['key'], error=str(exc))
        else:
            try:
                job = service.jobs.get(pid, current['id'])
            except KeyError:
                return None, dict(request_id=request['request_id'], key=context['key'], error='View expired. Zoom or reselect the view.')
        if job['state'] not in TERMINAL:
            return current, dict(request_id=request['request_id'], key=context['key'], pending=True, status=job['stage'])
        response = dict(request_id=request['request_id'], key=context['key'])
        if job['state'] != 'succeeded':
            response['error'] = job.get('error') or f"View {job['state']}. Zoom or reselect to retry."
        else:
            try:
                result = job['result']
                figure = result['figure']
                figure['layout']['meta']['view_key'] = context['key']
                response.update(figure=figure, status=result['status'])
            except Exception as exc:
                response['error'] = str(exc)
        return current, response

    app.clientside_callback(ClientsideFunction(namespace='reaxkitPlots', function_name='publish'),
        Output(GRAPH, 'figure'), Output('plot-query-status', 'children'),
        Output('plot-query-poll', 'disabled'),
        Input('plot-response', 'data'), Input('plot-request', 'data'), State(GRAPH, 'figure'))

    @app.callback(Output('plot-exact-value', 'children'), Input(GRAPH, 'clickData'),
                  Input('plot-request', 'data'), State('plot-context', 'data'), State(GRAPH, 'figure'),
                  prevent_initial_call=True)
    def exact_value(click, request, context, figure):
        from dash import ctx
        if ctx.triggered_id == 'plot-request' or not click or not context:
            return ''
        meta = (figure or {}).get('layout', {}).get('meta', {})
        if meta.get('view_key') != context['key'] or meta.get('kind') == 'histogram':
            return ''
        point = (click.get('points') or [{}])[0]
        data = point.get('customdata') or []
        if not data or data[0] is None:
            return f"Displayed aggregate: x={point.get('x')}, y={point.get('y')}"
        try:
            row_id = int(data[0])
            if row_id < 0:
                raise ValueError('Invalid row')
            artifact = service.store.get_artifact(context['pipeline_id'], context['artifact_id'])
            descriptor = table_descriptor(artifact.payload)
            if descriptor['revision'] != context['revision']:
                return 'Result changed; select a point in the current view.'
            row = service.query_table(context['pipeline_id'], context['artifact_id'],
                page=row_id, page_size=1, columns=descriptor['columns'][:30])['rows']
            text = json.dumps(row[0] if row else {}, ensure_ascii=False, default=str)
            return html.Details([html.Summary(f'Exact source row {row_id} (first 30 columns)'),
                                 html.Pre(text[:8000] + ('... See table for complete values.' if len(text) > 8000 else ''))], open=True)
        except (KeyError, ValueError, TypeError) as exc:
            return f'Exact value unavailable: {exc}'
