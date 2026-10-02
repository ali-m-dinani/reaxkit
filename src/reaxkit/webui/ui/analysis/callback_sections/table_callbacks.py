"""Bounded server table pages. Filters and sorts run in cancellable workers."""
from dash import Input, Output, State, dash_table, dcc, html, no_update

from reaxkit.webui.backend.artifact_tables import table_descriptor
from reaxkit.webui.backend.jobs import TERMINAL


def result_table(service, pipeline_id, artifact, row_filters=None):
    descriptor = table_descriptor(artifact.get('payload', {}))
    if descriptor is None:
        return 'No tabular result.'
    columns = descriptor['columns']
    first, initial_error = None, ''
    try:
        first = service.query_table(pipeline_id, artifact['id'], columns=columns[:20]) if columns and not row_filters else None
    except ValueError as exc:
        initial_error = str(exc)
    return html.Div([
        dcc.Store(id='table-context', data={'pipeline_id': pipeline_id, 'artifact_id': artifact['id'], 'row_filters': row_filters or []}),
        dcc.Store(id='table-query-job'), dcc.Interval(id='table-query-poll', interval=100, disabled=True),
        html.Div(f"{descriptor['row_count']:,} rows · full-result sorting and filtering"),
        dcc.Dropdown(id='table-columns', options=columns, value=columns[:20], multi=True),
        html.Div(initial_error, id='table-query-status'),
        dash_table.DataTable(id='result-table', data=first['rows'] if first else [], columns=[{'name': c, 'id': c} for c in columns[:20]],
            page_action='custom', page_current=0, page_size=100, page_count=first['page_count'] if first else 0,
            sort_action='custom', sort_mode='multi', sort_by=[], filter_action='custom', filter_query='',
            style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'fontSize': '12px'}),
    ], style={'overflow': 'auto', 'width': '100%'})


def register_table_callbacks(app, service):
    @app.callback(Output('table-query-job', 'data'), Output('table-query-poll', 'disabled', allow_duplicate=True),
                  Output('table-query-status', 'children', allow_duplicate=True),
                  Input('result-table', 'page_current'), Input('result-table', 'page_size'),
                  Input('result-table', 'sort_by'), Input('result-table', 'derived_filter_query_structure'),
                  Input('table-columns', 'value'), Input('table-context', 'data'),
                  State('table-query-job', 'data'), State('result-table', 'filter_query'),
                  prevent_initial_call='initial_duplicate')
    def submit_page(page, size, sort, tree, columns, context, previous, filter_text):
        if not context:
            return no_update, True, no_update
        pid = context['pipeline_id']
        if previous:
            try:
                service.jobs.cancel(pid, previous['id'])
            except KeyError:
                pass
        if filter_text and not tree:
            return None, True, 'Invalid filter expression.'
        try:
            job = service.jobs.submit(pid, 'query', {'artifact_id': context['artifact_id'], 'row_filters': context.get('row_filters'),
                'query': {'page': page or 0, 'page_size': size or 100, 'sort_by': sort,
                          'filter_tree': tree, 'columns': columns}})
            return {'id': job['id'], 'pipeline_id': pid}, False, 'Loading page…'
        except Exception as exc:
            return None, True, str(exc)

    @app.callback(Output('result-table', 'data'), Output('result-table', 'columns'),
                  Output('result-table', 'page_count'), Output('table-query-status', 'children'),
                  Output('table-query-poll', 'disabled'),
                  Input('table-query-poll', 'n_intervals'), State('table-query-job', 'data'),
                  prevent_initial_call=True)
    def finish_page(_tick, current):
        if not current:
            return no_update, no_update, no_update, no_update, True
        try:
            job = service.jobs.get(current['pipeline_id'], current['id'])
        except KeyError:
            return no_update, no_update, no_update, 'Query expired. Request the page again.', True
        if job['state'] not in TERMINAL:
            return no_update, no_update, no_update, no_update, False
        if job['state'] != 'succeeded':
            return no_update, no_update, no_update, job.get('error') or job['stage'], True
        result = job['result']
        # Preserve scientific 'id' columns; physical selection identifiers travel
        # separately in the query API rather than overwriting user data.
        return result['rows'], [{'name': c, 'id': c} for c in result['columns']], result['page_count'], \
            f"{result['row_count']:,} matching rows", True

    @app.callback(Output('result-table', 'page_current'),
                  Input('result-table', 'sort_by'), Input('result-table', 'filter_query'),
                  Input('table-columns', 'value'), prevent_initial_call=True)
    def reset_page(_sort, _filter, _columns):
        return 0
