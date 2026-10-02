"""Small status polling and atomic UI publication of completed jobs."""
from dash import Input, Output, State, html, no_update

from reaxkit.presentation.specs import ensure_presentation_spec, spec_to_dash_request
from reaxkit.webui.backend.jobs import TERMINAL


def register_job_callbacks(app, service):
    @app.callback(Output('canvas-preview-notice', 'children'),
                  Input('session-store', 'data'), Input('pipeline-store', 'data'))
    def preview_notice(session, snapshot):
        from reaxkit.webui.ui.analysis.callback_helpers import _resolve_canvas_source_node, _find_source_artifact
        from reaxkit.webui.backend.artifact_tables import table_descriptor
        node = _resolve_canvas_source_node(snapshot, session, None)
        if not node:
            return ''
        stale = 'Previous result; inputs changed or work is running. ' if node.get('status') in ('dirty', 'running') else ''
        if node.get('kind') == 'utility' or (node.get('request') or {}).get('visualization_type') == 'table':
            return stale
        artifact = _find_source_artifact(snapshot, node['id'], {})
        descriptor = table_descriptor((artifact or {}).get('payload', {}))
        if descriptor:
            return stale + 'Plot images save the displayed quality. Use table exports for complete scientific data.'
        return stale

    @app.callback(Output('job-poll', 'disabled', allow_duplicate=True),
                  Input('status-banner', 'children'), prevent_initial_call=True)
    def wake_jobs(_status):
        return False

    @app.callback(Output('job-status', 'children'), Output('job-seen-store', 'data'),
                  Output('pipeline-store', 'data', allow_duplicate=True),
                  Output('result-store', 'data', allow_duplicate=True),
                  Output('job-poll', 'disabled', allow_duplicate=True),
                  Input('job-poll', 'n_intervals'), Input('btn-cancel-jobs', 'n_clicks'),
                  State('session-store', 'data'), State('job-seen-store', 'data'),
                  prevent_initial_call=True)
    def poll_jobs(_tick, _cancel, session, seen):
        from dash import ctx
        if not session or not session.get('pipeline_id'):
            return no_update, no_update, no_update, no_update, True
        pid = session['pipeline_id']
        jobs = service.jobs.list(pid)
        if ctx.triggered_id == 'btn-cancel-jobs':
            for job in jobs:
                if job['state'] not in TERMINAL:
                    service.jobs.cancel(pid, job['id'])
            jobs = service.jobs.list(pid)
        known = set(seen or [])
        completed = [service.jobs.get(pid, j['id']) for j in jobs if j['state'] in TERMINAL and j['id'] not in known]
        changed = False
        for job in completed:
            known.add(job['id'])
            if job['operation'] == 'apply':
                changed = True
            if job['state'] != 'succeeded' or job['operation'] not in ('apply', 'load', 'probe', 'import_snapshot'):
                continue
            changed = True
            if job['operation'] == 'import_snapshot':
                from reaxkit.webui.ui.analysis import callback_helpers as helpers
                for cache in (helpers._ARTIFACT_OBJ_CACHE, helpers._ARTIFACT_ROWS_CACHE,
                              helpers._PLOT_ROWS_CACHE, helpers._FIGURE_CACHE, helpers._NODE_PIPELINE_CACHE):
                    cache.clear()
            if job['operation'] != 'apply':
                continue
            result = job.get('result') or {}
            node = result.get('node') or {}
            artifact = result.get('artifact') or {}
            snapshot = service.get_pipeline(pid)
            if node.get('kind') != 'analysis' or node.get('id') not in snapshot['nodes']:
                continue
            children = snapshot['children'].get(node['id'], [])
            if any(snapshot['nodes'][c]['kind'] == 'visualization' for c in children):
                continue
            for rec in artifact.get('recommended_views', []):
                spec = ensure_presentation_spec(rec)
                request = spec_to_dash_request(spec or rec)
                vtype = request.get('visualization_type', 'plot2d')
                service.add_node(pid, {'parent_id': node['id'], 'kind': 'visualization',
                    'name': (spec.label if spec else rec.get('label')) or vtype,
                    'metadata': {'visualization_type': vtype, 'auto_recommended': True, 'presentation_spec': rec},
                    'request': request})
        active = [j for j in jobs if j['state'] not in TERMINAL]
        if active:
            status = f"{len(active)} job(s): {active[0]['stage']}"
        elif completed:
            last = completed[-1]
            if last.get('error'):
                status = html.Details([html.Summary('Job failed'), html.Pre(last['error'])])
            elif last['operation'] in ('export_table', 'save_snapshot', 'export_bundle') and last['state'] == 'succeeded':
                status = f"Saved {last['result'].get('path') or last['result'].get('bundle_dir')}"
            else:
                status = last['stage']
        else:
            status = no_update
        snapshot = service.get_pipeline(pid) if changed else no_update
        result_refs = {nid: n['result_ref'] for nid, n in snapshot['nodes'].items()
                       if n.get('result_ref')} if changed else no_update
        return status, list(known)[-100:], snapshot, result_refs, not active
