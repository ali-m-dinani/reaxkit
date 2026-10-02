"""Check an installed wheel, including the Windows-spawn plot worker.

Set PYTHONPATH to an isolated pip --target installation, then pass that directory
as the first argument. The assertion prevents accidentally testing editable code.
"""
from pathlib import Path
import sys
import time


def main():
    import pandas as pd
    from reaxkit.webui.backend import plot_queries
    from reaxkit.webui.backend.jobs import TERMINAL
    from reaxkit.webui.backend.schemas import PipelineNode, ResultArtifact
    from reaxkit.webui.dash_app import create_dash_app
    assert Path(sys.argv[1]).resolve() in Path(plot_queries.__file__).resolve().parents
    app = create_dash_app()
    service = app.reaxkit_service
    try:
        with app.server.test_client() as client:
            for path in ('/', '/_dash-layout', '/_dash-dependencies', '/assets/performance.js', '/assets/plot_queries.js',
                         '/assets/workspace.js', '/assets/workspace.css', '/assets/workspace-icons.css'):
                response = client.get(path)
                assert response.status_code == 200, (path, response.status_code)
            assert b'reaxkitPlots' in client.get('/assets/plot_queries.js').data
        pid = service.create_pipeline()['id']
        service.store.upsert_node(pid, PipelineNode(id='source', kind='analysis', name='Source', result_ref='result'))
        service.store.store_artifact(pid, ResultArtifact(id='result', node_id='source', payload={
            'table': pd.DataFrame({'x': range(50_000), 'y': range(50_000)})}))
        job = service.jobs.submit(pid, 'plot', {'artifact_id': 'result', 'request': {'x_col': 'x', 'y_col': 'y'},
                                              'width': 320, 'view_id': 'wheel-smoke'})
        deadline = time.monotonic() + 45
        while job['state'] not in TERMINAL and time.monotonic() < deadline:
            time.sleep(0.05)
            job = service.jobs.get(pid, job['id'])
        assert job['state'] == 'succeeded', job.get('error') or job['state']
        assert job['result']['summary']['view_rows'] == 50_000
        assert job['result']['summary']['displayed_points'] <= job['result']['summary']['budget']
        assert job['result']['figure']['layout']['meta']['summary']
        print('Installed wheel passed: imports, Dash routes, browser assets, and isolated plot worker.')
    finally:
        service.close()


if __name__ == '__main__':
    main()
