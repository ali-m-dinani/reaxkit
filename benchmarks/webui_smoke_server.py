"""Local GUI fixture: one million result rows, without a simulation dependency."""
import atexit
import numpy as np
import pandas as pd
from reaxkit.webui.dash_app import create_dash_app
from reaxkit.webui.backend.schemas import PipelineNode, ResultArtifact


def main():
    app = create_dash_app()
    service = app.reaxkit_service
    create = service.create_pipeline
    def seeded(payload=None):
        result = create(payload)
        pid = result['id']
        store = service.store
        store.upsert_node(pid, PipelineNode(id='dataset', kind='dataset', name='Benchmark dataset',
            metadata={'dataset': {'frames': 1000, 'engine_detected': 'reaxff'}, 'run_dir': '.'}))
        store.upsert_node(pid, PipelineNode(id='analysis', kind='analysis', name='Million-row result',
            parent_id='dataset', result_ref='million', status='done'))
        store.store_artifact(pid, ResultArtifact(id='million', node_id='analysis', payload={'table': pd.DataFrame({
            'step': np.arange(1_000_000), 'value': np.sin(np.arange(1_000_000) / 100),
            'atom_id': np.arange(1_000_000) % 1000})}))
        store.upsert_node(pid, PipelineNode(id='table', kind='visualization', name='Full result table',
            parent_id='analysis', request={'visualization_type': 'table'}))
        store.upsert_node(pid, PipelineNode(id='plot', kind='visualization', name='Result plot',
            parent_id='analysis', request={'visualization_type': 'plot2d', 'x_col': 'step', 'y_col': 'value'}))
        store.upsert_node(pid, PipelineNode(id='histogram', kind='visualization', name='Exact histogram',
            parent_id='analysis', request={'visualization_type': 'histogram', 'x_col': 'value'}))
        count = 100_000
        points = np.random.default_rng(42).uniform(0, 100, (3 * count, 3))
        store.upsert_node(pid, PipelineNode(id='atoms', kind='analysis', name='Molecular viewport prototype',
            parent_id='dataset', result_ref='frames', status='done'))
        store.store_artifact(pid, ResultArtifact(id='frames', node_id='atoms', payload={
            'table': pd.DataFrame({**{a: points[:, i] for i, a in enumerate('xyz')},
                'atom_id': np.arange(3 * count) % count, 'frame_index': np.arange(3 * count) // count}),
            'cell': {'origin': [0, 0, 0], 'vectors': [[100, 0, 0], [0, 100, 0], [0, 0, 100]]},
            'bonds': [[i, i + 1] for i in range(1000)]}))
        store.upsert_node(pid, PipelineNode(id='viewport', kind='visualization', name='100k atoms per frame',
            parent_id='atoms', request={'visualization_type': 'scatter3d', 'x_col': 'x', 'y_col': 'y', 'z_col': 'z'}))
        return service.get_pipeline(pid)
    service.create_pipeline = seeded
    atexit.register(service.close)
    app.run(port=8067, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
