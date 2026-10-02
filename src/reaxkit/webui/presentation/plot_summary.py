"""Plotly adapter for bounded summaries; no artifact reads or decimation here."""
from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import plotly.graph_objects as go

from reaxkit.webui.presentation.perf_config import load_ui_performance_config


def summary_figure(summary, request, *, view_key):
    from reaxkit.webui.ui.analysis.callback_helpers import (
        _apply_2d_style, _apply_3d_style, _flag_on, _trace_styles_map, _trace_style_for_trace,
    )
    mapping = summary['mapping']
    kind = mapping['kind']
    fig = go.Figure()
    revision = summary['revision'] + ':' + json.dumps(mapping, sort_keys=True)
    color = request.get('line_color_rgb') or request.get('line_color')
    options = request.get('_presentation_options') or {}
    config = load_ui_performance_config()
    if kind == 'plot2d':
        styles = _trace_styles_map(request.get('trace_styles'))
        for index, trace in enumerate(summary['traces']):
            style = _trace_style_for_trace(styles, trace['name'], index)
            markers = _flag_on(style.get('show_markers', request.get('show_markers')), default=False)
            # Isolated extrema in a gap-containing bucket must remain visible.
            gaps = summary['reduced'] and any(x is None for x in trace['x'])
            line_color = style.get('line_color_rgb') or style.get('line_color') or (color if len(summary['traces']) == 1 else None)
            line = {'width': float(style.get('line_width') or request.get('line_width') or 2)}
            if line_color:
                line['color'] = line_color
            scatter = go.Scattergl if summary['displayed_points'] >= config['plot2d_scattergl_threshold'] else go.Scatter
            fig.add_trace(scatter(x=np.asarray(trace['x'], dtype=float), y=np.asarray(trace['y'], dtype=float),
                name=trace['name'], uid=sha256(trace['name'].encode()).hexdigest()[:16],
                ids=[f'{summary["revision"]}:{r}' if r is not None else '' for r in trace['row_ids']],
                customdata=[[r] for r in trace['row_ids']], connectgaps=False,
                mode='lines+markers' if markers or gaps else 'lines', line=line,
                marker={'size': float(style.get('marker_size') or request.get('marker_size') or (3 if gaps else 6))},
                hovertemplate='%{x}<br>%{y}<extra>%{fullData.name}</extra>'))
        _apply_2d_style(fig, request, apply_legend=True)
        fig.update_layout(title=options.get('title') or f"{mapping['y']} vs {mapping['x']}")
        fig.update_xaxes(title_text=request.get('x_title') or options.get('xlabel') or mapping['x'])
        fig.update_yaxes(title_text=request.get('y_title') or options.get('ylabel') or mapping['y'])
        if summary.get('x_range'):
            limits = summary['x_range']
            if fig.layout.xaxis.type == 'log':
                limits = np.log10(limits).tolist()
            fig.update_xaxes(range=limits, autorange=False)
    elif kind == 'histogram':
        bins = summary['bins']
        fig.add_trace(go.Bar(x=np.asarray([(b['left'] + b['right']) / 2 for b in bins]),
            y=np.asarray([b['count'] for b in bins], dtype=np.int64),
            width=[b['right'] - b['left'] or 1 for b in bins],
            customdata=[[b['left'], b['right'], ']' if i == len(bins) - 1 else ')'] for i, b in enumerate(bins)], marker_color=color,
            name='Exact bin counts', hovertemplate='[%{customdata[0]}, %{customdata[1]}%{customdata[2]}<br>Count: %{y}<extra></extra>'))
        _apply_2d_style(fig, request, apply_legend=True)
        fig.update_layout(bargap=0, title=options.get('title') or f"Distribution: {mapping['x']}")
        fig.update_xaxes(title_text=request.get('x_title') or mapping['x'])
        fig.update_yaxes(title_text=request.get('y_title') or 'Count')
    else:
        points = summary['points']
        marker = dict(size=float(request.get('marker_size') or 4), opacity=0.85)
        if mapping['color']:
            marker.update(color=np.asarray(points['color'], dtype=float), colorscale='Viridis',
                          colorbar={'title': mapping['color']})
        elif color:
            marker['color'] = color
        fig.add_trace(go.Scatter3d(**{a: np.asarray(points[a], dtype=np.float64) for a in ('x', 'y', 'z')},
            mode='markers', marker=marker, name='Atoms' if mapping['atom'] else 'Points', uid='frame-points',
            ids=[f'{summary["revision"]}:{r}' for r in points['row_ids']],
            customdata=[[r, a if a is not None else str(r)] for r, a in zip(points['row_ids'], points['atom_ids'])],
            hovertemplate='ID: %{customdata[1]}<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'))
        cell = summary.get('cell')
        if cell:
            corners = [np.asarray(cell['origin']) + np.asarray(bits) @ np.asarray(cell['vectors'])
                       for bits in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))]
            segments = [(corners[i], corners[i ^ bit]) for i in range(8) for bit in (1, 2, 4) if i < (i ^ bit)]
            fig.add_trace(_segments(segments, 'Periodic cell', '#7d8c9a'))
        atoms = {str(a): [points[axis][i] for axis in ('x', 'y', 'z')]
                 for i, a in enumerate(points['atom_ids']) if a is not None}
        bonds = [(atoms[a], atoms[b]) for a, b in summary.get('bonds', [])]
        if bonds:
            fig.add_trace(_segments(bonds, 'Visible bonds', '#8ba4ba'))
        _apply_3d_style(fig, request, apply_legend=True)
        fig.update_layout(title=options.get('title') or f"3D View: {mapping['x']}, {mapping['y']}, {mapping['z']}")
        fig.update_scenes(aspectmode='data', uirevision=revision,
            xaxis_title=request.get('x_title') or mapping['x'], yaxis_title=request.get('y_title') or mapping['y'],
            zaxis_title=request.get('z_title') or mapping['z'])
    if _flag_on(request.get('use_plot_title'), default=False) and request.get('plot_title'):
        fig.update_layout(title=request['plot_title'])
    fig.update_layout(autosize=True, height=None, uirevision=revision, selectionrevision=revision,
        legend_uirevision=revision, meta={'view_key': view_key, 'summary': True, 'kind': kind,
        'quality': summary_notice(summary), 'revision': summary['revision'],
        'workspace_theme': request.get('theme') in (None, '', 'workspace')})
    # Distinct defaults for new unstyled traces; explicit scientific colors win.
    if request.get('theme') in (None, '', 'workspace'):
        fig.update_layout(colorway=['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00'])
    return fig


def _segments(segments, name, color):
    coordinates = [[v for a, b in segments for v in (a[i], b[i], None)] for i in range(3)]
    return go.Scatter3d(x=coordinates[0], y=coordinates[1], z=coordinates[2], mode='lines',
                        name=name, line={'color': color, 'width': 2}, hoverinfo='skip')


def summary_notice(summary):
    kind = summary['mapping']['kind']
    if kind == 'histogram':
        return f"Exact histogram: {summary['valid_rows']:,} values; {summary['missing_rows']:,} missing/nonfinite. Final bin includes its right edge."
    if kind == 'scatter3d':
        return (f"Frame {summary['frame']}: {summary['displayed_points']:,} / {summary['valid_rows']:,} finite points. "
                f"{summary['method']}. {summary['missing_rows']:,} invalid coordinates omitted. "
                'Click a point for exact values. Bonds are limited to supplied, visible endpoints.')
    groups = f" Showing {summary['shown_groups']} of {summary['group_count']} groups." if summary['group_count'] > summary['shown_groups'] else ''
    quality = summary['method'] if summary['reduced'] else 'Full resolution in this range'
    aggregate = f" Exact {summary['aggregation']} aggregation before reduction." if summary['aggregation'] != 'none' else ''
    return (f"{quality}: {summary['displayed_points']:,} display points / {summary['view_rows']:,} range observations; "
            f"{summary['matching_rows']:,} matching source rows.{groups}{aggregate} "
            'Zoom to refine; click for exact values. Tables and exports retain all rows.')
