"""Compact workstation command bar; layout actions are entirely client-side."""
from dash import dcc, html
from reaxkit.webui.ui.shared.workspace import action


def topbar():
    return html.Div([
        html.Div([html.Span('RK', className='rk-brand-mark'),
            html.Div([html.Strong('ReaxKit'), html.Small('SCIENTIFIC WORKSPACE')])], className='rk-brand'),
        html.Div([
            html.Button('Workspace', id='btn-nav-analysis', n_clicks=0, className='rk-nav-btn active', **{'data-rk-action': 'analysis'}),
            html.Button('Logs', id='btn-nav-log', n_clicks=0, className='rk-nav-btn', **{'data-rk-action': 'tab-logs'}),
        ], className='rk-navigation'),
        html.Div([
            action('Show or hide sidebar', 'sidebar', icon_name='sidebar', **{'aria-expanded': 'true'}),
            html.Label('Layout', htmlFor='workspace-preset', className='rk-sr-only'),
            html.Select([html.Option(label, value=value) for value, label in
                [('analysis', 'Analysis layout'), ('visualization', 'Visualization layout'), ('results', 'Results layout'), ('classic', 'Classic sidebar')]],
                id='workspace-preset', title='Workspace layout preset', **{'data-rk-action': 'preset'}),
            action('Reset panel sizes', 'reset', icon_name='reset'),
            action('Switch color theme', 'theme', icon_name='theme', **{'aria-pressed': 'false'}),
        ], className='rk-workspace-tools'),
        html.Div([
            html.Span(id='job-status', className='rk-job-summary'),
            html.Button('Stop jobs', id='btn-cancel-jobs', n_clicks=0, className='rk-stop', title='Stop queued and running jobs in this pipeline'),
            dcc.Loading(id='execute-loading', type='circle',
                target_components={'execute-loading-proxy': 'children'},
                children=html.Div(id='execute-loading-proxy', className='rk-spinner-anchor')),
            html.Span(id='status-banner', className='rk-badge', role='status'),
        ], className='rk-status-wrap'),
        html.Div([
            html.Button('Help', id='help-menu-trigger', n_clicks=0, className='rk-help-trigger'),
            html.Div([
                html.A('ReaxKit documentation', href='https://ali-m-dinani.github.io/reaxkit/', target='_blank', rel='noopener noreferrer', className='rk-help-item'),
                html.A('Source & issues', href='https://github.com/ali-m-dinani/reaxkit', target='_blank', rel='noopener noreferrer', className='rk-help-item'),
                html.Button('Check for updates', id='btn-help-check-updates', n_clicks=0, className='rk-help-item rk-help-btn'),
                html.Div(id='help-update-status', className='rk-help-status'),
            ], id='help-menu-dropdown', className='rk-help-dropdown'),
        ], className='rk-help-menu'),
    ], className='rk-topbar')
