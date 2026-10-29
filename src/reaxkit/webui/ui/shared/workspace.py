"""Small, reusable workstation components. Interaction lives in workspace.js."""
from dash import html


def icon(name):
    return html.Span(className=f'rk-icon rk-icon-{name}', **{'aria-hidden': 'true'})


def action(label, action_name, *, icon_name=None, **kwargs):
    return html.Button([icon(icon_name)] if icon_name else label,
        title=label, className='rk-tool-button',
        **{'data-rk-action': action_name, 'aria-label': label, **kwargs})


def separator(name, orientation, controls):
    return html.Div(id=f'splitter-{name}', className=f'rk-splitter rk-splitter-{orientation}',
        role='separator', tabIndex=0, title='Drag to resize. Arrow keys adjust; Home/End set limits; Enter resets.',
        **{'data-rk-splitter': name, 'aria-orientation': orientation,
           'aria-label': {'sidebar': 'Sidebar width', 'hierarchy': 'Hierarchy and parameters', 'drawer': 'Activity drawer height'}[name],
           'aria-controls': controls, 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': '50'})


def empty_state(title, message, *, kind='empty', eyebrow='WORKSPACE'):
    return html.Div([html.Div(icon('plot'), className='rk-empty-symbol'),
        html.Small(eyebrow, className='rk-eyebrow'), html.H2(title), html.P(message)],
        className=f'rk-empty-state rk-state-{kind}', role='status')


def activity_drawer(log_panel):
    return html.Section([
        html.Div([
            html.Div([html.Button(label, id=f'btn-drawer-{tab}', role='tab',
                **{'data-rk-action': f'tab-{tab}', 'aria-controls': f'drawer-{tab}', 'aria-selected': 'true' if tab == 'jobs' else 'false'})
                for tab, label in [('jobs', 'Activity'), ('results', 'Results'), ('logs', 'Logs')]],
                role='tablist', **{'aria-label': 'Workspace drawer'}),
            html.Span(id='activity-count', className='rk-count'),
            action('Collapse or restore drawer', 'drawer', icon_name='collapse', **{'aria-expanded': 'true'}),
        ], className='rk-drawer-head'),
        html.Div([
            html.Div(id='activity-jobs', children=empty_state('Ready when you are',
                'Run an analysis to see progress and completed jobs here.', eyebrow='ACTIVITY')),
        ], id='drawer-jobs', role='tabpanel', **{'aria-labelledby': 'btn-drawer-jobs'}),
        html.Div(id='drawer-results', role='tabpanel', children=html.Div(id='activity-results'),
            **{'aria-labelledby': 'btn-drawer-results'}),
        html.Div(log_panel, id='drawer-logs', role='tabpanel', **{'aria-labelledby': 'btn-drawer-logs'}),
    ], id='panel-drawer', className='rk-panel rk-drawer', **{'aria-label': 'Activity and results'})
