/* Debounce locally; only the newest range/frame response may replace the graph. */
(function () {
    'use strict';
    let pending = null;
    const nu = () => window.dash_clientside.no_update;
    const graph = () => document.getElementById('canvas-content')?.querySelector('.js-plotly-plot');
    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        reaxkitPlots: {
            request: function (context, event, frame, refresh, previous) {
                if (!context) return nu();
                const same = previous && previous.key === context.key;
                let range = same ? (pending?.next?.key === context.key ? pending.next.x_range : previous.x_range) : null;
                const kind = context.mapping.kind;
                if (same && event && kind === 'plot2d') {
                    let bounds = event['xaxis.range'];
                    if (event['xaxis.range[0]'] !== undefined && event['xaxis.range[1]'] !== undefined)
                        bounds = [event['xaxis.range[0]'], event['xaxis.range[1]']];
                    if (event['xaxis.autorange']) range = null;
                    else if (bounds) {
                        range = bounds.map(Number);
                        if (graph()?.layout?.xaxis?.type === 'log') range = range.map(v => Math.pow(10, v));
                        if (!range.every(Number.isFinite)) return nu();
                        range.sort((a, b) => a - b);
                    }
                }
                const width = Math.round((graph()?.clientWidth || 1000) / 64) * 64;
                const next = {key: context.key, x_range: range, frame: frame ?? null, width, refresh: refresh || 0};
                if (pending?.next && ['key', 'x_range', 'frame', 'width', 'refresh'].every(k => JSON.stringify(pending.next[k]) === JSON.stringify(next[k])))
                    return nu();
                if (pending) { clearTimeout(pending.timer); pending.resolve(nu()); pending = null; }
                if (same && ['x_range', 'frame', 'width', 'refresh'].every(k => JSON.stringify(previous[k]) === JSON.stringify(next[k])))
                    return nu(); // Ignore camera/y-only/automatic relayout events.
                return new Promise(resolve => {
                    pending = {resolve, next, timer: setTimeout(() => {
                        pending = null;
                        next.request_id = Date.now().toString(36) + Math.random().toString(36).slice(2);
                        resolve(next);
                    }, same ? 180 : 0)};
                });
            },
            publish: function (response, request, previous) {
                if (!request) return [nu(), nu(), true];
                if (!response || response.key !== request.key || response.request_id !== request.request_id)
                    return [nu(), 'Preparing full-result view... Previous view remains available.', false];
                if (response.pending) return [nu(), response.status, false];
                if (response.error) return [nu(), response.error, true];
                const fig = response.figure;
                if (!fig) return [nu(), nu(), true];
                const live = graph();
                if (previous?.layout?.uirevision === fig.layout.uirevision && live) {
                    if (live.layout?.scene?.camera && fig.layout.scene)
                        fig.layout.scene.camera = JSON.parse(JSON.stringify(live.layout.scene.camera));
                    for (const axis of ['xaxis', 'yaxis']) {
                        const old = live.layout?.[axis];
                        if (old?.range && old.autorange === false && fig.layout[axis]) {
                            fig.layout[axis].range = old.range.slice();
                            fig.layout[axis].autorange = false;
                        }
                    }
                    for (const trace of fig.data) {
                        const old = live.data?.find(t => t.uid === trace.uid);
                        if (!old) continue;
                        if (old.visible !== undefined) trace.visible = old.visible;
                        if (Array.isArray(old.selectedpoints) && old.ids && trace.ids) {
                            const selected = new Set(old.selectedpoints.map(i => old.ids[i]));
                            trace.selectedpoints = trace.ids.map((id, i) => selected.has(id) ? i : -1).filter(i => i >= 0);
                        }
                    }
                }
                window.ReaxkitWorkspace?.themeFigure(fig);
                return [fig, response.status, true];
            }
        }
    });
}());
