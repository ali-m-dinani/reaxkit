/* Local panel geometry, keyboard access, versioned preferences and plot theming.
 * The pure functions are exported for Node tests; no Python calls during drag.
 */
(function (scope) {
    'use strict';
    const VERSION = 1, STORAGE = 'reaxkit.workspace.v1';
    const DEFAULT = {version: VERSION, sidebarWidth: 340, hierarchyShare: .4, drawerHeight: 170,
        sidebarOpen: true, hierarchyOpen: true, propertiesOpen: true, drawerOpen: true,
        drawerTab: 'jobs', maximized: false, theme: 'dark', preset: 'analysis'};
    const finite = (v, fallback) => typeof v === 'number' && Number.isFinite(v) ? v : fallback;
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(Math.max(lo, hi), v));
    function restore(raw) {
        const value = raw && raw.version === VERSION ? raw : {};
        const state = {...DEFAULT};
        for (const key of ['sidebarWidth', 'hierarchyShare', 'drawerHeight']) state[key] = finite(value[key], state[key]);
        for (const key of ['sidebarOpen', 'hierarchyOpen', 'propertiesOpen', 'drawerOpen', 'maximized'])
            if (typeof value[key] === 'boolean') state[key] = value[key];
        if (!state.hierarchyOpen && !state.propertiesOpen) state.hierarchyOpen = true;
        if (['jobs', 'results', 'logs'].includes(value.drawerTab)) state.drawerTab = value.drawerTab;
        if (['light', 'dark'].includes(value.theme)) state.theme = value.theme;
        if (['analysis', 'visualization', 'results', 'classic'].includes(value.preset)) state.preset = value.preset;
        state.sidebarWidth = clamp(state.sidebarWidth, 240, 1200);
        state.hierarchyShare = clamp(state.hierarchyShare, .1, .9);
        state.drawerHeight = clamp(state.drawerHeight, 70, 1200);
        return state;
    }
    function geometry(state, width, height) {
        width = Math.max(320, finite(width, 1366)); height = Math.max(140, finite(height, 600));
        const compact = width < 780, sideMax = compact ? width - 38 : Math.max(240, Math.min(width * .55, width - 340));
        const hierarchyMin = Math.min(110, (height - 8) * .35), hierarchyMax = Math.max(hierarchyMin, height - 8 - Math.min(160, height * .4));
        const drawerMin = Math.min(90, height * .3), drawerMax = Math.max(drawerMin, height - 8 - Math.min(200, height * .5));
        return {compact, sidebar: clamp(state.sidebarWidth, Math.min(260, sideMax), sideMax),
            hierarchy: clamp(state.hierarchyShare * (height - 8), hierarchyMin, hierarchyMax),
            drawer: clamp(state.drawerHeight, drawerMin, drawerMax),
            limits: {sidebar: [Math.min(260, sideMax), sideMax], hierarchy: [hierarchyMin, hierarchyMax], drawer: [drawerMin, drawerMax]}};
    }
    function preset(name, state, height = 600) {
        const next = {...DEFAULT, theme: state.theme, preset: name};
        if (name === 'visualization') Object.assign(next, {sidebarOpen: false, drawerOpen: false});
        else if (name === 'results') Object.assign(next, {drawerTab: 'results', drawerHeight: height * .48, sidebarWidth: 300});
        else if (name === 'classic') Object.assign(next, {sidebarWidth: 320, hierarchyShare: Math.max(.2, 1 - 268 / height), drawerOpen: false});
        return restore(next);
    }
    function resize(state, name, value, width, height) {
        const bounds = geometry(state, width, height).limits[name];
        if (!bounds) return state;
        value = clamp(value, ...bounds);
        const next = {...state};
        if (name === 'sidebar') next.sidebarWidth = value;
        if (name === 'hierarchy') next.hierarchyShare = value / Math.max(1, height - 8);
        if (name === 'drawer') next.drawerHeight = value;
        return next;
    }
    function themePatch(theme, is3d) {
        const dark = theme !== 'light', ink = dark ? '#e3edf7' : '#183049', bg = dark ? '#111c2b' : '#ffffff', grid = dark ? '#2b3a50' : '#dce5ed';
        const patch = {'paper_bgcolor': bg, 'plot_bgcolor': bg, 'font.color': ink, 'title.font.color': ink,
            'legend.font.color': ink, 'legend.bgcolor': bg};
        for (const axis of is3d ? ['scene.xaxis', 'scene.yaxis', 'scene.zaxis'] : ['xaxis', 'yaxis']) {
            Object.assign(patch, {[axis + '.color']: ink, [axis + '.gridcolor']: grid,
                [axis + '.linecolor']: grid, [axis + '.tickcolor']: ink, [axis + '.tickfont.color']: ink, [axis + '.title.font.color']: ink,
                [axis + '.zerolinecolor']: grid});
            if (is3d) patch[axis + '.backgroundcolor'] = bg;
        }
        return patch;
    }
    const api = {VERSION, STORAGE, DEFAULT, restore, geometry, preset, resize, themePatch};
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    if (!scope.document) return;
    const doc = scope.document;
    let state = {...DEFAULT};
    try { state = restore(JSON.parse(scope.localStorage.getItem(STORAGE))); } catch (_) {}
    doc.documentElement.dataset.rkTheme = state.theme;
    let root, body, canvas, drag = null, frame = 0, lastResize = 0, resizeTimer = 0;
    const plots = () => Array.from(doc.querySelectorAll('#canvas-content .js-plotly-plot'));
    const bounds = () => ({width: body.clientWidth, height: body.clientHeight});
    const announce = text => { const el = doc.getElementById('workspace-announcement'); if (el) el.textContent = text; };
    function publish() {
        try { scope.localStorage.setItem(STORAGE, JSON.stringify(state)); } catch (_) { announce('Layout changed. Browser storage is unavailable, so preferences will last for this session.'); }
        // A tiny state update on completed actions only. Analysis data is untouched.
        try { scope.dash_clientside?.set_props('workspace-state', {data: {...state}}); } catch (_) {}
        try {
            scope.dash_clientside?.set_props('activity-tick', {disabled: !state.drawerOpen || state.maximized || state.drawerTab !== 'jobs'});
            scope.dash_clientside?.set_props('log-refresh-tick', {disabled: !state.drawerOpen || state.maximized || state.drawerTab !== 'logs'});
        } catch (_) {}
    }
    function schedulePlotResize(settled = false) {
        if (settled) { clearTimeout(resizeTimer); resizeTimer = 0; }
        if (resizeTimer) return;
        resizeTimer = setTimeout(() => {
            resizeTimer = 0;
            if (drag && performance.now() - lastResize < 48) { schedulePlotResize(); return; }
            lastResize = performance.now();
            for (const graph of plots()) if (graph.clientWidth && graph.clientHeight && graph.layout && scope.Plotly) {
                try { Promise.resolve(scope.Plotly.Plots.resize(graph)).catch(() => {}); } catch (_) {}
            }
        }, settled ? 0 : 48);
    }
    function apply() {
        if (!root) return;
        const {width, height} = bounds(), g = geometry(state, width, height);
        root.style.setProperty('--rk-sidebar', `${g.sidebar}px`);
        root.style.setProperty('--rk-hierarchy', `${g.hierarchy}px`);
        root.style.setProperty('--rk-drawer', `${g.drawer}px`);
        root.dataset.compact = String(g.compact);
        for (const key of ['sidebarOpen', 'hierarchyOpen', 'propertiesOpen', 'drawerOpen', 'maximized', 'drawerTab']) root.dataset[key] = String(state[key]);
        doc.documentElement.dataset.rkTheme = state.theme;
        for (const name of ['sidebar', 'hierarchy', 'drawer']) {
            const splitter = doc.getElementById(`splitter-${name}`), [min, max] = g.limits[name];
            if (!splitter) continue;
            splitter.setAttribute('aria-valuemin', String(Math.round(min)));
            splitter.setAttribute('aria-valuemax', String(Math.round(max)));
            splitter.setAttribute('aria-valuenow', String(Math.round(g[name])));
            splitter.setAttribute('aria-valuetext', `${Math.round(g[name])} pixels`);
            const hidden = state.maximized || (name === 'sidebar' && (!state.sidebarOpen || g.compact)) ||
                (name === 'hierarchy' && (!state.sidebarOpen || !state.hierarchyOpen || !state.propertiesOpen)) || (name === 'drawer' && !state.drawerOpen);
            splitter.tabIndex = hidden ? -1 : 0;
            splitter.setAttribute('aria-disabled', String(hidden));
        }
        doc.getElementById('workspace-sidebar').inert = !state.sidebarOpen || state.maximized;
        doc.getElementById('panel-drawer').inert = state.maximized;
        for (const el of doc.querySelectorAll('[data-rk-action]')) {
            const action = el.dataset.rkAction;
            const key = {sidebar: 'sidebarOpen', hierarchy: 'hierarchyOpen', properties: 'propertiesOpen', drawer: 'drawerOpen'}[action];
            if (key) el.setAttribute('aria-expanded', String(state[key] && !state.maximized));
            if (action === 'maximize') el.setAttribute('aria-pressed', String(state.maximized));
            if (action === 'theme') {
                el.setAttribute('aria-pressed', String(state.theme === 'dark'));
                el.setAttribute('aria-label', `Switch to ${state.theme === 'dark' ? 'light' : 'dark'} theme`);
                el.title = el.getAttribute('aria-label');
            }
            if (action.startsWith('tab-') && el.getAttribute('role') === 'tab') {
                const selected = action.slice(4) === state.drawerTab;
                el.setAttribute('aria-selected', String(selected)); el.tabIndex = selected ? 0 : -1;
            }
        }
        doc.getElementById('workspace-preset').value = state.preset;
        schedulePlotResize();
    }
    function commit(message) { apply(); publish(); schedulePlotResize(true); if (message) announce(message); }
    function finish(event, cancel = false) {
        if (!drag || (event.pointerId !== undefined && event.pointerId !== drag.pointerId)) return;
        const previous = drag; drag = null;
        if (frame) { cancelAnimationFrame(frame); frame = 0; }
        if (cancel) state = previous.state;
        else if (previous.next) state = previous.next;
        doc.body.classList.remove('rk-dragging'); doc.body.style.cursor = '';
        if (previous.handle.hasPointerCapture?.(previous.pointerId)) previous.handle.releasePointerCapture(previous.pointerId);
        commit(cancel ? 'Resize cancelled.' : `${previous.name} resized.`);
    }
    doc.addEventListener('pointerdown', event => {
        const handle = event.target.closest?.('[data-rk-splitter]');
        if (!root || !handle || event.button !== 0 || handle.getAttribute('aria-disabled') === 'true') return;
        event.preventDefault(); handle.focus();
        const {width, height} = bounds(), g = geometry(state, width, height), name = handle.dataset.rkSplitter;
        drag = {handle, name, pointerId: event.pointerId, state: {...state}, width, height,
            start: name === 'sidebar' ? event.clientX : event.clientY, value: g[name]};
        handle.setPointerCapture(event.pointerId);
        doc.body.classList.add('rk-dragging'); doc.body.style.cursor = name === 'sidebar' ? 'col-resize' : 'row-resize';
    });
    doc.addEventListener('pointermove', event => {
        if (!drag || event.pointerId !== drag.pointerId) return;
        const coordinate = drag.name === 'sidebar' ? event.clientX : event.clientY;
        drag.next = resize(drag.state, drag.name, drag.value + (coordinate - drag.start) * (drag.name === 'drawer' ? -1 : 1), drag.width, drag.height);
        if (!frame) frame = requestAnimationFrame(() => { frame = 0; if (drag?.next) { state = drag.next; apply(); } });
    });
    doc.addEventListener('pointerup', event => finish(event));
    doc.addEventListener('pointercancel', event => finish(event, true));
    doc.addEventListener('lostpointercapture', event => finish(event, true));
    doc.addEventListener('keydown', event => {
        if (drag && event.key === 'Escape') { event.preventDefault(); finish(event, true); return; }
        const handle = event.target.closest?.('[data-rk-splitter]');
        if (handle && handle.getAttribute('aria-disabled') !== 'true') {
            const name = handle.dataset.rkSplitter, {width, height} = bounds(), g = geometry(state, width, height);
            let value = g[name], step = event.shiftKey ? 48 : 16, handled = true;
            if (event.key === 'Home') value = g.limits[name][0];
            else if (event.key === 'End') value = g.limits[name][1];
            else if (event.key === 'Enter') value = geometry(DEFAULT, width, height)[name];
            else if (name === 'sidebar' && ['ArrowLeft','ArrowRight'].includes(event.key)) value += event.key === 'ArrowRight' ? step : -step;
            else if (name !== 'sidebar' && ['ArrowUp','ArrowDown'].includes(event.key)) value += (event.key === 'ArrowDown' ? step : -step) * (name === 'drawer' ? -1 : 1);
            else handled = false;
            if (handled) { event.preventDefault(); state = resize(state, name, value, width, height); commit(`${name}: ${Math.round(value)} pixels.`); }
        }
        const tab = event.target.closest?.('[role="tab"]');
        if (tab && ['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) {
            const tabs = Array.from(tab.parentElement.querySelectorAll('[role="tab"]'));
            const index = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length-1 : (tabs.indexOf(tab) + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
            event.preventDefault(); tabs[index].focus(); tabs[index].click();
        }
        const treeNode = event.target.closest?.('.rk-tree-node');
        if (treeNode && ['ArrowUp','ArrowDown','Home','End'].includes(event.key)) {
            const nodes = Array.from(treeNode.parentElement.querySelectorAll('.rk-tree-node'));
            const index = event.key === 'Home' ? 0 : event.key === 'End' ? nodes.length-1 : clamp(nodes.indexOf(treeNode) + (event.key === 'ArrowDown' ? 1 : -1), 0, nodes.length-1);
            event.preventDefault(); nodes[index].focus();
        }
    });
    function runAction(action, value) {
        if (!root) return;
        const {height} = bounds();
        if (action === 'preset') state = preset(value, state, height);
        else if (action === 'reset') state = preset('analysis', state, height);
        else if (action === 'theme') state.theme = state.theme === 'dark' ? 'light' : 'dark';
        else if (action === 'sidebar') { state.sidebarOpen = !state.sidebarOpen; state.maximized = false; }
        else if (action === 'hierarchy') { state.hierarchyOpen = !state.hierarchyOpen; if (!state.hierarchyOpen) state.propertiesOpen = true; }
        else if (action === 'properties') { state.propertiesOpen = !state.propertiesOpen; if (!state.propertiesOpen) state.hierarchyOpen = true; }
        else if (action === 'drawer') { state.drawerOpen = !state.drawerOpen; state.maximized = false; }
        else if (action === 'maximize') state.maximized = !state.maximized;
        else if (action === 'analysis') { state.maximized = false; state.sidebarOpen = true; }
        else if (action.startsWith('tab-') && ['jobs','results','logs'].includes(action.slice(4))) {
            state.drawerTab = action.slice(4); state.drawerOpen = true; state.maximized = false;
        } else return;
        commit(action === 'theme' ? `${state.theme} theme.` : 'Workspace layout updated.');
        if (action === 'theme') for (const graph of plots()) if (graph.layout?.meta?.workspace_theme && scope.Plotly) {
            try { Promise.resolve(scope.Plotly.relayout(graph, themePatch(state.theme, !!graph.layout.scene))).catch(() => {}); } catch (_) {}
        }
    }
    doc.addEventListener('click', event => {
        const el = event.target.closest?.('[data-rk-action]');
        if (el && el.tagName !== 'SELECT') runAction(el.dataset.rkAction);
    });
    doc.addEventListener('change', event => {
        const el = event.target.closest?.('select[data-rk-action="preset"]'); if (el) runAction('preset', el.value);
    });
    api.themeFigure = function (figure) {
        if (!figure?.layout?.meta?.workspace_theme) return figure;
        for (const [key, value] of Object.entries(themePatch(state.theme, !!figure.layout.scene))) {
            const parts = key.split('.'); let target = figure.layout;
            for (const part of parts.slice(0,-1)) target = target[part] || (target[part] = {});
            target[parts.at(-1)] = value;
        }
        return figure;
    };
    scope.ReaxkitWorkspace = api;
    function boot() {
        if (root) return;
        root = doc.getElementById('rk-workspace'); if (!root) return;
        body = doc.getElementById('workspace-body'); canvas = doc.getElementById('panel-canvas');
        if (!body || !canvas) { root = null; return; }
        const observer = new ResizeObserver(() => { if (!drag) apply(); schedulePlotResize(); });
        observer.observe(body); observer.observe(canvas);
        apply(); setTimeout(publish, 100);
    }
    new MutationObserver(records => {
        if (!root) { boot(); return; }
        if (records.some(record => Array.from(record.addedNodes).some(node => node.nodeType === 1 &&
            (node.classList?.contains('dash-graph') || node.querySelector?.('.dash-graph'))))) schedulePlotResize();
    }).observe(doc.documentElement, {childList: true, subtree: true});
    boot();
})(typeof window !== 'undefined' ? window : globalThis);
