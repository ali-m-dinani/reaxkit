const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const filename = path.join(__dirname, '../../src/reaxkit/webui/assets/workspace.js');
const workspace = require(filename);

test('versioned preferences reject invalid values and keep one editor visible', () => {
    assert.deepEqual(workspace.restore({version: 999, sidebarWidth: 500}), workspace.DEFAULT);
    const state = workspace.restore({version: 1, sidebarWidth: Infinity, hierarchyShare: -1,
        drawerHeight: 1e10, theme: 'invalid', hierarchyOpen: false, propertiesOpen: false});
    assert.equal(state.sidebarWidth, 340);
    assert.equal(state.hierarchyShare, .1);
    assert.equal(state.drawerHeight, 1200);
    assert.equal(state.theme, 'dark');
    assert.equal(state.hierarchyOpen, true);
});

test('layout bounds remain usable at target sizes and 125–200% scaling', () => {
    for (const [width, height] of [[1366,768], [1920,1080]]) for (const scale of [1, 1.25, 1.5, 2]) {
        const w = width / scale - 20, h = height / scale - 100;
        for (const name of ['sidebar', 'hierarchy', 'drawer']) for (const value of [-1e9, 1e9]) {
            const state = workspace.resize(workspace.DEFAULT, name, value, w, h);
            const geometry = workspace.geometry(state, w, h);
            assert.ok(geometry[name] >= geometry.limits[name][0]);
            assert.ok(geometry[name] <= geometry.limits[name][1] + 1e-8);
            assert.ok(geometry.hierarchy + 8 < h);
            assert.ok(geometry.drawer + 8 < h);
            if (!geometry.compact) assert.ok(w - geometry.sidebar - 8 >= 320);
        }
    }
});

test('presets preserve theme and restore familiar and results layouts', () => {
    const state = {...workspace.DEFAULT, theme: 'light'};
    const viz = workspace.preset('visualization', state);
    assert.equal(viz.sidebarOpen, false);
    assert.equal(viz.drawerOpen, false);
    const results = workspace.preset('results', viz, 800);
    assert.equal(results.theme, 'light');
    assert.equal(results.drawerTab, 'results');
    assert.equal(results.drawerHeight, 384);
    assert.equal(workspace.preset('classic', state).sidebarWidth, 320);
});

function browser() {
    const listeners = {}, elements = {}, frames = new Map(), timers = new Map(), writes = [], saved = [];
    let next = 1;
    function el(id, dataset = {}) {
        return elements[id] = {id, dataset, style: {setProperty() {}}, clientWidth: 1300, clientHeight: 620,
            attrs: {}, classList: {add() {}, remove() {}}, setAttribute(k,v) { this.attrs[k] = v; },
            getAttribute(k) { return this.attrs[k]; }, focus() {}, closest(selector) {
                return (selector === '[data-rk-splitter]' && this.dataset.rkSplitter) ||
                    (selector === '[data-rk-action]' && this.dataset.rkAction) ? this : null;
            },
            setPointerCapture(id) { this.capture = id; }, hasPointerCapture(id) { return this.capture === id; },
            releasePointerCapture() { this.capture = null; }};
    }
    for (const id of ['rk-workspace','workspace-body','panel-canvas','workspace-sidebar','panel-drawer','workspace-preset','workspace-announcement']) el(id);
    for (const name of ['sidebar','hierarchy','drawer']) el('splitter-' + name, {rkSplitter: name});
    const graph = {clientWidth: 900, clientHeight: 400, layout: {meta: {workspace_theme: true}, xaxis: {range: [2,9]}}};
    const document = {documentElement: {dataset: {}}, body: el('body'), getElementById: id => elements[id],
        querySelectorAll: selector => selector.includes('.js-plotly-plot') ? [graph] : [],
        addEventListener: (name, callback) => listeners[name] = callback};
    const window = {document, localStorage: {getItem: () => null, setItem: (_key,value) => saved.push(JSON.parse(value))},
        dash_clientside: {set_props: (id,value) => writes.push({id,value})},
        Plotly: {Plots: {resize: () => {}}, relayout: (_graph,patch) => { graph.patch = patch; }}};
    const context = {window, performance: {now: () => 100}, ResizeObserver: class {observe() {}}, MutationObserver: class {observe() {}},
        setTimeout: fn => { const id = next++; timers.set(id,fn); return id; }, clearTimeout: id => timers.delete(id),
        requestAnimationFrame: fn => { const id = next++; frames.set(id,fn); return id; }, cancelAnimationFrame: id => frames.delete(id)};
    vm.runInNewContext(fs.readFileSync(filename, 'utf8'), context);
    const flush = queue => { const work = [...queue.values()]; queue.clear(); work.forEach(fn => fn()); };
    flush(timers); writes.length = 0; saved.length = 0;
    function fire(name, target, properties = {}) { listeners[name]({target, preventDefault() {}, ...properties}); }
    return {window, elements, writes, saved, graph, frames, timers, flush, fire,
        action: name => fire('click', el('action', {rkAction: name}))};
}

test('pointer dragging coalesces frames and publishes only on release; cancellation restores', () => {
    const b = browser(), handle = b.elements['splitter-sidebar'];
    b.fire('pointerdown', handle, {button: 0, pointerId: 7, clientX: 340});
    for (let x = 341; x < 500; x++) b.fire('pointermove', handle, {pointerId: 7, clientX: x});
    assert.equal(b.frames.size, 1);
    b.flush(b.frames);
    assert.equal(b.writes.length, 0);
    assert.equal(b.saved.length, 0);
    b.fire('pointerup', handle, {pointerId: 7});
    assert.equal(b.writes.filter(w => w.id === 'workspace-state').length, 1);
    assert.equal(b.saved.at(-1).sidebarWidth, 499);
    b.fire('pointerdown', handle, {button: 0, pointerId: 8, clientX: 499});
    b.fire('pointermove', handle, {pointerId: 8, clientX: 600}); b.flush(b.frames);
    b.fire('keydown', handle, {key: 'Escape'});
    assert.equal(b.saved.at(-1).sidebarWidth, 499);
});

test('keyboard resize, maximize restore, collapse and hidden polling work locally', () => {
    const b = browser();
    b.fire('keydown', b.elements['splitter-sidebar'], {key: 'ArrowRight', shiftKey: true});
    assert.equal(b.saved.at(-1).sidebarWidth, 388);
    b.action('maximize'); assert.equal(b.saved.at(-1).maximized, true);
    b.action('maximize'); assert.equal(b.saved.at(-1).sidebarWidth, 388);
    b.action('drawer'); assert.equal(b.saved.at(-1).drawerOpen, false);
    assert.equal(b.writes.filter(w => w.id === 'activity-tick').at(-1).value.disabled, true);
    b.action('tab-logs'); assert.equal(b.saved.at(-1).drawerOpen, true);
    assert.equal(b.writes.filter(w => w.id === 'log-refresh-tick').at(-1).value.disabled, false);
});

test('application theme preserves scientific styling, camera and explicit plot templates', () => {
    const b = browser();
    const fig = {data: [{marker: {color: [1,2,3]}}], layout: {meta: {workspace_theme: true},
        scene: {camera: {eye: {x: 2}}}, xaxis: {range: [2,9]}}};
    b.window.ReaxkitWorkspace.themeFigure(fig);
    assert.deepEqual(fig.data[0].marker.color, [1,2,3]);
    assert.equal(fig.layout.scene.camera.eye.x, 2);
    assert.deepEqual(fig.layout.xaxis.range, [2,9]);
    assert.equal(fig.layout.paper_bgcolor, '#111c2b');
    b.action('theme'); assert.equal(b.graph.patch.paper_bgcolor, '#ffffff');
    const explicit = {layout: {meta: {workspace_theme: false}, paper_bgcolor: 'pink'}};
    b.window.ReaxkitWorkspace.themeFigure(explicit);
    assert.equal(explicit.layout.paper_bgcolor, 'pink');
});
