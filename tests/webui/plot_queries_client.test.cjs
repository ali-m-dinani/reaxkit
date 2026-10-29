// Run with node --test tests/webui/plot_queries_client.test.cjs.
// Pure client callback tests; these do not claim browser or GPU validation.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/reaxkit/webui/assets/plot_queries.js', 'utf8');

function fixture() {
    const noUpdate = {};
    const live = {clientWidth: 1024, layout: {}, data: []};
    const window = {dash_clientside: {no_update: noUpdate}};
    vm.runInNewContext(source, {window, document: {getElementById: () => ({querySelector: () => live})},
        setTimeout, clearTimeout});
    const functions = window.dash_clientside.reaxkitPlots;
    return {api: {request: (c, e, f, p, refresh = 0) => functions.request(c, e, f, refresh, p), publish: functions.publish}, live, noUpdate};
}

test('rapid zooms coalesce to newest request, including return to original range', async () => {
    const {api, noUpdate} = fixture();
    const context = {key: 'view', mapping: {kind: 'plot2d'}};
    const first = await api.request(context, null, null, null);
    const old = api.request(context, {'xaxis.range': [100, 200]}, null, first);
    const latest = api.request(context, {'xaxis.range': [800, 900]}, null, first);
    assert.equal(await old, noUpdate);
    assert.deepEqual(Array.from((await latest).x_range), [800, 900]);
    const superseded = api.request(context, {'xaxis.range': [100, 200]}, null, first);
    assert.equal(api.request(context, {'xaxis.autorange': true}, null, first), noUpdate);
    assert.equal(await superseded, noUpdate);
});

test('camera and y-only moves do not query; logarithmic ranges are data-space', async () => {
    const {api, live, noUpdate} = fixture();
    const context = {key: 'view', mapping: {kind: 'plot2d'}};
    const first = await api.request(context, null, null, null);
    assert.equal(api.request(context, {'yaxis.range': [2, 3]}, null, first), noUpdate);
    live.layout.xaxis = {type: 'log'};
    const zoom = await api.request(context, {'xaxis.range': [1, 3]}, null, first);
    assert.deepEqual(Array.from(zoom.x_range), [10, 1000]);
    const frameContext = {key: '3d', mapping: {kind: 'scatter3d'}};
    const frame = await api.request(frameContext, null, 0, null);
    assert.equal(api.request(frameContext, {'scene.camera': {}}, 0, frame), noUpdate);
    assert.equal((await api.request(frameContext, null, 1, frame)).frame, 1);
});

test('late results cannot overwrite the current view or stop its polling', () => {
    const {api, noUpdate} = fixture();
    const request = {key: 'view', request_id: 'new'};
    const stale = api.publish({key: 'view', request_id: 'old', figure: {}}, request, {});
    assert.equal(stale[0], noUpdate);
    assert.equal(stale[2], false);
    const navigated = api.publish({key: 'old-view', request_id: 'new'}, request, {});
    assert.equal(navigated[0], noUpdate);
    const error = api.publish({key: 'view', request_id: 'new', error: 'Cancelled'}, request, {});
    assert.equal(error[1], 'Cancelled');
    assert.equal(error[2], true);
});

test('publication preserves camera, zoom, legend and stable selections', () => {
    const {api, live} = fixture();
    live.layout = {scene: {camera: {eye: {x: 2, y: 1, z: 3}}}, xaxis: {range: [100, 200], autorange: false}};
    live.data = [{uid: 'trace', visible: 'legendonly', selectedpoints: [1], ids: ['r1', 'r2']}];
    const figure = {layout: {uirevision: 'revision', scene: {}, xaxis: {}}, data: [{uid: 'trace', ids: ['r2', 'r3']}]};
    const [result, status, disabled] = api.publish({key: 'view', request_id: 'new', figure, status: 'Ready'},
        {key: 'view', request_id: 'new'}, {layout: {uirevision: 'revision'}});
    assert.equal(result.layout.scene.camera.eye.x, 2);
    assert.deepEqual(Array.from(result.layout.xaxis.range), [100, 200]);
    assert.equal(result.data[0].visible, 'legendonly');
    assert.deepEqual(Array.from(result.data[0].selectedpoints), [0]);
    assert.equal(status, 'Ready');
    assert.equal(disabled, true);
});

test('refresh retries identical viewport and pending progress keeps polling', async () => {
    const {api, noUpdate} = fixture();
    const context = {key: 'view', mapping: {kind: 'plot2d'}};
    const first = await api.request(context, null, null, null);
    const retry = await api.request(context, null, null, first, 1);
    assert.notEqual(retry.request_id, first.request_id);
    const result = api.publish({key: 'view', request_id: retry.request_id, pending: true, status: 'Preparing geometry'}, retry, {});
    assert.equal(result[0], noUpdate);
    assert.equal(result[1], 'Preparing geometry');
    assert.equal(result[2], false);
});
