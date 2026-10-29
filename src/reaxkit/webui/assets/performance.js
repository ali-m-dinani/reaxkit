/* Opt-in local recorder: open the GUI with ?rk_profile=1. No telemetry uploads. */
(() => {
    if (!new URLSearchParams(location.search).has('rk_profile')) return;
    const records = [];
    const record = (entry) => { records.push(entry); if (records.length > 1000) records.shift(); };
    for (const type of ['longtask', 'resource']) {
        try {
            new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    if (type === 'resource' && !entry.name.includes('_dash-update-component')) continue;
                    record({type, start: entry.startTime, duration: entry.duration,
                        bytes: entry.encodedBodySize || 0});
                }
            }).observe({type, buffered: true});
        } catch (_) { /* Browser may not implement long-task observation. */ }
    }
    document.addEventListener('pointerdown', () => {
        const start = performance.now();
        requestAnimationFrame(() => requestAnimationFrame(() => record({type: 'input-to-paint', duration: performance.now() - start})));
    }, {passive: true});
    window.addEventListener('load', () => {
        const button = document.createElement('button');
        button.textContent = 'Download performance report';
        button.style.cssText = 'position:fixed;bottom:8px;right:8px;z-index:9999';
        button.addEventListener('click', () => {
            const report = {browser: navigator.userAgent, viewport: [innerWidth, innerHeight],
                memory: performance.memory ? {usedJSHeapSize: performance.memory.usedJSHeapSize} : null, records};
            const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: 'application/json'}));
            const anchor = document.createElement('a');
            anchor.href = url; anchor.download = 'reaxkit-browser-performance.json'; anchor.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        });
        document.body.appendChild(button);
    });
})();
