"""Shared style definitions for the ReaxKit web UI."""

_CSS = """
body { margin: 0; font-family: Segoe UI, Tahoma, sans-serif; background: #edf3f7; }
.rk-grid {
  display: grid; height: 100vh; gap: 8px; padding: 8px;
  grid-template-columns: 320px 1fr;
  grid-template-rows: 56px 1fr 260px 42px;
  grid-template-areas:
    "top top"
    "left canvas"
    "props results"
    "info info";
}
.rk-panel { background: #fff; border: 1px solid #cedae3; border-radius: 10px; padding: 10px; overflow: auto; }
.rk-top { grid-area: top; overflow: visible; z-index: 20; }
.rk-left { grid-area: left; }
.rk-canvas { grid-area: canvas; }
.rk-props { grid-area: props; }
.rk-results { grid-area: results; }
.rk-info { grid-area: info; }
.rk-topbar { display: flex; align-items: center; gap: 16px; }
.rk-nav-btn {
  border: none;
  background: transparent;
  color: #2f4a63;
  border-radius: 0;
  padding: 0 4px;
  margin: 0;
  cursor: pointer;
  font-size: 15px;
  line-height: 1.2;
  font-weight: 500;
  font-family: "Segoe UI", Tahoma, sans-serif;
  appearance: none;
  -webkit-appearance: none;
}
.rk-nav-btn.active {
  background: transparent;
  border: none;
  font-weight: 600;
}
.rk-help-menu { position: relative; }
.rk-help-trigger {
  border: none;
  background: transparent;
  color: #2f4a63;
  border-radius: 0;
  padding: 0 4px;
  margin: 0;
  cursor: pointer;
  display: inline-flex;
  font-size: 15px;
  line-height: 1.2;
  font-weight: 500;
  font-family: "Segoe UI", Tahoma, sans-serif;
}
.rk-help-trigger.active { font-weight: 600; }
.rk-help-dropdown {
  display: none;
  position: absolute;
  top: 34px;
  left: 0;
  min-width: 220px;
  z-index: 9999;
  background: #ffffff;
  border: 1px solid #c8d7e3;
  border-radius: 8px;
  box-shadow: 0 8px 18px rgba(0, 0, 0, 0.12);
  padding: 8px;
  gap: 4px;
}
.rk-help-item {
  display: block;
  text-decoration: none;
  color: #2f4a63;
  background: #f8fbfe;
  border: 1px solid #e0e8ef;
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 13px;
}
.rk-help-item:hover { background: #eaf3fb; }
.rk-help-btn { text-align: left; cursor: pointer; font-family: inherit; }
.rk-help-status {
  margin-top: 4px;
  color: #36526b;
  font-size: 12px;
  white-space: pre-wrap;
}
.rk-status-wrap { margin-left: auto; display: flex; align-items: center; justify-content: flex-end; gap: 8px; min-width: 120px; }
.rk-spinner-anchor { width: 18px; height: 18px; }
.rk-badge { margin-left: auto; background: #e5f2fb; border: 1px solid #b6d2e8; border-radius: 14px; padding: 2px 8px; font-size: 12px; }
.rk-badge-error { margin-left: auto; background: #fde8e8; border: 1px solid #f1b5b5; color: #8a1c1c; border-radius: 14px; padding: 2px 8px; font-size: 12px; }
.rk-badge-warn { margin-left: auto; background: #fff5dc; border: 1px solid #f0d18d; color: #7a4f00; border-radius: 14px; padding: 2px 8px; font-size: 12px; }
.rk-stack { display: grid; gap: 8px; margin-bottom: 10px; }
.rk-subtitle { margin: 8px 0 8px; font-size: 14px; color: #38536a; }
.rk-inline-actions { display: grid; grid-template-columns: repeat(6, minmax(90px, auto)); gap: 8px; align-items: center; margin: 8px 0; }
.rk-btn-save {
  width: 180px;
  height: 30px;
}
.rk-btn-exec {
  width: 90px;
  height: 30px;
}
.rk-tree {
  border: 1px solid #d4e0e8;
  background: #f9fcff;
  border-radius: 8px;
  padding: 8px;
  min-height: 190px;
  max-height: 250px;
  overflow: auto;
  display: grid;
  gap: 2px;
  margin-bottom: 10px;
}
.rk-tree-node {
  border: 1px solid transparent; border-radius: 4px; background: transparent;
  text-align: left; padding: 4px 8px; display: flex; align-items: center; gap: 8px; cursor: pointer;
  font-family: "Segoe UI", Tahoma, sans-serif;
}
.rk-tree-node:hover { background: #eef5fb; border-color: #d4e3f0; }
.rk-tree-node.selected { background: #d8e7f8; border-color: #b8cfe8; }
.rk-tree-prefix { color: #70879c; font-family: Consolas, "Courier New", monospace; white-space: pre; }
.rk-tree-icon { width: 16px; }
.rk-tree-label { font-weight: 600; color: #2f4a63; }
.rk-tree-status { margin-left: auto; color: #61778a; font-size: 12px; }
.rk-tree-meta { color: #3f5d74; padding: 2px 8px; display: flex; align-items: center; gap: 8px; font-family: "Segoe UI", Tahoma, sans-serif; }
.rk-tree-empty { color: #607788; font-size: 13px; }
.rk-help-inline { display: inline-flex; align-items: center; gap: 6px; }
.rk-help-dot {
  display: inline-flex; align-items: center; justify-content: center;
  width: 16px; height: 16px; border-radius: 50%;
  border: 1px solid #9cb2c3; color: #3d5568; font-size: 11px; font-weight: 700;
  cursor: help; background: #f2f7fb;
}
.rk-canvas-box, .rk-results-box { border: 1px dashed #bfd0de; border-radius: 8px; min-height: 140px; padding: 10px; }
.rk-canvas-head { display: flex; align-items: flex-start; justify-content: space-between; }
.rk-canvas-actions { display: grid; gap: 6px; justify-items: end; }
.rk-page-full {
  grid-column: 1 / span 2;
  grid-row: 2 / span 3;
}
.rk-log-page {
  display: grid;
  gap: 12px;
}
.rk-log-section {
  display: grid;
  gap: 6px;
}
.rk-log-name {
  color: #4a667c;
  font-size: 12px;
}
.rk-log-box {
  margin: 0;
  min-height: 180px;
  max-height: 320px;
  overflow: auto;
  border: 1px solid #d4e0e8;
  border-radius: 8px;
  padding: 10px;
  background: #f9fcff;
  white-space: pre-wrap;
  font-family: Consolas, "Courier New", monospace;
  font-size: 12px;
}
@media (max-width: 980px) {
  .rk-grid {
    grid-template-columns: 1fr;
    grid-template-rows: 56px 280px 250px 260px 220px 42px;
    grid-template-areas:
      "top"
      "left"
      "props"
      "canvas"
      "results"
      "info";
  }
}
"""
